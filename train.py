import torch
import wandb
import time
import os
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from datasets import load_dataset

from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from trl.gpt2 import GPT2HeadWithValueModel
from trl.gptj import GPTJHeadWithValueModel
from trl.gpt_neo import GPTNeoHeadWithValueModel
from trl.ppo import PPOTrainer

tqdm.pandas()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

config = {
    "run_name": str(os.environ.get('RUN_NAME', "test-run")),
    "dataset_path": str(os.environ.get('DATASET_PATH', "ChaiML/user_model_inputs")),
    "model_name": str(os.environ.get('MODEL_NAME', "gpt2")),
    "cls_model_name": str(os.environ.get('CLS_MODEL_NAME', "ChaiML/rewardModel90kEpoch2K1M3")),
    "cls_tokenizer_name": str(os.environ.get('CLS_TOKENIZER_NAME', "roberta-large-mnli")),
    "auth_token": str(os.environ.get('AUTH_TOKEN', "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")),
    "wandb_key": str(os.environ.get('WANDB_KEY', "15f2e48c1a90469a539fd5c91c25b5f04f4f3c66")),
    "steps": int(os.environ.get('STEPS', 20000)),
    "batch_size": int(os.environ.get('BATCH_SIZE', 1)),
    "forward_batch_size": int(os.environ.get('FORWARD_BATCH_SIZE', 1)),
    "txt_out_max_len": int(os.environ.get('TEXT_OUT_MAX_LEN', 2048)),
    "ppo_epochs": int(os.environ.get('PPO_EPOCHS', 4)),
    "lr": float(os.environ.get('LR', 1.41e-5)),
    "init_kl_coef": float(os.environ.get('INIT_KL_COEF', 0.2)),
    "target": int(os.environ.get('TARGET', 6)),
    "horizon": int(os.environ.get('HORIZONT', 10000)),
    "gamma": int(os.environ.get('GAMMA', 1)),
    "lam": float(os.environ.get('LAM', 0.95)),
    "cliprange": float(os.environ.get('CLIPRANGE', 0.2)),
    "cliprange_value": float(os.environ.get('CLIPRANGE_VALUE', 0.2)),
    "vf_coef": float(os.environ.get('VF_COEF', 0.1)),
    "generation_kwargs": {
        "top_k": float(os.environ.get('TOP_K', 0.0)),
        "top_p": float(os.environ.get('TOP_P', 1.0)),
        "temperature": float(os.environ.get('TEMPERATURE', 1.0)),
        "do_sample": bool(os.environ.get('DO_SAMPLE', True)),
        "eos_token": str(os.environ.get('EOS_TOKEN', "\n")),
        "pad_token": str(os.environ.get('PAD_TOKEN', "\n")),
        "early_stopping": bool(os.environ.get('EARLY_STOPPING', True)),
        "max_new_tokens": int(os.environ.get('MAX_NEW_TOKENS', 2048)),
    },
    "sentiment_kwargs": {
        "function_to_apply": str(os.environ.get('FUNCTION_TO_APPLY', "none")),
    }
}

wandb.login(key=config["wandb_key"])
wandb.init(name=config["run_name"], project='lm-ppo', config=config)

dataset = load_dataset(config["dataset_path"], split="train", use_auth_token=config["auth_token"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    config["cls_model_name"],
    use_auth_token=config["auth_token"]
)

tokenizer = AutoTokenizer.from_pretrained(
    config["cls_tokenizer_name"],
    use_auth_token=config["auth_token"]
)

model_class = None
if "gpt2" in config["model_name"]:
    model_class = GPT2HeadWithValueModel
if "gpt-neo" in config["model_name"]:
    model_class = GPTNeoHeadWithValueModel
else:
    model_class = GPTJHeadWithValueModel

model_base = model_class.from_pretrained(config['model_name'])
model_ref = model_class.from_pretrained(config['model_name'])

tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
tokenizer.pad_token = tokenizer.eos_token

wandb.watch(model_base, log='all')

if pipe_device != -1:
    sentiment_model.to(pipe_device)
    model_base.to(device)
    model_ref.to(device)

if pipe_device != -1:
    sentiment_pipe = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=tokenizer, device=pipe_device)
else:
    sentiment_pipe = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=tokenizer)

max_position_embeddings = sentiment_pipe.model.config.max_position_embeddings

sent_kwargs = {
    "return_all_scores": True,
    "top_k": len(sentiment_pipe.model.config.id2label),
    "function_to_apply": "none",
    "batch_size": config["forward_batch_size"]
}


def tokenize(sample):
    sample["tokens"] = tokenizer.encode(sample["text"])
    sample["query"] = sample["text"]
    return sample


dataset = dataset.map(tokenize, batched=False)

gen_kwargs = config["generation_kwargs"]
eos_token = gen_kwargs.pop("eos_token")
gen_kwargs["eos_token_id"] = int(tokenizer(eos_token, return_tensors="pt").input_ids[0][0])
pad_token = gen_kwargs.pop("pad_token")
gen_kwargs["pad_token_id"] = int(tokenizer(pad_token, return_tensors="pt").input_ids[0][0])
model_base.config.task_specific_params["text-generation"] = gen_kwargs
model_ref.config.task_specific_params["text-generation"] = gen_kwargs


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collater)


def calculate_reward(sentiment_output):
    sentiment_output_dict = {}
    for result_dict in sentiment_output:
        sentiment_output_dict[result_dict["label"]] = result_dict["score"]
    return sentiment_output_dict["CONTRADICTION"]


def evaluate(df_batch):
    print("evaluation")
    game_data = dict()
    game_data['query'] = df_batch['query']
    query_tensors = [torch.tensor(t).long().to(device) for t in df_batch['tokens']]

    response_tensors_ref, response_tensors = [], []
    bs = len(query_tensors)

    #### get response from gpt2 and gpt2_ref
    for i in range(bs):
        output = model_ref.generate(query_tensors[i].unsqueeze(dim=0).to(device),
                                    **gen_kwargs).squeeze()[len(query_tensors[i]):-1]
        response_tensors_ref.append(output)
        output = model_base.generate(query_tensors[i].unsqueeze(dim=0).to(device),
                                     **gen_kwargs).squeeze()[len(query_tensors[i]):-1]
        response_tensors.append(output)

    #### decode responses
    game_data['response (before)'] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
    game_data['response (after)'] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

    #### sentiment analysis of query/response pairs before/after
    texts = [q + r for q, r in zip(game_data['query'], game_data['response (before)'])]
    texts = [
        sentiment_pipe.tokenizer.decode(sentiment_pipe.tokenizer(text).input_ids[-max_position_embeddings + 3:],
                                        skip_special_tokens=True) for text in texts
    ]
    game_data['rewards (before)'] = [calculate_reward(output) for output in sentiment_pipe(texts, **sent_kwargs)]

    texts = [q + r for q, r in zip(game_data['query'], game_data['response (after)'])]
    texts = [
        sentiment_pipe.tokenizer.decode(sentiment_pipe.tokenizer(text).input_ids[-max_position_embeddings + 3:],
                                        skip_special_tokens=True) for text in texts
    ]
    game_data['rewards (after)'] = [calculate_reward(output) for output in sentiment_pipe(texts, **sent_kwargs)]

    # store results in a dataframe
    df_results = pd.DataFrame(game_data)

    logs = dict()
    logs.update({'evaluation/comparison_table': wandb.Table(dataframe=df_results)})
    logs.update(
        {'evaluation/mean_reward_before': torch.mean(torch.tensor(game_data['rewards (before)'])).cpu().numpy()})
    logs.update({'evaluation/mean_reward_after': torch.mean(torch.tensor(game_data['rewards (after)'])).cpu().numpy()})

    return logs


def training_loop():
    ppo_trainer = PPOTrainer(model_base, model_ref, tokenizer, **config)

    total_ppo_epochs = int(np.ceil(config["steps"] / config['batch_size']))

    dataloader_iterator = iter(dataloader)
    evaluation_batch = dataloader_iterator.next()

    for epoch in trange(total_ppo_epochs):
        batch = dataloader_iterator.next()
        logs, timing = dict(), dict()
        t0 = time.time()
        query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

        #### Get response from gpt2
        t = time.time()
        response_tensors = []
        for i in range(config['batch_size']):
            try:
                response = model_base.generate(query_tensors[i].unsqueeze(dim=0), max_length=None,
                                               # model_base.config.n_positions
                                               **gen_kwargs)
            except Exception as ex:
                print(query_tensors[i].unsqueeze(dim=0))
                print(batch['query'][i])
                print(ex)
                return
            response_cropped = response.squeeze()[len(query_tensors[i]):-1]
            # text_response = tokenizer.decode(response_cropped.squeeze())
            # index = text_response.find("\nUser:")
            # if index > 0:
            #   text_response = text_response[:index]
            # response_result = tokenizer(text_response, return_tensors="pt")["input_ids"].squeeze()

            response_result = response_cropped
            response_tensors.append(response_result)
        batch['response'] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        timing['time/get_response'] = time.time() - t

        #### Compute sentiment score
        t = time.time()
        texts = [q + r for q, r in zip(batch['query'], batch['response'])]
        texts = [
            sentiment_pipe.tokenizer.decode(sentiment_pipe.tokenizer(text).input_ids[-max_position_embeddings + 3:],
                                            skip_special_tokens=True) for text in texts
        ]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = torch.tensor([calculate_reward(output) for output in pipe_outputs]).to(device)
        timing['time/get_sentiment_preds'] = time.time() - t

        #### Run PPO step
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        timing['time/optimization'] = time.time() - t

        #### Log everything
        # timing['time/epoch'] = time.time()-t0
        table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]
        logs.update({'game_log': wandb.Table(columns=['query', 'response', 'reward'], rows=table_rows)})
        # logs.update(timing)
        # logs.update(stats)
        logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
        logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
        logs['env/reward_dist'] = rewards.cpu().numpy()
        for key in logs:
            if isinstance(logs[key], list):
                if isinstance(logs[key][0], torch.Tensor):
                    logs[key] = [array.cpu().numpy() for array in logs[key]]

        evaluation_logs = evaluate(evaluation_batch)
        logs.update(evaluation_logs)
        wandb.log(logs)


training_loop()
