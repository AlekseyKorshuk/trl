docker build --platform linux/amd64 -t alekseykorshuk/ppo-trainer:v1  .
docker push alekseykorshuk/ppo-trainer:v1
kubectl delete pod ppo-trainer
kubectl apply -f deploy.yaml