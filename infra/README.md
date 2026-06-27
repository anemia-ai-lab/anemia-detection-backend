# CDK Python — ECS Fargate (anemia API)

Ver guía completa: [`docs/DEPLOYMENT_AWS.md`](../docs/DEPLOYMENT_AWS.md).

## Setup local

```bash
cd infra
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export AWS_REGION=us-east-1
aws sts get-caller-identity   # verificar credenciales
cdk bootstrap aws://ACCOUNT_ID/us-east-1
```

## Comandos

```bash
cdk synth          # validar plantilla
cdk diff           # ver cambios
cdk deploy         # crear/actualizar stack
cdk deploy -c imageTag=abc123   # tag de imagen ECR
```

## Tras el primer deploy

1. Editar secret `anemia-api/prod` en Secrets Manager (valores Supabase reales).
2. `docker build` + `docker push` a la URI del output `EcrRepositoryUri`.
3. `aws ecs update-service --cluster anemia-api-cluster --service anemia-api-service --force-new-deployment`

## Estructura

- `app.py` — entrypoint CDK
- `stacks/api_stack.py` — ECR, ECS Fargate 2vCPU/4GB, ALB, logs, secrets
