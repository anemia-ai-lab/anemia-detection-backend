# CDK Python — ECS Fargate (anemia API)

Ver guía completa: [`docs/DEPLOYMENT_AWS.md`](../docs/DEPLOYMENT_AWS.md).

## Prerrequisitos

- **Node.js 22** — fijado en [`.nvmrc`](../.nvmrc) (`nvm use` en la raíz del repo)
- Python 3.11+ (venv en `infra/.venv`)
- CDK CLI: `npm install -g aws-cdk@2`
- AWS CLI configurado (`aws configure`), región **us-west-2** (misma que Supabase)

## Setup local

```bash
# Desde la raíz del repo
nvm use
npm install -g aws-cdk@2

cd infra
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export AWS_REGION=us-west-2
aws sts get-caller-identity   # verificar credenciales
cdk bootstrap aws://ACCOUNT_ID/us-west-2
```

Validar plantilla desde la raíz: `make cdk-synth`.

## Comandos

```bash
cdk synth          # validar plantilla
cdk diff           # ver cambios
cdk deploy         # crear/actualizar stack
cdk deploy -c imageTag=abc123   # tag de imagen ECR
```

## Tras el primer deploy

1. Editar secret `anemia-api/prod` en Secrets Manager (valores Supabase reales).
2. Desde la raíz: `make docker-push-ecr` (build **linux/amd64** + push a ECR), o ver [`docs/DEPLOYMENT_AWS.md`](../docs/DEPLOYMENT_AWS.md) § build.
3. `aws ecs update-service --cluster anemia-api-cluster --service anemia-api-service --force-new-deployment --region us-west-2`

## Estructura

- `app.py` — entrypoint CDK
- `stacks/api_stack.py` — ECR, ECS Fargate 2vCPU/4GB, ALB, circuit breaker, logs, secrets
