# CDK Python — ECS Fargate (anemia API)

Ver guía completa: [`docs/DEPLOYMENT_AWS.md`](../docs/DEPLOYMENT_AWS.md).

## Prerrequisitos

- Python 3.11+ (venv en `infra/.venv`)
- **Node.js 22+** para CDK CLI (`npm install -g aws-cdk@2`)
- AWS CLI configurado (`aws configure`), región **us-west-2** (misma que Supabase)

## Setup local

```bash
cd infra
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
npm install -g aws-cdk@2

export AWS_REGION=us-west-2
export JSII_SILENCE_WARNING_DEPRECATED_NODE_VERSION=1
aws sts get-caller-identity   # verificar credenciales
cdk bootstrap aws://ACCOUNT_ID/us-west-2
```

Desde la raíz del repo también puedes validar la plantilla con `make cdk-synth`.

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
3. `aws ecs update-service --cluster anemia-api-cluster --service anemia-api-service --force-new-deployment --region us-west-2`

## Estructura

- `app.py` — entrypoint CDK
- `stacks/api_stack.py` — ECR, ECS Fargate 2vCPU/4GB, ALB, circuit breaker, logs, secrets
