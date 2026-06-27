"""ECS Fargate + ALB + ECR for the anemia detection API."""

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    SecretValue,
    Stack,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_ecr as ecr,
)
from aws_cdk import (
    aws_ecs as ecs,
)
from aws_cdk import (
    aws_ecs_patterns as ecs_patterns,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct

_INFERENCE_MODEL_PATHS = (
    "ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras,"
    "ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed123.keras,"
    "ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed456.keras"
)


class AnemiaApiStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        image_tag = self.node.try_get_context("imageTag") or "latest"
        desired_count = int(self.node.try_get_context("desiredCount") or 1)

        repository = ecr.Repository(
            self,
            "ApiRepository",
            repository_name="anemia-detection-backend",
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=10)],
        )

        api_secret = secretsmanager.Secret(
            self,
            "ApiSecret",
            secret_name="anemia-api/prod",
            description="Supabase keys and METRICS_BEARER_TOKEN for anemia API (edit JSON after deploy)",
            secret_object_value={
                "SUPABASE_URL": SecretValue.unsafe_plain_text("REPLACE_ME"),
                "SUPABASE_KEY": SecretValue.unsafe_plain_text("REPLACE_ME"),
                "SUPABASE_SERVICE_ROLE_KEY": SecretValue.unsafe_plain_text("REPLACE_ME"),
                "METRICS_BEARER_TOKEN": SecretValue.unsafe_plain_text("REPLACE_ME"),
            },
        )

        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
            ],
        )

        cluster = ecs.Cluster(
            self,
            "Cluster",
            cluster_name="anemia-api-cluster",
            vpc=vpc,
        )

        log_group = logs.LogGroup(
            self,
            "ApiLogGroup",
            log_group_name="/ecs/anemia-api",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        fargate = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "ApiService",
            service_name="anemia-api-service",
            cluster=cluster,
            cpu=2048,
            memory_limit_mib=4096,
            desired_count=1 if desired_count == 0 else desired_count,
            public_load_balancer=True,
            assign_public_ip=True,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            min_healthy_percent=100,
            max_healthy_percent=200,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_ecr_repository(repository, tag=image_tag),
                container_port=8000,
                environment={
                    "APP_ENV": "production",
                    "DEBUG": "false",
                    "TRUST_PROXY_HEADERS": "true",
                    "INFERENCE_MODEL_PATHS": _INFERENCE_MODEL_PATHS,
                },
                secrets={
                    "SUPABASE_URL": ecs.Secret.from_secrets_manager(api_secret, "SUPABASE_URL"),
                    "SUPABASE_KEY": ecs.Secret.from_secrets_manager(api_secret, "SUPABASE_KEY"),
                    "SUPABASE_SERVICE_ROLE_KEY": ecs.Secret.from_secrets_manager(
                        api_secret,
                        "SUPABASE_SERVICE_ROLE_KEY",
                    ),
                    "METRICS_BEARER_TOKEN": ecs.Secret.from_secrets_manager(
                        api_secret,
                        "METRICS_BEARER_TOKEN",
                    ),
                },
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="api",
                    log_group=log_group,
                ),
            ),
            health_check_grace_period=Duration.seconds(120),
        )

        if desired_count == 0:
            cfn_service = fargate.service.node.default_child
            cfn_service.add_property_override("DesiredCount", 0)

        fargate.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        CfnOutput(self, "LoadBalancerDNS", value=fargate.load_balancer.load_balancer_dns_name)
        CfnOutput(self, "EcrRepositoryUri", value=repository.repository_uri)
        CfnOutput(self, "SecretArn", value=api_secret.secret_arn)
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "ServiceName", value=fargate.service.service_name)
