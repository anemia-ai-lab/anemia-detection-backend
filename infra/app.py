#!/usr/bin/env python3
"""CDK app entrypoint for anemia-detection-backend on ECS Fargate."""

import os

import aws_cdk as cdk
from stacks.api_stack import AnemiaApiStack

app = cdk.App()

account = os.environ.get("CDK_DEFAULT_ACCOUNT") or os.environ.get("AWS_ACCOUNT_ID")
region = os.environ.get("CDK_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-west-2"

AnemiaApiStack(
    app,
    "AnemiaApiStack",
    env=cdk.Environment(account=account, region=region),
)

app.synth()
