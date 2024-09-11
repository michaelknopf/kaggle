#!/bin/zsh
# https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html

set -e

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
source "$ROOT_DIR/scripts/utils.sh"

# login to ECR to pull base image
account_id="$AWS_SAGEMAKER_ACCOUNT_ID" \
  region="$AWS_REGION" \
  ecr_login

# build python package
poetry@1.8 build --format wheel --output "$ROOT_DIR/dist"

# build image
docker build \
  -t 'ml-soln-trainer:latest' \
  --build-context dist="$ROOT_DIR/dist" \
  "$ROOT_DIR/docker/ml_soln_trainer"
