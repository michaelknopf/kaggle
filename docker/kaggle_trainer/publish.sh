#!/bin/zsh
# https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html

set -e

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
REPOSITORY_NAME='kaggle-trainer'
source "$ROOT_DIR/scripts/utils.sh"

# login to ECR to push to our account's repository
account_id="$AWS_ACCOUNT_ID" \
  region="$AWS_REGION" \
  ecr_login

function describe_repo {
  echo "Finding repository $REPOSITORY_NAME ..."
  aws ecr describe-repositories \
    --repository-names "$REPOSITORY_NAME" \
    --output yaml \
    --no-cli-pager
}

function create_repo {
  echo "Creating repository $REPOSITORY_NAME ..."
  aws ecr create-repository \
    --repository-name "$REPOSITORY_NAME" \
    --image-tag-mutability MUTABLE \
    --output yaml \
    --no-cli-pager
}

# describe_repo fails if repo does not exist, so then create will run.
describe_repo || create_repo

account_ecr_domain="$(
  account_id="$AWS_ACCOUNT_ID" \
  region="$AWS_REGION" \
    ecr_domain
)"

full_image_name="$account_ecr_domain/${REPOSITORY_NAME}:latest"
docker tag "${REPOSITORY_NAME}:latest" "$full_image_name"
docker push "$full_image_name"
