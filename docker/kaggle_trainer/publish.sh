# https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html

set -e

ROOT_DIR="$(git rev-parse --show-toplevel)"
source "$ROOT_DIR/scripts/utils.sh"

# login to ECR to push to our account's repository
account_id="$AWS_ACCOUNT_ID" \
  region="$AWS_REGION" \
  ecr_login

full_name="$(ecr_domain)/kaggle-trainer:latest"
docker tag "kaggle-trainer:latest" "$full_name"
docker push "$full_name"
