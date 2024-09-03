
export ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
export AWS_SAGEMAKER_ACCOUNT_ID=763104351884

function __init_private_rc() {
  # Create a custom initialization script at scripts/private_rc.sh
  # This path is gitignored, allowing personalization for your private machine.

  # Skip if previously sourced
  if [[ "$_PRIVATE_RC_SOURCED" == 1 ]]; then
    return
  fi

  # Define the file you want to check
  private_rc="$ROOT_DIR/scripts/private_rc.sh"

  # Check if the file exists and is a regular file
  if [ -f "$private_rc" ]; then
      source "$private_rc"
      export _PRIVATE_RC_SOURCED=1
  fi
}
__init_private_rc

function aws_whoami {
  # print current authenticated AWS user/account
  aws sts get-caller-identity | cat
}

function ecr_login {
  # env variables required: account_id, region
  # see https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html#cli-authenticate-registry
  aws ecr get-login-password --region "$region" \
    | docker login \
      --username AWS \
      --password-stdin \
      "$(ecr_domain)"
}

function ecr_domain {
  # env variables required: account_id, region
  echo "$account_id.dkr.ecr.$region.amazonaws.com"
}
