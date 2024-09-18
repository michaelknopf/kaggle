#!/bin/zsh

set -euxo pipefail

# see https://kislyuk.github.io/argcomplete
activate-global-python-argcomplete
eval "$(register-python-argcomplete mlops)"
