#!/usr/bin/env bash
# Build the Python environment of the Gated DeltaNet baseline (docs/gated_deltanet_baseline.md)
# in a new directory: the project's locked environment (uv.lock: the torch, triton and every
# other package of every other run) plus flash-linear-attention's kernels at the pinned release
# (scripts/gated_deltanet_requirements.txt: two wheels, hash-checked, installed without their
# dependencies, so nothing else is installed or upgraded). No existing environment is changed:
# the directory must not exist yet.
#
#   scripts/gated_deltanet_env.sh /path/to/gdn-env
#   PYTHONPATH=<checkout>/python /path/to/gdn-env/bin/python -m hsn.experiments worker ...
#
# The Rust module (hsn_rs) is not built: the baseline does not use it
# (`uv run --no-sync maturin develop --uv -m rust/hsn-py/Cargo.toml` with UV_PROJECT_ENVIRONMENT
# set to the new directory builds it, for the whole test suite to run there).
set -euo pipefail
ENV=${1:?usage: scripts/gated_deltanet_env.sh <directory of the new environment>}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -e "$ENV" ]; then
  echo "refusing: $ENV exists (the environment is built in a new directory)" >&2
  exit 2
fi
UV_PROJECT_ENVIRONMENT="$ENV" uv sync --project "$ROOT" --locked --extra dev --extra cuda
uv pip install --python "$ENV/bin/python" --no-deps --require-hashes \
  -r "$ROOT/scripts/gated_deltanet_requirements.txt"
"$ENV/bin/python" -c 'import fla, torch, triton
print(f"gated deltanet environment: fla {fla.__version__}, torch {torch.__version__}, triton {triton.__version__}")'
