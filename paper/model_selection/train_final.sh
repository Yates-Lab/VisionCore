#!/usr/bin/env bash
# Train the single pinned final twin, at the configuration the sweep settled.
#
# This script deliberately contains no training flags. It hands off to
# launch.py, which builds the final model's command with the same
# `build_command` that produced every arm the model was selected against.
#
# That indirection is the entire point. `experiments/train_digital_twin_120_long.sh`
# carried its own copy of the flags, drifted from the checkpoint it supposedly
# produced, and left the paper model's real settings recorded nowhere -- which
# is why paper/model_selection/ exists. A second shell script with a second copy
# of the flag list would rebuild that defect.
#
# It will refuse to run until FINAL_SETTINGS in launch.py is filled in from
# collect.py / stability.py. That refusal is a feature: training the defaults
# and calling the result "final" is the failure mode being designed out.
#
#   ./paper/model_selection/train_final.sh 0        # GPU index, default 0
#
# Run from the VisionCore root. One GPU at a time -- the box is shared.

set -euo pipefail

GPU="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

cd "${ROOT}"

echo "== Final model, GPU ${GPU}"
echo "-- checking the sweep has settled the configuration"
uv run python paper/model_selection/launch.py FINAL --dry-run

cat <<'EOF'

-- pre-flight
   1. Is the GPU free, and has it been free for long enough that you are not
      about to contend with someone else's multi-hour job?  nvidia-smi
   2. Does collect.py show every arm the settings were read from as `done`
      and evaluated?
   3. Is the protocol hash in the dry-run above the one the arms were run
      under? A different hash means the settings were chosen under a protocol
      this run will not share.

EOF

read -r -p "Proceed with the final training run? [y/N] " reply
case "${reply}" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "aborted"; exit 1 ;;
esac

uv run python paper/model_selection/launch.py FINAL --gpu "${GPU}"

echo
echo "-- evaluate it before anything is pinned to it"
echo "   uv run python paper/model_selection/evaluate.py FINAL --gpu ${GPU}"
