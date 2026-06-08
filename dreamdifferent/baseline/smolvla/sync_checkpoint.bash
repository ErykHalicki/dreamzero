#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN_NAME="$2"
            shift 2
            ;;
        --run=*)
            RUN_NAME="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 --run <run_name>" >&2
            exit 1
            ;;
    esac
done

if [ -z "$RUN_NAME" ]; then
    echo "Missing --run <run_name>" >&2
    exit 1
fi

SMOLVLA_ROOT=/work/courses/3dv/team21/workspace/dreamzero/dreamdifferent/baseline/smolvla
SCRATCH_ROOT=/work/scratch/ehalicki

SRC=$SCRATCH_ROOT/$RUN_NAME/checkpoints
DST=$SMOLVLA_ROOT/$RUN_NAME/checkpoints

if [ ! -d "$SRC" ]; then
    echo "Source checkpoints not found: $SRC" >&2
    exit 1
fi

# Highest numbered step directory (checkpoints are named by zero-padded step).
LATEST_STEP=$(ls -1 "$SRC" | grep -E '^[0-9]+$' | sort -n | tail -1)

if [ -z "$LATEST_STEP" ]; then
    echo "No numbered checkpoint steps found in $SRC" >&2
    exit 1
fi

echo "Moving step $LATEST_STEP"
echo "  from $SRC/$LATEST_STEP"
echo "  to   $DST/$LATEST_STEP"

mkdir -p "$DST"
rm -rf "$DST/$LATEST_STEP"
mv "$SRC/$LATEST_STEP" "$DST/$LATEST_STEP"

# Repoint the 'last' symlink at the moved step.
ln -sfn "$LATEST_STEP" "$DST/last"

echo "Done. 'last' -> $LATEST_STEP"
