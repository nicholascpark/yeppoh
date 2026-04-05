#!/bin/bash
# Sync training results from cloud instance back to your Mac.
#
# Usage (run on your Mac):
#   bash scripts/cloud_sync.sh user@cloud-ip
#   bash scripts/cloud_sync.sh user@cloud-ip runs/02_locomotion
#
# This downloads checkpoints, logs, and videos to your local runs/ dir.

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/cloud_sync.sh user@cloud-ip [run_dir]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/cloud_sync.sh root@203.0.113.42"
    echo "  bash scripts/cloud_sync.sh root@203.0.113.42 runs/02_locomotion"
    exit 1
fi

REMOTE="$1"
RUN_DIR="${2:-runs/}"

echo "Syncing ${RUN_DIR} from ${REMOTE}..."

rsync -avz --progress \
    --include="*/" \
    --include="*.pt" \
    --include="*.mp4" \
    --include="*.glb" \
    --include="*.gif" \
    --include="*.yaml" \
    --include="events.out.*" \
    --exclude="*" \
    "${REMOTE}:~/yeppoh/${RUN_DIR}" \
    "./${RUN_DIR}"

echo ""
echo "Synced to ./${RUN_DIR}"
echo "View checkpoints: ls ${RUN_DIR}/*/*.pt"
echo "View in TensorBoard: tensorboard --logdir ${RUN_DIR}"
