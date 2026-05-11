# python scripts/openpi_policy_client.py \
#   --host 127.0.0.1 \
#   --port 23261 \
#   --prompt "pick up the shuttlecock and place it into the container" \
#   --mode async \
#   --fps 15 \
#   --camera-fps 30 \
#   --duration-sec 30 \
#   --execute-actions-per-chunk 6 \
#   --chunk-size-threshold 0.9 \
#   --max-relative-target 5

python scripts/openpi_policy_client.py \
  --host 127.0.0.1 \
  --port 23261 \
  --prompt "pick up the shuttlecock and place it into the container" \
  --mode sync \
  --fps 12 \
  --camera-fps 30 \
  --duration-sec 120 \
  --execute-actions-per-chunk 3 \
  --max-relative-target 3 \
  --no-resize-images \
  --debug-dump-observation
