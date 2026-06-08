# SmolVLA baseline

## Notes
- need different venv from the top level dreamzero repo (requires newer lerobot)
`uv pip install "lerobot[smolvla]"`

## Run names
- `smolvla_finetune_so101_bottle` — homogenous
- `smolvla_finetune_so101_multi_object_new` — heterogenous
- `smolvla_finetune_so101_bottle_frozen_encoder` — homogenous, frozen image encoder
- `smolvla_finetune_so101_multi_object_new_frozen_encoder` — heterogenous, frozen image encoder

## Async inference
Start the policy server (pick a run name from above):
```
./async_server.bash --run <RUN_NAME>
```

Then start the robot client (point it at the checkpoint the server prints):
```
./async_inference.bash --checkpoint <CHECKPOINT_PATH> --task "<prompt>"
```
