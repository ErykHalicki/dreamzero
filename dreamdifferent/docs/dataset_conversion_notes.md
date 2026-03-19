# Egoverse dataset convesion notes

## Egoverse data
Since the egoverse franka arm data is on the Euler cluster, I copied a chunk over using:
```
rsync  -av --progress ehalicki@euler.ethz.ch:/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250828_105242.h5 ~/Documents/School/ETH/3d_vision/dreamzero/dreamdifferent/data/bag_grocery_data.h5
```

Egoverse dataset has no annotations? Maybe it is still possible to fine tune using just the high level instruction "bag the groceries" and "put the object in the bowl" without any manual episode annotation?

## Conversion to lerobot
### What version of LeRobot to use???
`dreamzero/scripts/data/convert_droid.py` and `dreamzero/scripts/data/convert_agibot.py` are some examples of dataset conversions to lerobot v2.0
LeRobotDataset v2.0 is (very) outdated so cannot use the newest versions of LeRobot. Also the LeRobot version used by the dreamzero repo is not in pyproject.toml?? TT

LeRobotDataset was updated to v2.0 at `huggingface/lerobot/commit/32eb0cec8f322a7d93a1ec2008dd1a11ae6286b3` which corresponds to the LeRobot v0.3.2 release
The `lerobot.common.datasets.lerobot_dataset` import at the top of `dreamzero/scripts/data/convert_agibot.py` confirms this, since the lerobot.common.datasets directory got renamed to lerobot.datasets after commit `e23b41e79a71396fb1c44dd09fd8864f18a438ec`

### How to convert from Egoverse .h5 to lerobot v2.0?
Would it make sense to update the dreamzero repo to just use LeRobotDataset v3.0? 



## Conversion to gear format
Follow `dreamzero/docs/DATASET_TO_GEAR_AND_TRAIN.md`

