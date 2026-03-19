# Egoverse dataset convesion notes

## Egoverse data
- All data is stored in h5 format, with episode names based on datetime (eg. 20250826_111157.h5 and 20250827_110220.h5)
- 2 tasks are available: "bag_groceries" and "object_in_bowl_processed_50hz" with different embodiments and h5 file structures
    - "bag_groceries" uses a 2 arm setup
    - "object_in_bowl" uses a 1 arm setup
- There are no task annotations inside the h5 files, and there is no metadata provided describing epsiode boundaries (might be needed since it seems there are multiple episodes per file in the "object in bowl" task
    - Maybe it is still possible to fine-tune using just the high level instruction "bag the groceries" and "put the object in the bowl" without any manual episode annotation?
- Looks like the data is recorded at 50Hz based on the folder name inside Euler (`object_in_bowl_processed_50hz`)
- observation and action dim is 24 per arm (17 dof hand, 7 dof arm)
    - observations.qpos_hand (17), observations.qpos_arm (7) and actions_arm (7), actions_hand(17) 
    - values seem normalized between -1 to 1 (but could also be radians)

## Copying data from Euler
Since the egoverse franka arm data is on the Euler cluster, I copied an episode using:
```
rsync  -av --progress ehalicki@euler.ethz.ch:/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250828_105242.h5 dreamzero/dreamdifferent/data/bag_grocery_data.h5
```
## Conversion to lerobot
### What version of LeRobot to use???
LeRobotDataset v2.0 is (very) outdated so cannot use the newest versions of LeRobot. Also the LeRobot version used by the dreamzero repo is not in pyproject.toml?? TT

LeRobotDataset was updated to v2.0 at `huggingface/lerobot/commit/32eb0cec8f322a7d93a1ec2008dd1a11ae6286b3` which corresponds to the LeRobot v0.3.2 release
The `lerobot.common.datasets.lerobot_dataset` import at the top of `dreamzero/scripts/data/convert_agibot.py` confirms this, since the lerobot.common.datasets directory got renamed to lerobot.datasets after commit `e23b41e79a71396fb1c44dd09fd8864f18a438ec`

### How to convert from Egoverse .h5 to lerobot v2.0?
Would it make sense to update the dreamzero repo to just use LeRobotDataset v3.0? 
Probably not, the repo does not seem well organized, so the refactor would probably take a 5+ hours

Referencing: [LeRobotDataset @ LeRobot v0.3.2](https://github.com/huggingface/lerobot/blob/32eb0cec8f322a7d93a1ec2008dd1a11ae6286b3/lerobot/common/datasets/lerobot_dataset.py)


`dreamzero/scripts/data/convert_droid.py` and `dreamzero/scripts/data/convert_agibot.py` are some examples of dataset conversions to lerobot v2.0

`dreamzero/scripts/data/convert_agibot.py` is especially promising, since it converts from h5 to lerobot v2.0
 

## Conversion to gear format
Follow `dreamzero/docs/DATASET_TO_GEAR_AND_TRAIN.md`

