# Egoverse dataset convesion notes

## Egoverse data
- All data is stored in h5 format, with episode names based on datetime (eg. 20250826_111157.h5 and 20250827_110220.h5)
- 2 tasks are available: "bag_groceries" and "object_in_bowl_processed_50hz" with different embodiments and h5 file structures
    - "bag_groceries" uses a 2 arm setup
    - "object_in_bowl" uses a 1 arm setup
- There are no task annotations inside the h5 files, and there is no metadata provided describing epsiode boundaries (might be needed since it seems there are multiple episodes per file in the "object in bowl" task
    - Maybe it is still possible to fine-tune using just the high level instruction "bag the groceries" and "put the object in the bowl" without any manual episode annotation?
- Looks like the data is recorded at 50Hz based on the folder name inside Euler (`object_in_bowl_processed_50hz`)
- robot state observation and action dim is 24 per arm (17 dof hand, 7 dof arm)
    - observations.qpos_hand (17), observations.qpos_arm (7) and actions_arm (7), actions_hand(17) 
    - values seem normalized between -1 to 1 (but could also be radians)
- camera observations are stored uncompressed with shape `[episode_len, height, width, channels]`
    - "bag_groceries" has 2 camera views, observations.images.aria_rgb_cam and observations.images.oakd_front_view
    - "object_in_bowl_processed_50hz" has only 1 view, observations.images.aria_rgb_cam

## Copying data from Euler
Since the egoverse franka arm data is on the Euler cluster, I copied an episode using:
```
rsync  -av --progress ehalicki@euler.ethz.ch:/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250828_105242.h5 dreamzero/dreamdifferent/data/bag_grocery_data.h5
```
## Conversion to lerobot
### What version of LeRobot to use???
LeRobotDataset v2.0 is (very) outdated so cannot use the newest versions of LeRobot. Also the LeRobot version used by the dreamzero repo is not in pyproject.toml?? TT

LeRobotDataset was updated to v2.0 at `huggingface/lerobot/commit/32eb0cec8f322a7d93a1ec2008dd1a11ae6286b3` which corresponds to BEFORE the LeRobot v0.3.2 release
The `lerobot.common.datasets.lerobot_dataset` import at the top of `dreamzero/scripts/data/convert_agibot.py` confirms this, since the lerobot.common.datasets directory got renamed to lerobot.datasets before v0.3.2 

One small problem, there is no LeRobot version < 0.3.2 published to pip anymore, so we need to manually install the lerobot repo at that specific commit hash :D

Due to some dependency weirdness, directly installing from github doesnt work, so I've added the scipt `dreamdifferent/scripts/install_compatible_lerobot_linux.sh` (and macos version for local testing).

### How to convert from Egoverse .h5 to lerobot v2.0?
Would it make sense to update the dreamzero repo to just use LeRobotDataset v3.0? 
Probably not, the repo does not seem well organized, so the refactor would probably take a 5+ hours

`dreamzero/scripts/data/convert_droid.py` and `dreamzero/scripts/data/convert_agibot.py` are some examples of dataset conversions to lerobot v2.0

Referencing [LeRobotDataset Github](https://github.com/huggingface/lerobot/blob/32eb0cec8f322a7d93a1ec2008dd1a11ae6286b3/lerobot/common/datasets/lerobot_dataset.py) and `dreamzero/scripts/data/convert_agibot.py` (which also does h5 -> lerobot)

#### scripts/egoverse_to_lerobotv2.py
We use `LeRobotDataset.create()` to make an empty dataset object which we can directly write into.
Then using `dataset.add_frame`, `dataset.save_episode`, and `dataset.consolidate`, we append observations and actions from the h5 into the LeRobotDataset object, which are automatically saved to disk.

Since the images are compressed into .mp4 format, the resulting lerobot dataset is MUCH smaller than the original h5 file (1.8TB -> 7GB)


Also, the default lerobot method of writing pngs and then combining them into a single mp4 afterwards was significantly slowing down dataset conversion due to disk write bottlenecks
To speed things up, I avoided the png writing process entirely and just piped image data directly into an ffmpeg subprocess, keeping everything in memory. This changed the dataset conversion time from ~100 hours for 3 episodes to ~5 hours without any major parallelization

### Batching the lerobot conversion script on Euler
Using [Euler HPC Docs](https://docs.hpc.ethz.ch/batchsystem/slurm/) as reference

Using slurm was simpler than I expected. `srun` runs commands direclty, and `sbatch` runs batch scripts.
run `sbatch dreamzero/dreamdifferent/scripts/convert_full_dataset.bash` to convert the entire egoverse dataset on euler

Then I rsynced the dataset to the D-INFK student cluster, specifically `/work/courses/3dv/team21/datasets`
`rsync -av --progress /cluster/scratch/ehalicki/egoverse/ ehalicki@student-cluster.inf.ethz.ch://work/courses/3dv/team21/datasets/`

As of March 20, 2026, I've only converted the grocery bagging dataset, since it uses our target bimanual embodiment.

## Conversion to gear format
Follow `dreamzero/docs/DATASET_TO_GEAR_AND_TRAIN.md`

