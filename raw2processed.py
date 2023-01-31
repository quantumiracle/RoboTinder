from moviepy.editor import VideoFileClip
import os
import json
from os import listdir
from os.path import isfile, join, isdir
import shutil

itr = 4
filter = False  # only save videos after passing the filter: video_info.json
VIDEO_INFO = f'google_drive_data/processed/itr{itr}/video_info.json'
if filter:
    with open(VIDEO_INFO, 'r') as fp:
        filter_info = json.load(fp)

saved_path = f'google_drive_data/processed/itr{itr}' if not filter else 'filter_processed'
os.makedirs(saved_path, exist_ok=True)

# all 20 envs
# ENVS = ['ShadowHand', 'ShadowHandCatchAbreast', 'ShadowHandOver', 'ShadowHandBlockStack', 'ShadowHandCatchUnderarm',
# 'ShadowHandCatchOver2Underarm', 'ShadowHandBottleCap', 'ShadowHandLiftUnderarm', 'ShadowHandTwoCatchUnderarm',
# 'ShadowHandDoorOpenInward', 'ShadowHandDoorOpenOutward', 'ShadowHandDoorCloseInward', 'ShadowHandDoorCloseOutward',
# 'ShadowHandPushBlock', 'ShadowHandKettle', 
# 'ShadowHandScissors', 'ShadowHandPen', 'ShadowHandSwingCup', 'ShadowHandGraspAndPlace', 'ShadowHandSwitch']

# selected 17 envs
ENVS = ['ShadowHand', 'ShadowHandCatchAbreast', 'ShadowHandOver', 'ShadowHandBlockStack', 'ShadowHandCatchUnderarm',
        'ShadowHandCatchOver2Underarm', 'ShadowHandBottleCap', 'ShadowHandLiftUnderarm', 'ShadowHandTwoCatchUnderarm', 'ShadowHandDoorOpenInward',
        'ShadowHandDoorOpenOutward', 'ShadowHandDoorCloseInward', 'ShadowHandGraspAndPlace', 'ShadowHandPushBlock',
        'ShadowHandScissors', 'ShadowHandPen', 'ShadowHandSwitch']

# unseen 4 envs
# ENVS = ['ShadowHandCatchAbreastPen', 'ShadowHandCatchUnderarmPen', 'ShadowHandTwoCatchAbreast', 'ShadowHandGraspAndPlaceEgg']


for env in ENVS:
    os.makedirs(join(saved_path, env), exist_ok=True)

mypath = f'google_drive_data/raw/itr{itr}'
seedfolders = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
for seedfolder in seedfolders:
    envfolders = [f for f in listdir(seedfolder) if isdir(join(seedfolder, f))]
    print(envfolders)
    for folder in envfolders:
        print(folder)
        fs = [join(seedfolder, folder, f) for f in listdir(join(seedfolder, folder)) if isfile(join(seedfolder, folder, f))]
        # print(fs)
        if filter:
            env_files = filter_info[folder]
            env_files = [f.split('/')[-1] for f in env_files]
            # print(env_files)
        for f in fs:
            if f.endswith(".mp4"):
                if filter:
                    if f.split('/')[-1] in env_files:
                        shutil.copyfile(f, join(saved_path, folder, f.split('/')[-1]))
                else:
                    shutil.copyfile(f, join(saved_path, folder, f.split('/')[-1]))
