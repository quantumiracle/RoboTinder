import os
import random
import base64
import numpy as np
import json
import pandas as pd

itr = 3
VIDEO_PATH = f'google_drive_data/processed/itr{itr}'
FORMAT = ['mp4', 'gif'][0]
VIDEO_INFO = os.path.join(VIDEO_PATH, 'video_info.json')


def parse_envs(folder=VIDEO_PATH, filter=False, MAX_ITER=20000, DEFAULT_ITER=20000):
    """
    return a dict of env_name: video_paths
    """
    files = {}
    if filter:
        df = pd.read_csv('Bidexhands_Video.csv')
        # print(df)
    for env_name in os.listdir(folder):
        env_path = os.path.join(folder, env_name)
        if os.path.isdir(env_path):
            videos = os.listdir(env_path)
            video_files = []
            for video in videos:  # video name rule: EnvName_Alg_Seed_Timestamp_Checkpoint_video-episode_EpisodeID
                if video.endswith(f'.{FORMAT}'):
                    if filter:
                        if len(video.split('_')) < 6:
                            print(f'{video} is wrongly named.')
                        seed = video.split('_')[2]
                        checkpoint = video.split('_')[4]
                        try:
                            succeed_iteration = df.loc[(df['seed'] == int(seed)) & (df['env_name'] == str(env_name))]['succeed_iteration'].iloc[0]
                        except:
                            print(f'Env {env_name} with seed {seed} not found in Bidexhands_Video.csv')
                            
                        if 'unsolved' in succeed_iteration:
                            continue
                        elif pd.isnull(succeed_iteration):
                            min_iter = DEFAULT_ITER
                            max_iter = MAX_ITER
                        elif '-' in succeed_iteration:
                            [min_iter, max_iter] = succeed_iteration.split('-')
                        else:
                            min_iter = succeed_iteration
                            max_iter = MAX_ITER
                        print(min_iter, max_iter)
                        # check if the checkpoint is in the valid range
                        valid_checkpoints = np.arange(int(min_iter), int(max_iter)+1000, 1000)
                        if int(checkpoint) not in valid_checkpoints:
                            continue
                    
                    video_path = os.path.join(folder, env_name, video)
                    video_files.append(video_path)
                    print(video_path)

            files[env_name] = video_files

    with open(VIDEO_INFO, 'w') as fp:
        json.dump(files, fp)

    return files

# df = pd.read_csv('Bidexhands_Video.csv', index_col=False)
# print(df)
# succeed_iteration = df.loc[(df['seed'] == 10) & (df['env_name'] == 'ShadowHandDoorOpenOutward')]['succeed_iteration'].iloc[0]
# if '-' in succeed_iteration:
#     [min_iter, max_iter] = succeed_iteration.split('-')
#     print(min_iter, max_iter)
#     valid_iters = np.arange(int(min_iter), int(max_iter)+1000, 1000)
#     print(valid_iters)
parse_envs()