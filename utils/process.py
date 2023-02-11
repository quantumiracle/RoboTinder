from pathlib import Path
import pickle
import numpy as np
import pandas as pd

def get_traj_from_video(video_info):
    traj = None
    video_name = video_info.split('/')[-1]
    traj_name = video_name.replace('.mp4', '.pkl')
    traj_name = traj_name.replace('video', 'traj')
    data_path = 'google_drive_data/raw/'
    try:
        traj_file_path = list(Path(data_path).rglob(traj_name))[0]
        with open(traj_file_path, 'rb') as f:
            traj = pickle.load(f)
    except:
        print("Trajectory data for {} is not found.".format(video_name))
    return traj

# Observation space:
# Index       Description
# 0 - 23	    right shadow hand dof position
# 24 - 47	    right shadow hand dof velocity
# 48 - 71	    right shadow hand dof force
# 72 - 136	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
# 137 - 166	right shadow hand fingertip force, torque (5 x 6)
# 167 - 169	right shadow hand base position
# 170 - 172	right shadow hand base rotation
# 173 - 198	right shadow hand actions
# 199 - 222	left shadow hand dof position
# 223 - 246	left shadow hand dof velocity
# 247 - 270	left shadow hand dof force
# 271 - 335	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
# 336 - 365	left shadow hand fingertip force, torque (5 x 6)
# 366 - 368	left shadow hand base position
# 369 - 371	left shadow hand base rotation
# 372 - 397	left shadow hand actions

def data_process(traj, env):
    # process observation
    if env == 'ShadowHandOver':
        right_hand_obs_idx = np.arange(24).tolist()
        left_hand_obs_idx = np.arange(187, 211).tolist()
    elif env == 'ShadowHand':
        right_hand_obs_idx = np.arange(24).tolist()  # only right hand
        left_hand_obs_idx = []
    else:
        right_hand_obs_idx = np.arange(24).tolist()
        left_hand_obs_idx = np.arange(199, 223).tolist()
    obs_idx = right_hand_obs_idx + left_hand_obs_idx
    # prcess action
    if env == 'ShadowHandOver':
        right_hand_action_idx = np.arange(20).tolist()  # only care 20 joints on hand
        left_hand_action_idx = np.arange(20, 40).tolist()
    elif env == 'ShadowHand':
        right_hand_action_idx = np.arange(20).tolist()  # only care 20 joints on hand
        left_hand_action_idx = []
    else:
        right_hand_action_idx = np.arange(6, 26).tolist()  # only care 20 joints on hand
        left_hand_action_idx = np.arange(32, 52).tolist()
    action_idx = right_hand_action_idx + left_hand_action_idx
    obs = np.array(traj['obs']).squeeze()[:, obs_idx]  # (episode_length, dim)
    action = np.array(traj['actions']).squeeze()[:, action_idx]
    # print(obs.shape, action.shape)
    return obs, action
