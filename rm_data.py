## preprocess data for the reward model 

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from utils.process import data_process, get_traj_from_video
from common import frame_number

def get_data(verbose=True):
    df = pd.read_csv('crowdsourced-robotinder-demo/flag_data1.csv')
    df.columns = ['env_name', 'user_choice', 'left_video', 'right_video', 'time'] # add column names
    print(df.head(5))
    print(f'total number of samples: {len(df.index)}')

    # load data in dict
    trajs = {'left': [], 
            'right': [],
            'user_choice': []}
    for index, row in df.iterrows():
        print(f"sample: ({index}, {row['env_name']}, {row.user_choice})") if verbose else None
        trajs['user_choice'].append(row.user_choice)

        l_traj = get_traj_from_video(row.left_video)
        r_traj = get_traj_from_video(row.left_video)
        if l_traj is not None and r_traj is not None:
            l_obs, l_action = data_process(l_traj, row['env_name'])
            r_obs, r_action = data_process(r_traj, row['env_name'])
            trajs['left'].append({'obs': l_obs, 'action': l_action})
            trajs['right'].append({'obs': r_obs, 'action': r_action})
            if verbose:
                print(f"left:, ori obs shape:{np.array(l_traj['obs']).shape}, proc obs shape: {l_obs.shape}, action shape: {l_action.shape}")
                print(f"right:, ori obs shape:{np.array(r_traj['obs']).shape}, proc obs shape: {r_obs.shape}, action shape: {r_action.shape}")

    def preprocess_data_for_reward_model(data, stack_size=4, obs_dim=24, action_dim=20):
        merge_data = {'good': np.array([]), 'bad': np.array([])}
        user_choie = data['user_choice']  # [nan 'Right' 'Both Bad' 'Left' 'Not Sure' 'Both Good']

        for l, r, c in zip(data['left'], data['right'], data['user_choice']):
            two_hands = False
            if l['obs'].shape[-1]==obs_dim and l['action'].shape[-1]==action_dim:  # right hand only
                two_hands = False
            elif l['obs'].shape[-1]==2*obs_dim and l['action'].shape[-1]==2*action_dim:
                two_hands = True
            else:
                two_hands = False
                raise ValueError('obs or action shape is not correct')
                continue

            slice_l_obs = [l['obs'][i:i+stack_size] for i in range(len(l['obs'])-stack_size+1)]  # [1,2,3] to [[1,2], [2,3]] with stack_size=2
            slice_l_action = [l['action'][i:i+stack_size] for i in range(len(l['action'])-stack_size+1)]
            slice_r_obs = [r['obs'][i:i+stack_size] for i in range(len(r['obs'])-stack_size+1)]
            slice_r_action = [r['action'][i:i+stack_size] for i in range(len(r['action'])-stack_size+1)]

            if two_hands: # split right and left hand and stack them
                def merge_two_hands(data, dim):
                    data = np.array(data).reshape((-1, stack_size, 2, dim))  
                    data = np.swapaxes(data, 1, 2)  # [batch, 2, stack_size, dim]
                    data = data.reshape((-1, stack_size, dim))
                    return data
                slice_l_obs = merge_two_hands(slice_l_obs, obs_dim)  # [batch, stack_size, 2*obs_dim] -> [2*batch, stack_size, obs_dim]
                slice_l_action = merge_two_hands(slice_l_action, action_dim)
                slice_r_obs = merge_two_hands(slice_r_obs, obs_dim)
                slice_r_action = merge_two_hands(slice_r_action, action_dim)
            
            if c == 'Left':
                slice_l_obs_action = [np.concatenate((slice_l_obs[i], slice_l_action[i]), axis=1).reshape(-1) for i in range(len(slice_l_obs))]  # [[oo], [oo]] and [[aa], [aa]] to [oaoa, oaoa]
                merge_data['good'] = np.concatenate((merge_data['good'], np.array(slice_l_obs_action)), axis=0) if merge_data['good'].size else np.array(slice_l_obs_action)
                select_index_from_bad = np.random.choice(len(slice_r_obs), size=len(slice_l_obs), replace=True)
                select_slice_r_obs = [slice_r_obs[i] for i in select_index_from_bad]
                select_slice_r_action = [slice_r_action[i] for i in select_index_from_bad]
                slice_r_obs_action = [np.concatenate((select_slice_r_obs[i], select_slice_r_action[i]), axis=1).reshape(-1) for i in range(len(select_slice_r_obs))]
                merge_data['bad'] = np.concatenate((merge_data['bad'], np.array(slice_r_obs_action)), axis=0) if merge_data['bad'].size else np.array(slice_r_obs_action)

            elif c == 'Right':
                slice_r_obs_action = [np.concatenate((slice_r_obs[i], slice_r_action[i]), axis=1).reshape(-1) for i in range(len(slice_r_obs))]
                merge_data['good'] = np.concatenate((merge_data['good'], np.array(slice_r_obs_action)), axis=0) if merge_data['good'].size else np.array(slice_r_obs_action)
                select_index_from_bad = np.random.choice(len(slice_l_obs), size=len(slice_r_obs), replace=True)
                select_slice_l_obs = [slice_l_obs[i] for i in select_index_from_bad]
                select_slice_l_action = [slice_l_action[i] for i in select_index_from_bad]
                slice_l_obs_action = [np.concatenate((select_slice_l_obs[i], select_slice_l_action[i]), axis=1).reshape(-1) for i in range(len(select_slice_l_obs))]
                merge_data['bad'] = np.concatenate((merge_data['bad'], np.array(slice_l_obs_action)), axis=0) if merge_data['bad'].size else np.array(slice_l_obs_action)

        return merge_data

    merg_trajs = preprocess_data_for_reward_model(trajs, stack_size=frame_number)

    return merg_trajs


data = get_data() # return: {'good': data, 'bad': data}, data shape: [batch, stack_size*(obs_dim+action_dim)]
print(data['good'].shape, data['bad'].shape)

with open(f'processed_data/processed_data_{frame_number}.pkl', 'wb') as f:
    pickle.dump(data, f)

with open(f'processed_data/processed_data_{frame_number}.pkl', 'rb') as f:
    load_data = pickle.load(f)
print(load_data['good'].shape, load_data['bad'].shape)
