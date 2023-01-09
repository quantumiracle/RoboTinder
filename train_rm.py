## train the reward model 
from pathlib import Path
import pickle
import torch
import numpy as np
from utils.model import RewardModel
from common import frame_number

# load data
with open(f'processed_data/processed_data_{frame_number}.pkl', 'rb') as f:
    load_data = pickle.load(f)
print(load_data['good'].shape, load_data['bad'].shape)


# train
reward_model = RewardModel(frame_num=frame_number, itr=10000, state_only=False, save_logs=True)
reward_model.train(good_samples=load_data['good'], bad_samples=load_data['bad'], model_path='./model/')


# test
model = torch.jit.load(f'./model/model_{frame_number}.pt')
good_v = model(torch.from_numpy(load_data['good'][:10]).float())
bad_v = model(torch.from_numpy(load_data['bad'][:10]).float())
print(good_v, good_v.mean(), bad_v, bad_v.mean())