## train the reward model 
from pathlib import Path
import pickle
import torch
import numpy as np
from utils.model import RewardModel
from common import frame_number, iter

good_data = []
bad_data = []
for i in range(1, iter+1): # load data from all previous and current iterations
    with open(f'processed_data/itr{i}/processed_data_{frame_number}.pkl', 'rb') as f:
        load_data = pickle.load(f)
    print(load_data['good'].shape, load_data['bad'].shape)
    if i == 1:
        good_data = load_data['good']
        bad_data = load_data['bad']
    else:
        good_data = np.concatenate((good_data, load_data['good']), axis=0)
        bad_data = np.concatenate((bad_data, load_data['bad']), axis=0)

print('final training data shape: ', good_data.shape, bad_data.shape)


# train from pretrained
reward_model = RewardModel(frame_num=frame_number, itr=50000, prev_iter_checkpoint=f'./model/itr{iter-1}/model_{frame_number}_gpu.pt', state_only=False, save_logs=True)
# train from scratch
# reward_model = RewardModel(frame_num=frame_number, itr=50000, state_only=False, save_logs=True)
reward_model.train(good_samples=good_data, bad_samples=bad_data, model_path=f'./model/itr{iter}/')


# test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(f'./model/itr{iter}/model_{frame_number}_gpu.pt')
good_v = model(torch.from_numpy(load_data['good'][:10]).float().to(device))
bad_v = model(torch.from_numpy(load_data['bad'][:10]).float().to(device))
print(good_v, good_v.mean(), bad_v, bad_v.mean())
