import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
import torch.nn.functional as Function
from torch.autograd import Variable
import copy
import os
# utility functions
from .fc_network import JitFCNetwork, FCNetwork

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

class RewardModel():
    def __init__(self,
                 hidden_size=(512, 512, 512, 128, 32),
                 frame_num=4,
                 prev_iter_checkpoint=None,
                 state_only=False,
                 itr = 100,
                 save_logs=False,
                 input_normalization=None,
                 log_dir=None,
                 **kwargs
                 ):
        """
        """
        self.frame_num = frame_num
        self.state_only = state_only
        self.save_logs = save_logs
        obs_dim = 24
        act_dim = 20
        if state_only:
            self.model = JitFCNetwork(frame_num*obs_dim, 1, hidden_sizes=hidden_size, output_nonlinearity='tanh', device=device).to(device)
        else:
            self.model = JitFCNetwork(frame_num*(obs_dim+act_dim), 1, hidden_sizes=hidden_size, output_nonlinearity='tanh', device=device).to(device)

        if prev_iter_checkpoint is not None:
            self.model = torch.jit.load(prev_iter_checkpoint)
        
        self.itr = itr

        self.input_normalization = input_normalization
        if self.input_normalization is not None:
            if self.input_normalization > 1 or self.input_normalization <= 0:
                self.input_normalization = None
        self.writer = SummaryWriter(f"runs/{log_dir}")
        # Loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)

        self.good_samples = None
        self.bad_samples = None
    
    # def forward(self, x):
    #     return self.model(x)
    
    def loss(self, good, bad):
        return -torch.log(torch.sigmoid((good - bad))).mean()

    def train(self, good_samples, bad_samples, batch_size=4096, model_path='./model'):
        self.good_samples = torch.FloatTensor(good_samples).to(device)
        self.bad_samples = torch.FloatTensor(bad_samples).to(device)
        os.makedirs(model_path, exist_ok=True) # data saving dir

        if batch_size is not None:
            dataset = torch.utils.data.TensorDataset(self.good_samples, self.bad_samples)
            sampler = torch.utils.data.RandomSampler(dataset)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
            for i in range(self.itr):
                for _, (good_samples, bad_samples) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    loss = self.loss(self.model(good_samples), self.model(bad_samples))
                    loss.backward()
                    self.optimizer.step()

                self.scheduler.step()

                # Log information
                if self.save_logs:
                    self.writer.add_scalar(f"metric/loss", loss, i)

                if i % 100 == 0:
                    self.jit_save_model(path=model_path+f'model_{self.frame_num}_gpu')
                    print(f"Step: {i}/{self.itr}  |  Loss: {loss}")
            

        else:
            for i in range(self.itr):
                self.optimizer.zero_grad()
                loss = self.loss(self.model(self.good_samples), self.model(self.bad_samples))
                loss.backward()
                self.optimizer.step()

                self.scheduler.step()

                # Log information
                if self.save_logs:
                    self.writer.add_scalar(f"metric/loss", loss, i)

                if i % 100 == 0:
                    self.jit_save_model(path=model_path+f'model_{self.frame_num}_gpu')
                    print(f"Step: {i}/{self.itr}  |  Loss: {loss}")

    def save_model(self, path):
        try:  # for PyTorch >= 1.7 to be compatible with loading models from any lower version
            torch.save(self.model.state_dict(), path, _use_new_zipfile_serialization=False) 
        except:  # for lower versions
            torch.save(self.model.state_dict(), path)

    def load_model(self, path, eval=True):
        self.model.load_state_dict(torch.load(path))

        if eval:
            self.model.eval()           

    def jit_save_model(self, path):
        # check TorchScript model saving
        # and loading without specifing model class:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        model_scripted = torch.jit.script(self.model) # Export to TorchScript
        model_scripted.save(path+'.pt')

    def jit_load_model(self, path, eval=True):
        self.model = torch.jit.load(path+'.pt')

        if eval:
            self.model.eval()         
