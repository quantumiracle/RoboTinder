import numpy as np
import torch
import torch.nn as nn

class GradientReverse(torch.autograd.Function):
    scale = 1.0
    scale = torch.tensor(scale, requires_grad=False)
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    
def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)

class JitFCNetwork(torch.jit.ScriptModule):
    # model can be saved as torchscript (no need to specify class when calling)
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 output_nonlinearity=None,
                 device='cpu',
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(JitFCNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.device = device

        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])
        # self.layer_idx = list(torch.arange(0, len(self.layer_sizes)-1))
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh
        self.output_nonlinearity = output_nonlinearity
        self.output_act = None
        if output_nonlinearity is not None:
            if output_nonlinearity == 'relu':
                self.output_act = torch.relu
            elif output_nonlinearity == 'sigmoid':
                self.output_act = torch.sigmoid
            elif output_nonlinearity == 'tanh':
                self.output_act = torch.tanh                
            elif output_nonlinearity == 'softmax':
                self.output_act = nn.Softmax(dim=-1)                
            else:
                raise NotImplementedError

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)).to(self.device) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)).to(self.device) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)).to(self.device) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)).to(self.device) if out_scale is not None else torch.ones(self.act_dim)
    
    @torch.jit.script_method
    def forward(self, x):
        # TODO(Aravind): Remove clamping to CPU
        # This is a temp change that should be fixed shortly
        # if x.is_cuda:
        #     out = x.to('cpu')
        # else:
        out = x
        out = (out - self.in_shift)/(self.in_scale + 1e-8)
        # This does not work with TorchScript due to indexing layers.
        # Ref: https://github.com/pytorch/pytorch/issues/47336
        # for i in range(len(self.fc_layers)-1):  
        # for i in self.layer_idx[:-1]:
        #     out = self.fc_layers[i](out)
        i = 0
        for l in self.fc_layers:
            if i < len(self.fc_layers)-1:
                out = l(out)
                out = self.nonlinearity(out)
                i+=1
        out = self.fc_layers[-1](out)
        if self.output_nonlinearity is not None:
            out = self.output_act(out)
        out = out * self.out_scale + self.out_shift
        return out

class FCNetwork(nn.Module):
    # can be serialized for multiprocessing
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 output_nonlinearity=None,
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(FCNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])
        self.layer_idx = list(torch.arange(0, len(self.layer_sizes)-1))
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh
        self.output_nonlinearity = output_nonlinearity
        self.output_act = None
        if output_nonlinearity is not None:
            if output_nonlinearity == 'relu':
                self.output_act = torch.relu
            elif output_nonlinearity == 'sigmoid':
                self.output_act = torch.sigmoid
            elif output_nonlinearity == 'tanh':
                self.output_act = torch.tanh                
            elif output_nonlinearity == 'softmax':
                self.output_act = nn.functional.softmax                
            else:
                raise NotImplementedError

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)
    
    def forward(self, x):
        # TODO(Aravind): Remove clamping to CPU
        # This is a temp change that should be fixed shortly
        # if x.is_cuda:
        #     out = x.to('cpu')
        # else:
        out = x
        out = (out - self.in_shift)/(self.in_scale + 1e-8)
        for i in range(len(self.fc_layers)-1):  # this does not work with TorchScript due to indexing layers
        # for i in self.layer_idx[:-1]:
            out = self.fc_layers[i](out)
        out = self.fc_layers[-1](out)
        if self.output_nonlinearity is not None:
            out = self.output_act(out)
        out = out * self.out_scale + self.out_shift
        return out
