
import torch
import torch.nn as nn
from torchdiffeq import odeint

#Model NODE

class ODEfunc(nn.Module):
    def __init__(self, input_size):
        super(ODEfunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),        #Kích thước đầu vào (số tính năng)
            nn.Tanh(),
            nn.Linear(64, input_size)
        )

    def forward(self, t, y):
        return self.net(y)

class NeuralODE(nn.Module):
    def __init__(self, odefunc):
        super(NeuralODE, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0,1]).float()


    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_time)
        out = out[1]
        out = out[:,-1,:]
        return out