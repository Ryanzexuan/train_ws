from torch import nn
from tcn_common_for_multi import TemporalConvNet, MLP
import torch.nn.functional as F
import ml_casadi.torch as mc
import casadi as cs
import torch
from ml_casadi.torch.modules import TorchMLCasadiModule
import numpy as np

class NormalizedTCN(mc.TorchMLCasadiModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_size = self.model.input_size
        self.output_size = self.model.output_size


    def forward(self, x):
        return self.model(x) 


class TCN_withMLP(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_withMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = MLP(num_channels[-1], output_size)

    def acceleration_vel(self, x, y):
        time_horizon = 5
        vel_type = 3 
        ##  casadi version
        new_x = torch.zeros((2*x.shape[0],x.shape[1])) # x 5*6
        acc = torch.zeros((x.shape[0],vel_type))
        vel = torch.zeros((x.shape[0],vel_type))
        ## numpy version
        # new_x = np.zeros((2*x.shape[0],x.shape[1]))
        # acc = np.zeros((x.shape[0],vel_type))
        # vel = np.zeros((x.shape[0],vel_type))

        # print(f'new_x shape:{acc.shape}')
        # print(f'x shape:{x.shape}')
        # print(f'y shape:{y.shape}')
        # print(f'slice_x:{x[:,1]}')
        print(f'x 1st:{x}')
        for i in range(x.shape[0]-1):
            acc[i,:] = x[i+1,:vel_type] - x[i,:vel_type]
        acc[-1,:] = y - x[-1,:vel_type]
        print(f'acc:{acc}')
        # vel
        # vel[:time_horizon-1,:] = x[1:,:vel_type]
        # print(f'function y:{y}')
        vel = torch.vstack((x[1:,:vel_type], y))
        # vel[time_horizon-1,:] = y
        print(f'vel:{vel}')

        # augment vel and acc
        new_x = torch.hstack((vel,acc))
        # print(f'x:{x[:,:vel_type]}')
        # print(f'y:{y}')
        # print(f'new_x:{new_x}')
        return new_x

    def forward(self, x):
        x = x.unsqueeze(0)
        # x is (batchsize, length, channel)
        # TCN input shape is (batchsize, channel, length)
        # linear input shape must be (batchsize, length, channel)
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2) # output is (batchsize, length, channel) length for each batch supposed to be 1
        # print(f'tcn out:{output.shape}')
        output = self.linear(output) # output is (batchsize, channel, length)
        # print(f'after mlp shape:{output.shape}')
        # print(f'tcn out:{output[:, -1, :].unsqueeze(1).shape}')
        out = output[:, -1, :]
        # print(f'tcn out without acc:{out}')
        out = self.acceleration_vel(x.squeeze(0),out)
        return out

class TCN_withMLP_casadi(TorchMLCasadiModule):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_withMLP_casadi, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = MLP(num_channels[-1], output_size)


    def forward(self, x):
        # output = self.tcn(x)# output is (batchsize, length, channel) length for each batch supposed to be 1
        # output = cs.transpose(output)
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2) 
        output = self.linear(output) # output is (batchsize, channel, length)
        # print(f'after mlp shape:{output.shape}')
        return output[-1, :].unsqueeze(1)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        
    def forward(self, x):
        # TCN input shape is (batchsize, channel, length) B,L,C
        # linear input shape must be (batchsize, length, channel)
        output = self.tcn(x.transpose(1,2)) # output is (batchsize, length, channel) length for each batch supposed to be 1
        # print(f'tcn out:{output.shape}')
        return output[:, -1, :].unsqueeze(1)


class NaiveTCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(NaiveTCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # original x (batch, lenth, channels)
        x = self.tcn(x.permute(0, 2, 1)) # tcn input (batch, channels, lenth) 
        print(f"Output shape after tcn: {x.shape}") # tcn output (batch, channels, lenth)
        x = x.permute(0, 2, 1)
        # print(f"Output shape after second permute: {x.shape}")
        x = self.fc(x) # # linear input (batch, lenth, channels),out (batch, channels, lenth)
        # print(f"fc out: {x.shape}")
        x = x.permute(0, 2, 1)
        # print(f"final out: {x.shape}")
        return x