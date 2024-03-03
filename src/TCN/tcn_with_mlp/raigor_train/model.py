from torch import nn
from tcn_common_for_multi import TemporalConvNet, MLP
import torch.nn.functional as F
import ml_casadi.torch as mc


class NormalizedTCN(mc.TorchMLCasadiModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        


class TCN_withMLP(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_withMLP, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = MLP(num_channels[-1], output_size)


    def forward(self, x):
        # TCN input shape is (batchsize, length, channel)
        # linear input shape must be (batchsize, length, channel)
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2) # output is (batchsize, length, channel) length for each batch supposed to be 1
        # print(f'tcn out:{output.shape}')
        output = self.linear(output) # output is (batchsize, channel, length)
        # print(f'after mlp shape:{output.shape}')
        return output[:, -1, :].unsqueeze(1)


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