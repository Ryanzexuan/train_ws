from torch import nn
from tcn_common_for_multi import TemporalConvNet, MLP
import torch.nn.functional as F


class TCN_withMLP(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_withMLP, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = MLP(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed
        # TCN input shape is (batchsize, channel, length)
        # linear input shape must be (batchsize, length, channel)
        output = self.tcn(x).transpose(1, 2) # output is (batchsize, length, channel) length for each batch supposed to be 1
        # print(f'tcn out:{output.shape}')
        output = self.linear(output[:, -1, :].unsqueeze(1)).transpose(1, 2) # output is (batchsize, channel, length)
        return self.sig(output)


