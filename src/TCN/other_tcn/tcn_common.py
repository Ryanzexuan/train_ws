import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# 定义 Chomp1d 模块
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
# 定义 TemporalBlock 模块
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# 定义 TemporalConvNet 模块
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 定义 TCN 模型（根据之前的 TCN 架构进行修改）
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))  # 调整数据维度
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x


    
    import numpy as np
import torch

def get_training_datasets(data, target_len, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split the data into training, validation, and testing datasets.
    
    Parameters:
    - data: Numpy array containing input sequences.
    - target_len: Length of the target sequence (e.g., 1 for next time step).
    - train_ratio: Ratio of data used for training (default is 0.7).
    - val_ratio: Ratio of data used for validation (default is 0.15).
    - test_ratio: Ratio of data used for testing (default is 0.15).
    
    Returns:
    - x_train, x_val, x_test, y_train, y_val, y_test: PyTorch tensors containing data splits.
    """
    
    total_len = len(data)
    
    # Calculate the split indices
    train_split = int(train_ratio * total_len)
    val_split = train_split + int(val_ratio * total_len)
    
    # Split the data
    x_train = data[:train_split, :-target_len, :]
    y_train = data[:train_split, target_len:, :]
    
    x_val = data[train_split:val_split, :-target_len, :]
    y_val = data[train_split:val_split, target_len:, :]
    
    x_test = data[val_split:, :-target_len, :]
    y_test = data[val_split:, target_len:, :]
    
    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    
    x_val = torch.tensor(x_val).float()
    y_val = torch.tensor(y_val).float()
    
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()
    
    return x_train, x_val, x_test, y_train, y_val, y_test

# Example usage:
# Assuming you have your 'data' as a numpy array and 'target_len' as the length of the target sequence
# x_train, x_val, x_test, y_train, y_val, y_test = get_training_datasets(data, target_len=1)
