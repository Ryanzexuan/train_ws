import torch
import os
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import ml_casadi.torch as mc


class TrajectoryDataset(Dataset):
    def __init__(self, data, sequence_length, start_dice):
        self.data = data
        self.sequence_length = sequence_length
        self.start_indices = start_dice
        self.traj_num,_,self.feature_num = data.shape

    def __len__(self):
        return sum([(self.start_indices[i+1] - self.start_indices[i] - self.sequence_length) for i in range(self.traj_num-1)])
        # return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        # print(f"index:{index}")
        # 从数据中提取序列
        input_sequence = np.random.rand(self.sequence_length, 3).astype(np.float32)
        output_value = np.random.rand(1, 3).astype(np.float32)
        input_sequence = self.data[index : index + self.sequence_length]
        output_value[0] = self.data[index + self.sequence_length]
        # print(f"pre item:{input_sequence}\n,{output_value}\n")

        # 转换为 PyTorch 的 Tensor
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        output_value = torch.tensor(output_value, dtype=torch.float32)
        # print(f"item:{input_sequence}\n,{output_value}\n")


        # trajectory_index = 0
        # while index >= (self.start_indices[trajectory_index + 1] - self.start_indices[trajectory_index] - self.sequence_length):
        #     index -= (self.start_indices[trajectory_index + 1] - self.start_indices[trajectory_index] - self.sequence_length)
        #     trajectory_index += 1

        # start_index = self.start_indices[trajectory_index] + index
        # end_index = start_index + self.sequence_length

        # input_sequence = self.data[trajectory_index, start_index:end_index, :]
        # target_sequence = self.data[trajectory_index, end_index, :]


        return input_sequence, output_value

# TCN model
class TCNModeltrain(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(TCNModeltrain, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(hidden_size, output_size)
        # self.alias_mapping = {
        #     'tcn.0.weight': 'model.tcn.0.weight',
        #     'tcn.0.bias': 'model.tcn.0.bias',
        #     'tcn.2.weight': 'model.tcn.2.weight',
        #     'tcn.2.bias': 'model.tcn.2.bias',
        #     'tcn.4.weight': 'model.tcn.4.weight',
        #     'tcn.4.bias': 'model.tcn.4.bias',
        #     'fc.weight': 'model.fc.weight',
        #     'fc.bias': 'model.fc.bias',
        # }

    def forward(self, x):
        # print(f"Input shape before permute: {x.shape}")
        x = self.tcn(x.permute(0, 2, 1))  
        # print(f"Output shape after permute: {x.shape}")
        x = x.permute(0, 2, 1)
        # print(f"Output shape after second permute: {x.shape}")
        # x = self.tcn(x)  
        x = self.fc(x)
        # print(f"fc out: {x.shape}")
        x = x.permute(0, 2, 1)
        # print(f"final out: {x.shape}")
        return x

# 定义 TCN 模型
class TCNModel(mc.TorchMLCasadiModule):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(TCNModel, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        with torch.no_grad():
            self.fc.weight.fill_(0.)
            self.fc.bias.fill_(0.)

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))  # 调整数据维度
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x


class NormalizedTCN(mc.TorchMLCasadiModule):
    def __init__(self, model, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.model = model
        self.input_size = self.model.input_size
        self.output_size = self.model.output_size
        # self.register_buffer('x_mean', x_mean)
        # self.register_buffer('x_std', x_std)
        # self.register_buffer('y_mean', y_mean)
        # self.register_buffer('y_std', y_std)

    def forward(self, x):
        return (self.model((x - self.x_mean) / self.x_std) * self.y_std) + self.y_mean

    def cs_forward(self, x):
        return (self.model((x - self.x_mean.cpu().numpy()) / self.x_std.cpu().numpy()) * self.y_std.cpu().numpy()) + self.y_mean.cpu().numpy()
