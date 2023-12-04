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
        self.input_and_out_length = sequence_length + 1
        self.start_indices = start_dice.squeeze(0)
        print(f'read data shape:{data.shape}')
        _,self.feature_num = data.shape
        self.traj_num = start_dice.shape[1]
        print(f'traj num : {self.traj_num}')

    def __len__(self):
        # if self.traj_num == 1:
        # # If there is only one trajectory, use its length directly
        #     return self.start_indices[0] - self.sequence_length
        # else:
        # # Calculate the total length excluding overlaps
        #     return sum([(self.start_indices[i+1] - self.start_indices[i] - self.sequence_length) for i in range(self.traj_num-1)])
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        # print(f"index:{index}")
        # 从数据中提取序列
        input_sequence = np.random.rand(self.sequence_length, 3).astype(np.float32)
        output_value = np.random.rand(1, 3).astype(np.float32)
        input_sequence = self.data[index : index + self.sequence_length]
        output_value = self.data[index + self.sequence_length]
        # print(f"pre item:{input_sequence}\n,{output_value}\n")
        # print(f"item:{input_sequence}\n,{output_value}\n")

        first_index = np.argmax(self.start_indices - index > 0) 
        # print(f'first index:{first_index}')
        # print(f'{self.start_indices[first_index]}')
        remain_num = self.start_indices[first_index] - index
        if remain_num <= (self.sequence_length + 1):
            start = self.start_indices[first_index] - self.input_and_out_length
            input_sequence = self.data[start : start + self.sequence_length]
            output_value = self.data[start + self.sequence_length]

        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        output_value = torch.tensor(output_value, dtype=torch.float32)
        # input_sequence = self.data[trajectory_index, start_index:end_index, :]
        # target_sequence = self.data[trajectory_index, end_index, :]
        return input_sequence, output_value



