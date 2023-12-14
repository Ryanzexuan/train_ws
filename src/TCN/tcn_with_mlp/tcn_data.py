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
    def __init__(self, data, sequence_length, start_dice, output_channelsize):
        self.data = data
        self.sequence_length = sequence_length
        self.output_size = output_channelsize
        self.input_and_out_length = sequence_length + 1
        self.start_indices = np.insert(start_dice, 0, 0) if start_dice.size == 0 or start_dice[0] != 0 else start_dice.squeeze(0)

        print(f'read data shape:{data.shape}')
        _,self.feature_num = data.shape
        self.traj_num = 1 if start_dice.size == 0 else start_dice.shape[1]
        print(f'traj num : {self.traj_num}')

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
       
        # 从数据中提取序列
        # print(f'data shape:{self.data.shape[1]}')
        input_sequence = np.random.rand(self.sequence_length, self.data.shape[1]).astype(np.float32)
        output_value = np.random.rand(1, self.data.shape[1]).astype(np.float32)
        input_sequence = self.data[index : index + self.sequence_length]
        output_value[:, :self.output_size] = self.data[index + self.sequence_length, :self.output_size]
        # print(f"pre item:{input_sequence}\n,{output_value}\n")
        # print(f"input:{input_sequence.shape}\n,ouput:{output_value.shape}\n")
        search = index - self.start_indices
        min_search_positive = np.min(search[search >= 0])
        first_index = np.where(search == min_search_positive)[0][0]
        # first_index = np.argmin((index - self.start_indices) > 0) - 1
        print(f"index:{index}")
        print(f'first index:{first_index}')
        print(f'{self.start_indices}')
        if first_index + 1 == self.start_indices.shape[0]:
            print(f'final traj')
            remain_num = self.data.shape[0] - index
        else:
            print(f'not final traj')
            remain_num = self.start_indices[first_index + 1] - index
        first_index = first_index + 1
        if remain_num <= (self.sequence_length + 1):
            # print(f'yes')
            start = self.start_indices[first_index] - self.input_and_out_length
            input_sequence = self.data[start : start + self.sequence_length]
            output_value[:, :self.output_size] = self.data[start + self.sequence_length, :self.output_size]

        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        output_value = torch.tensor(output_value, dtype=torch.float32)

        
        ## print shape
        # print(f"input:{input_sequence.shape}\n,ouput:{output_value.shape}\n")
        ## print value
        # print(f"input:{input_sequence}\n,ouput:{output_value}\n")
        return input_sequence, output_value



