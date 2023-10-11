import numpy as np
import sys
from torch.utils.data import Dataset

import ml_casadi.torch as mc


class RawDataset(Dataset):
    def __init__(self,dataset):
        
        ## raw data init
        # control input
        self.vel_x_input = None
        self.vel_z_input = None
        # state output
        self.state_pose_x = None
        self.state_pose_y = None
        self.state_orientation_y = None
        self.load_data(dataset)

    def load_data(self,ds):
        # control input
        vel_x_input = ds['vel_x']
        vel_z_input = ds['vel_z']
        # state output
        state_pose_x = ds['x_position']
        state_pose_y = ds['y_position'] 
        # state_orientation_y = ds['']
        
        self.vel_x_input = vel_x_input
        self.vel_z_input = vel_z_input
        self.state_pose_x = state_pose_x
        self.state_pose_y = state_pose_y
        # self.state_orientation_y = state_orientation_y

    @property
    def x(self):
        return self.getx()

    def getx(self):
        data = np.column_stack((self.vel_x_input, self.vel_z_input))
        # print(data.shape[1])
        return data
    
    @property
    def y(self):
        return self.gety()
    
    def gety(self):
        data = np.column_stack((self.state_pose_x, self.state_pose_y))
        return data


class MlpDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.x = dataset.x
        self.y = dataset.y
        # print(self.x.shape)
        # print(self.y.shape)

    
    def __len__(self):
        return len(self.x)

    def __getitem__(self,item):
        return self.x[item], self.y[item]


    def stats(self):
        x_mean = np.hstack([self.x.mean(axis=0), np.zeros(2)])
        x_std = np.hstack([self.x.std(axis=0), np.ones(2)])
        # print(x_mean)
        return self.x.mean(axis=0), self.x.std(axis=0), self.y.mean(axis=0), self.y.std(axis=0)



class NormalizedMlp(mc.TorchMLCasadiModule):
    def __init__(self, model, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.model = model
        self.input_size = self.model.input_size
        self.output_size = self.model.output_size
        self.register_buffer('x_mean', x_mean)
        self.register_buffer('x_std', x_std)
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('y_std', y_std)

    def forward(self, x):
        # print(x.shape)
        # print(self.x_mean.shape)

        # print(self.)
        return (self.model((x - self.x_mean) / self.x_std) * self.y_std) + self.y_mean

    def cs_forward(self, x):
        return (self.model((x - self.x_mean.cpu().numpy()) / self.x_std.cpu().numpy()) * self.y_std.cpu().numpy()) + self.y_mean.cpu().numpy()