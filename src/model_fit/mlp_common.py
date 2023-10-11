import numpy as np
from torch.utils.data import Dataset

import ml_casadi.torch as mc


class RawDataset(Dataset):
    def __init__(self,dataset):
        
        ## raw data init
        # control input
        self.vel_x_input = None
        self.vel_y_input = None
        # state output
        self.state_pose_x = None
        self.state_pose_y = None
        self.state_orientation_y = None


        self.load_data(dataset)

    def load_data(self,ds):
        # control input
        vel_x_input = ds['']
        vel_y_input = ds['']
        # state output
        state_pose_x = ds['']
        state_pose_y = ds[''] 
        state_orientation_y = ds['']
        
        self.vel_x_input = vel_x_input
        self.vel_y_input = vel_y_input
        self.state_pose_x = state_pose_x
        self.state_pose_y = state_pose_y
        self.state_orientation_y = state_orientation_y


    def getx(self):
        data = np.concatenate(self.vel_x_input, self.vel_y_input)
        return data
    
    def gety(self):
        data = np.concatenate(self.state_pose_x, self.state_pose_y, self.state_orientation_y)
        return data

    @property
    def x(self):
        return self.getx()

    @property
    def y(self):
        return self.gety()



class MlpDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.x = dataset.x
        self.y = dataset.y


class NormalizedMlp(mc.TorchMLCasadiModule):
    def __init__(self, model, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.model = model