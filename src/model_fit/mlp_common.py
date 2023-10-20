import numpy as np
import sys
import casadi as cs
from torch.utils.data import Dataset

import ml_casadi.torch as mc


class RawDataset(Dataset):
    def __init__(self,dataset):
        
        ## raw data init
        # time
        self.control_time = None
        self.output_state_time = None
        self.dt = None
        # state input
        self.state_pose_x_input = None
        self.state_pose_y_input = None
        self.state_orientation_y_input = None   
        # control input
        self.vel_x_input = None
        self.vel_z_input = None
        # state output
        self.state_pose_x_output = None
        self.state_pose_y_output = None
        self.state_orientation_y_output = None
        # nomial state == ref state
        self.nomial_x = None
        self.nomial_y = None
        self.nomial_yaw =  None
        # residuals
        self.res_x = None
        self.res_y =  None
        self.res_yaw = None

        self.load_data(dataset)

    def load_data(self,ds):
        # time
        control_time = ds['con_time']
        output_state_time = ds['output_time']
        dt = output_state_time - control_time
        # state input
        state_pose_x_input = ds['x_position_input']
        state_pose_y_input = ds['y_position_input']
        state_orientation_y_input = ds['yaw_input']  
        # control input
        vel_x_input = ds['con_x_input']
        vel_z_input = ds['con_z_input']
        # state output
        state_pose_x_output = ds['x_position_output']
        state_pose_y_output = ds['y_position_output'] 
        state_orientation_y_output = ds['yaw_output']
        # nomial state
        nomial_state_x = ds['nomial_x']
        nomial_state_y = ds['nomial_y']
        nomial_state_yaw = ds['nomial_yaw']

        # time
        self.dt = dt
        # state input
        self.state_pose_x_input = state_pose_x_input
        self.state_pose_y_input = state_pose_y_input
        self.state_orientation_y_input = state_orientation_y_input   
        # control input
        self.vel_x_input = vel_x_input
        self.vel_z_input = vel_z_input
        # state output
        self.state_pose_x_output = state_pose_x_output
        self.state_pose_y_output = state_pose_y_output
        self.state_orientation_y_output = state_orientation_y_output
        # nomial state 
        self.nomial_x = nomial_state_x
        self.nomial_y = nomial_state_y
        self.nomial_yaw = nomial_state_yaw
        # resdiuals = output - nomial
        self.res_x = self.state_pose_x_output - self.nomial_x
        self.res_y = self.state_pose_y_output - self.nomial_y
        self.res_yaw = self.state_orientation_y_output - self.nomial_yaw


    @property
    def x(self):
        return self.getx()

    def getx(self):
        data = np.column_stack((self.state_pose_x_input,self.state_pose_y_input,self.state_orientation_y_input,self.vel_x_input, self.vel_z_input,self.dt)) # 6 dims
        # print(data.shape[1])
        return data
    
    @property
    def y(self):
        return self.gety()
    
    def gety(self):
        data = np.column_stack((self.res_x, self.res_y, self.res_yaw)) # 3 dims
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
        return (self.model((x - self.x_mean) / self.x_std) * self.y_std) + self.y_mean

    def cs_forward(self, x):
        return (self.model((x - self.x_mean.cpu().numpy()) / self.x_std.cpu().numpy()) * self.y_std.cpu().numpy()) + self.y_mean.cpu().numpy()


class NomialModel:
    def __init__(self,data,row_dim,state_col_dim):
        self.data = data
        self.model_expl = np.zeros((row_dim, state_col_dim))
        # self.model_expl = np.array([[]])
        self.u = cs.MX.sym('u', 3)
        self.x = cs.MX.sym('x', 3)
        self.dt = cs.MX.sym('dt', 1)
        self.nomial_model()
        self.extract_state()
        

    def extract_state(self):
        cur_state_x = self.data['x_position_input']
        cur_state_y = self.data['y_position_input']
        cur_state_yaw = self.data['yaw_input']
        cur_vel_x = self.data['con_x_input']
        cur_vel_y = self.data['con_z_input']
        self.lenth = cur_vel_x.shape[0]
        cur_dt = np.zeros((self.lenth,1)) + 0.01
        # print(cur_dt.shape[0])
        self.x = np.column_stack((cur_state_x, cur_state_y, cur_state_yaw))
        self.u = np.column_stack((cur_vel_x, cur_vel_y, cur_dt))
        self.dt = cur_dt

        
    def nomial_model(self):
        u = self.u
        x = self.x
        dt =self.dt
        rhs = [x[0]+u[0]*cs.cos(x[2])*dt,x[1]+u[1]*cs.sin(x[2])*dt,x[2]+u[1]*dt]
        self.f = cs.Function('nomial', [x, u, dt], [cs.vcat(rhs)], ['x', 'u', 'dt'],['next_state_nomial'])    

    def calc_nomial(self):  
        # print(self.u.shape[0])
        # print(self.x.shape[0])
        # self.model_expl = []
        # print(self.f(self.x[0], self.u[0], self.dt[0])[0])
        for j in range(self.x.shape[0]):
            self.model_expl[j,0] = self.f(self.x[j], self.u[j], self.dt[j])[0]
            self.model_expl[j,1] = self.f(self.x[j], self.u[j], self.dt[j])[1]
            self.model_expl[j,2] = self.f(self.x[j], self.u[j], self.dt[j])[2]
        return self.model_expl
