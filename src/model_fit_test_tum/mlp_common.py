import numpy as np
import sys
import casadi as cs
from torch.utils.data import Dataset

import ml_casadi.torch as mc


class RawDataset(Dataset):
    def __init__(self,dataset):
        
        self.x_raw = None
        self.x_out_raw = None
        self.u_raw = None
        self.y_raw = None
        self.x_pred_raw = None
        self.dt_raw = None

        self.load_data(dataset)

    def load_data(self,ds):
        if isinstance(ds, np.lib.npyio.NpzFile):
            x_raw = ds['state_in']
            
            x_out = ds['state_out']
            x_pred = ds['state_pred']
            u_raw = ds['input_in']
            dt = ds["dt"]
            # print("is there?")
        else:
            x_raw = self.undo_jsonify(ds['state_in'].to_numpy())
            # print(x_raw[1,7]) # 8 9 10 columns
            x_out = self.undo_jsonify(ds['state_out'].to_numpy())
            # print(x_out.shape)
            # print(x_out[0,:])
            x_pred = self.undo_jsonify(ds['state_pred'].to_numpy())
            # print(x_pred[0,:])
            u_raw = self.undo_jsonify(ds['input_in'].to_numpy())
            dt = ds["dt"].to_numpy()
            # print("is there?")
        invalid = np.where(dt == 0)

        # Remove invalid entries (dt = 0)
        x_raw = np.delete(x_raw, invalid, axis=0)
        x_out = np.delete(x_out, invalid, axis=0)
        x_pred = np.delete(x_pred, invalid, axis=0)
        u_raw = np.delete(u_raw, invalid, axis=0)
        dt = np.delete(dt, invalid, axis=0)

        # Rotate velocities to body frame and recompute errors
        x_raw = world_to_body_velocity_mapping(x_raw)
        x_pred = world_to_body_velocity_mapping(x_pred)
        x_out = world_to_body_velocity_mapping(x_out)
        y_err = x_out - x_pred

        # Normalize error by window time (i.e. predict error dynamics instead of error itself)
        y_err /= np.expand_dims(dt, 1)

        # Select features
        self.x_raw = x_raw
        self.x_out_raw = x_out
        self.u_raw = u_raw
        # self.y_raw = y_err # when using learning residuals
        self.y_raw = x_out  # when using next_state
        self.x_pred_raw = x_pred
        self.dt_raw = dt

    def get_x(self, cluster=None, pruned=True, raw=False):
        # 合并原始输入数据和控制数据
        x_f = self.x_features
        u_f = self.u_features
        data = np.concatenate((self.x_raw[:, x_f], self.u_raw[:, u_f]), axis=1) if u_f else self.x_raw[:, x_f]
        print(f'u_f:{self.u_raw[0,[]]}')
        print(data[1])
        return data

    @property
    def x(self):
        return self.get_x()

    def get_x_out(self, cluster=None, pruned=True):

        if cluster is not None:
            assert pruned

        if pruned or cluster is not None:
            data = self.x_out_raw[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

            return data

        return self.x_out_raw[tuple(self.pruned_idx)] if pruned else self.x_out_raw

    @property
    def x_out(self):
        return self.get_x_out()

    def get_u(self, cluster=None, pruned=True, raw=False):

        if cluster is not None:
            assert pruned

        if raw:
            return self.u_raw[tuple(self.pruned_idx)] if pruned else self.u_raw

        data = self.u_raw[:, self.u_features] if self.u_features is not None else self.u_raw
        data = data[:, np.newaxis] if len(data.shape) == 1 else data

        if pruned or cluster is not None:
            data = data[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

        return data

    @property
    def u(self):
        return self.get_u()

    def get_y(self, cluster=None, pruned=True, raw=False):
       return self.y_raw[self.pruned_idx] 


    @property
    def y(self):
        return self.get_y()



    

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
        # print("forward")
        return (self.model((x - self.x_mean) / self.x_std) * self.y_std) + self.y_mean

    def cs_forward(self, x):
        return (self.model((x - self.x_mean.cpu().numpy()) / self.x_std.cpu().numpy()) * self.y_std.cpu().numpy()) + self.y_mean.cpu().numpy()
    
    def forward_test(self, x):
        return self.model(x)

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
        # time
        control_time = self.data['con_time']
        output_state_time = self.data['output_time']
        dt = output_state_time - control_time

        cur_state_x = self.data['x_position_input']
        cur_state_y = self.data['y_position_input']
        cur_state_yaw = self.data['yaw_input']
        cur_vel_x = self.data['con_x_input']
        cur_vel_y = self.data['con_z_input']
        print(dt)
        print(np.where(dt == 0))
        # self.lenth = cur_vel_x.shape[0]
        cur_dt = dt
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


def world_to_body_velocity_mapping(state_sequence):
    p, q, v_w, w = separate_variables(state_sequence)
    v_b = []
    for i in range(len(q)):
        v_b.append(v_dot_q(v_w[i], quaternion_inverse(q[i])))
    v_b = np.stack(v_b)
    return np.concatenate((p, q, v_b, w), 1)

def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)

def separate_variables(traj):
    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]
    return [p_traj, a_traj, v_traj, r_traj]
def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return cs.mtimes(rot_mat, v)

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

    else:
        rot_mat = cs.vertcat(
            cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat