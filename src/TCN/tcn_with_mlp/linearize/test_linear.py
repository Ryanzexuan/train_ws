import argparse
import torch
import os
from model import TCN_withMLP, TCN, NormalizedTCN
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tcn_data import TrajectoryDataset
import random
import casadi as cs


parser = argparse.ArgumentParser(description='Sequence Modeling')
parser.add_argument('--lambda_value', type=float, default=0.7,
                     help='hyperparameter for PI Loss (default=0.7)')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (default: 0.3)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--n_input', type=int, default=6,
                    help='input dimension (default: 3)')
parser.add_argument('--n_out', type=int, default=3,
                    help='output dimension (default: 3)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batchsize (default: 128)')
parser.add_argument('--tcn_channels', nargs='+', type=int, default=[16,16,16],
                    help='dropout applied to layers (default: [16,16,16])')
parser.add_argument('--select', type=int, default=123)

args = parser.parse_args()
print(args)
lambda_v = args.lambda_value
print(lambda_v)
selection = args.select

cur_path = os.getcwd()
print(f'cur_path:{cur_path}')
# uneven_manual, data, dataset_gzsim_nominal, tele_random, uneven, uneven.mpc 
data_path = os.path.join(cur_path + '/data/test/data.csv')
pt_path_withoutYaw = os.path.join(cur_path + '/results/tcn.pt')
pt_path_withPI = os.path.join(cur_path + '/results/PItcn_withyaw.pt')
pt_path_withoutPI = os.path.join(cur_path + '/results/tcn_withyaw.pt')
print(f'data_path:{data_path}')
# print(f'pt_path:{pt_path}')
### ctrl + yaw + PI loss : 111 means has all
saved_dict_100 = torch.load(pt_path_withoutYaw)
saved_dict_110 = torch.load(pt_path_withoutPI)
saved_dict_111 = torch.load(pt_path_withPI)



def nominal_calc(input):
    # print(f'input:{input}')
    v = input[:, -1, 3]
    w = input[:, -1, 4]
    theta = input[:, -1, 5]
    # print(f'v:{v.shape}')
    v_x = v * np.cos(theta)
    v_y = v * np.sin(theta)
    w = w
    # print(f'v_x:{v_x}')
    # print(f'v_y:{v_y}')
    # print(f'w:{w}')
    return v_x,v_y,w
def nominal_loss_calc(vx, vy, w, real_data):
    print(f'real_data:{real_data.shape}')
    print(vx.unsqueeze(1).shape)
    nominal = torch.cat([vx.unsqueeze(1), vy.unsqueeze(1), w.unsqueeze(1)], dim=1)
    print(f'nominal:{nominal.shape}')
    nominal_critic = nn.MSELoss()
    print(f'data:{nominal.shape},{torch.tensor(real_data[:, :3]).shape}')
    out2 = torch.tensor(real_data[:, :3])
    out1 = nominal
    loss = []
    for i, (elem1, elem2) in enumerate(zip(out1, out2)):
        loss.append(torch.sqrt(nominal_critic(elem1, elem2)))
    # loss = nominal_critic(out1, out2)
    print(f'loss:{len(loss)}')
    return loss


def test():
    rec_file = data_path
    raw_data  = pd.read_csv(rec_file)
    if raw_data.empty:
        print(f'not reading anything!!!')
    else:
        # print(f'raw data:{raw_data}')
        idx_data = raw_data['x_position_input']
        idx = np.array(np.where(idx_data == 0))
        # print(f"idx:{idx}")
        # state
        raw_yaw = np.array(raw_data['yaw_input'])
        # state dot
        raw_x_dot = np.array(raw_data['vel_x_input'])
        raw_y_dot = np.array(raw_data['vel_y_input'])
        raw_yaw_dot = np.array(raw_data['vel_w_input']) # vel_w too small
        # print(f'raw x dot{raw_x_dot}')
        # control 
        raw_u_v = np.array(raw_data['con_x_input'])
        raw_u_w = np.array(raw_data['con_z_input'])
        # print(f'raw x dot{raw_u_v}')
        time_sequence_noyaw = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w))
        print(f'time sequence :{time_sequence_noyaw}')
        data_length = 1000 # whole data length
        sequence_length = 5 # net input length

        data_traj_100 = TrajectoryDataset(time_sequence_noyaw, sequence_length, idx, saved_dict_100['output_size'])
        data_loader_100 = DataLoader(data_traj_100, batch_size=128, shuffle=True)


        # init training model
        input_size=args.n_input
        output_size=args.n_out # channels
        hidden_size=64
        num_layers=3
        num_channels = [16,16,16] # args.tcn_channels
        kernel_size = 3
        dropout = 0.3
        # print(saved_dict['num_channels'])
        
        
        ## Model 100 read
        model_100 = TCN_withMLP(input_size=saved_dict_100['input_size'], output_size=saved_dict_100['output_size'], 
                            num_channels=saved_dict_100['num_channels'], kernel_size=saved_dict_100['kernel_size'], dropout=saved_dict_100['dropout'])
        model_100.load_state_dict(saved_dict_100['state_dict'])
        # Set evaluate mode
        # model_100.eval()  
        model = NormalizedTCN(model_100)
        model.eval()

        
        # input:
        # input:[[ 0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00]
        #  [-9.21757e-04 -2.71761e-04  3.06975e-03  4.77448e-01  1.38134e-01]
        #  [ 6.90237e-03 -7.76679e-04  1.46043e-02  4.77406e-01  1.38061e-01]
        #  [ 4.11539e-02 -2.25573e-03  5.54318e-02  4.76764e-01  1.37465e-01]
        #  [ 8.36845e-02 -3.78852e-03  7.14965e-02  4.75102e-01  1.36501e-01]]
        # out:tensor([[ 0.1257, -0.0079,  0.0219,  0.6252,  0.8857]])
        #
        with torch.no_grad():
            input, output = data_traj_100[0]
            print(f'input:{np.array(input)}')
            print(f'out:{output}')
            print(f'model value :{model(torch.tensor(input).unsqueeze(0))}')
            input = torch.tensor(input).unsqueeze(0)
            vel = cs.MX.sym('vel', 3)
            u = cs.MX.sym('u', 2)
            input_def = cs.vertcat(vel, u)
            
            linear_model = model.approx(input_def)
            p = model.sym_approx_params(order=1, flat=True)
            params = model.approx_params(input, flat=True, order=1)# obtain value:(a, f_a, vec(df_a))
            a = params
            f_a = params  
            df_a = params
            
            print(f'linear model:{linear_model}')
            print(f'p:{p}')
            print(f'params:{params}')
            print(f'a:{a}')
            print(f'f_a:{f_a}')
            print(f'df_a:{df_a}')
        # N = 100

        # criterion = nn.MSELoss()
        # # getting slice
        # test_num = 10
        # start = random.randint(0, time_sequence_noyaw.shape[0]-test_num-1)
        # test_100 = []
        


if __name__ == "__main__":
    test()