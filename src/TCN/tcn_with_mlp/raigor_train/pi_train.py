import argparse
import torch
import os
from model import TCN_withMLP
from tcn_common_for_multi import TemporalConvNet
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tcn_data import TrajectoryDataset



parser = argparse.ArgumentParser(description='Sequence Modeling')
parser.add_argument('--lambda_value', type=float, default=0.2,
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

args = parser.parse_args()
print(args)
lambda_v = args.lambda_value
print(lambda_v)

cur_path = os.getcwd()
print(f'cur_path:{cur_path}')
data_path = os.path.join(cur_path + '/bag/train.csv') # 3490
print(f'data_path:{data_path}')
## To be done
def physics_loss(out, input):
    v_in = input[:, -1, 3]
    w_in = input[:, -1, 4]
    theta_in = input[:, -1, 5]
    # output
    vx_out = out[:, :, 0]
    vy_out = out[:, :, 1]
    w_out = out[:, :, 2]
    
    v_x_nominal = torch.unsqueeze(v_in * torch.cos(theta_in), 1)
    v_y_nominal = torch.unsqueeze(v_in * torch.sin(theta_in), 1)
    w_nominal = torch.unsqueeze(w_in, 1)
    nominal = torch.cat([v_x_nominal.unsqueeze(1), v_y_nominal.unsqueeze(1), w_nominal.unsqueeze(1)], dim=2)
    
    nominal_critic = nn.MSELoss()
    loss = nominal_critic(out, nominal)
    # print(f'loss:{loss}')
    return loss

def train():
    rec_file = data_path
    raw_data  = pd.read_csv(rec_file)
    
    print(f'data:{raw_data}')
    # state
    raw_x = np.array(raw_data['x_position_input'])
    raw_y = np.array(raw_data['y_position_input'])
    raw_yaw = np.array(raw_data['yaw_input'])
    # state dot
    raw_x_dot = np.array(raw_data['vel_linear_x'])
    raw_y_dot = np.array(raw_data['vel_linear_y'])
    raw_yaw_dot = np.array(raw_data['vel_angular_z']) # vel_w too small
    # time
    raw_con_t = np.array(raw_data['con_time'])
    # raw_state_t = np.array(raw_data['state_in_time']) 
    # control 
    raw_u_v = np.array(raw_data['con_x_input'])
    raw_u_w = np.array(raw_data['con_z_input'])
    
    # time_sequence = np.column_stack((raw_x, raw_y, raw_yaw, raw_v, raw_w))
    idx_data = raw_data['vel_linear_x'][:3000]
    idx = np.array(np.where(idx_data == 0))
    print(f"idx:{idx}")
    time_sequence = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w, raw_yaw)) # with control
    # time_sequence = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot))[:3000] # with no control

    sequence_length = 5 # net input length

    data_traj = TrajectoryDataset(time_sequence, sequence_length, idx, args.n_out)

    print(f"data_traj :{data_traj[0]}")

    data_loader = DataLoader(data_traj, batch_size=128, shuffle=True)


    # init training model
    input_size=args.n_input
    output_size=args.n_out # channels
    hidden_size=64
    num_layers=3
    num_channels = [16,16,16] # args.tcn_channels
    kernel_size = 3
    dropout = 0.3


    model = TCN_withMLP(input_size=input_size, output_size=3, num_channels=[16,16,16], kernel_size=3, dropout=0.3)
    # num_inputs:特征数
    # num_channels:每一层的输出特征数 


    # init Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 100
    losses = []  
    loss_infos = []

    cuda_name = 'cuda:0'
    if torch.cuda.is_available():
        model = model.to(cuda_name)

    # ### train ### #
    bar = tqdm(range(num_epochs))
    for epoch in bar:
        for input_data, target_data in data_loader:
            if torch.cuda.is_available():
                    # print(f"pre ground truth: {target_data.shape}\n")
                    input_data = input_data.to(cuda_name)
                    target_data = target_data[:,:,:args.n_out].to(cuda_name)
            optimizer.zero_grad()
            output = model(input_data)
            # print(f"input data:{input_data.shape}\n")
            # print(f"ground truth: {target_data.shape}\n")
            # print(f"net out: {output.shape}\n")
            loss = criterion(output, target_data) + lambda_v * physics_loss(output, input_data)
            # print(f'loss:{loss}')
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        train_loss_mean = np.mean(losses)
        loss_info = train_loss_mean
        loss_infos.append(loss_info)
        bar.set_description(f'Train Loss: {train_loss_mean:.6f}')
        bar.refresh()

    if True: 
        plt.plot(loss_infos)
        plt.show()
    hidden_size = [64, 32, 32]
    hidden_layers = 3
    # input_size, hidden_size, output_size, n_hidden, activation=None
    save_dict = {
                'state_dict': model.state_dict(),
                'input_size': input_size,
                'num_channels': num_channels,
                'output_size': output_size,
                'kernel_size': kernel_size,
                'dropout': dropout,
                'hidden_size': hidden_size,
                'mlp_inputsize': num_channels[-1],
                'mlp_outputsize': output_size,
                'hidden_layers': hidden_layers
            }
    save_file_path = os.path.join(cur_path + '/../results/raigor_pi.pt')

    torch.save(save_dict, save_file_path)

    # Set evaluate mode
    model.eval()  
    N = 100

    # getting slice
    start = 40
    test_num = 500
    test = []

    for i in range(test_num):
        test_input,_ = data_traj[i + start]
        # print(f"test input:{test_input}")
        input_sequence_tensor = test_input
        test.append(input_sequence_tensor)
    test = torch.tensor(np.array(test))
    
if __name__ == "__main__":
    train()