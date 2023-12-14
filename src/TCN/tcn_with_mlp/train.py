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
parser.add_argument('--lambda_value', type=float, default=0.7,
                     help='hyperparameter for PI Loss (default=0.7)')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (default: 0.3)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--n_input', type=int, default=5,
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
data_path = os.path.join(cur_path + '/data/data2.csv')
print(f'data_path:{data_path}')
## To be done
def physics_loss(out, input):
    return 0

def train():
    rec_file = data_path
    raw_data  = pd.read_csv(rec_file)
    idx_data = raw_data['vel_x_input']
    idx = np.array(np.where(idx_data == 0))
    print(f"idx:{idx}")
    print(f'data:{raw_data}')
    # state
    raw_x = np.array(raw_data['x_position_input'])
    raw_y = np.array(raw_data['y_position_input'])
    raw_yaw = np.array(raw_data['yaw_input'])
    # state dot
    raw_x_dot = np.array(raw_data['vel_x_input'])
    raw_y_dot = np.array(raw_data['vel_y_input'])
    raw_yaw_dot = np.array(raw_data['vel_w_input']) # vel_w too small
    # time
    raw_con_t = np.array(raw_data['con_time'])
    raw_state_t = np.array(raw_data['state_in_time']) 
    # control 
    raw_u_v = np.array(raw_data['con_x_input'])
    raw_u_w = np.array(raw_data['con_z_input'])
    
    # time_sequence = np.column_stack((raw_x, raw_y, raw_yaw, raw_v, raw_w))
    time_sequence = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w)) # with control
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

    save_dict = {
                'state_dict': model.state_dict(),
                'input_size': input_size,
                'num_channels': num_channels,
                'output_size': output_size,
                'kernel_size': kernel_size,
                'dropout': dropout
            }
    save_file_path = os.path.join(cur_path + '/results/tcn.pt')

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