import argparse
import torch
import os
from model import TCN_withMLP,TCN
from tcn_common_for_multi import TemporalConvNet
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tcn_data import TrajectoryDataset
import random


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

        # print(f'raw x:{raw_y}')
        # time_sequence = np.column_stack((raw_x, raw_y, raw_yaw))
        # time_sequence = np.column_stack((raw_x_dot[:1000], raw_y[:1000], raw_yaw[:1000]))
        # time_sequence = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w))# with control
        time_sequence_withyaw = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w, raw_yaw))# with control and yaw
        time_sequence_noyaw = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w))
        print(f'time sequence :{time_sequence_noyaw}')
        data_length = 1000 # whole data length
        sequence_length = 5 # net input length

        data_traj_100 = TrajectoryDataset(time_sequence_noyaw, sequence_length, idx, saved_dict_100['output_size'])
        data_traj_110 = TrajectoryDataset(time_sequence_withyaw, sequence_length, idx, saved_dict_110['output_size'])
        data_traj_111 = TrajectoryDataset(time_sequence_withyaw, sequence_length, idx, saved_dict_111['output_size'])

        # print(f"data_traj :{data_traj[0]}")

        data_loader_100 = DataLoader(data_traj_100, batch_size=128, shuffle=True)
        data_loader_110 = DataLoader(data_traj_110, batch_size=128, shuffle=True)
        data_loader_111 = DataLoader(data_traj_111, batch_size=128, shuffle=True)


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
        model_100.eval()  
        ## model 110 read
        model_110 = TCN_withMLP(input_size=saved_dict_110['input_size'], output_size=saved_dict_110['output_size'], 
                            num_channels=saved_dict_110['num_channels'], kernel_size=saved_dict_110['kernel_size'], dropout=saved_dict_110['dropout'])
        
        model_110.load_state_dict(saved_dict_110['state_dict'])
        # Set evaluate mode
        model_110.eval()  

        ## model 111 read
        model_111 = TCN_withMLP(input_size=saved_dict_111['input_size'], output_size=saved_dict_111['output_size'], 
                            num_channels=saved_dict_111['num_channels'], kernel_size=saved_dict_111['kernel_size'], dropout=saved_dict_111['dropout'])
        
        model_111.load_state_dict(saved_dict_111['state_dict'])
        # Set evaluate mode
        model_111.eval()  

        N = 100

        criterion = nn.MSELoss()
        # getting slice
        test_num = 10
        start = random.randint(0, time_sequence_withyaw.shape[0]-test_num-1)
        test_100 = []
        test_110 = []
        test_111 = []

        losses_100 = []
        losses_110 = []
        losses_111 = []

        # print(f"actual_data:{actual_data}")

        real_data = []
        for i in range(test_num):
            print(f'start:{start}')
            test_input_100,real_out = data_traj_100[i + start]
            test_input_110,_ = data_traj_110[i + start]
            test_input_111,_ = data_traj_111[i + start]
            real_data.append(real_out)
            test_100.append(test_input_100)
            test_110.append(test_input_110)
            test_111.append(test_input_111)
        test_100 = torch.tensor(np.array(test_100))
        test_110 = torch.tensor(np.array(test_110))
        test_111 = torch.tensor(np.array(test_111))
        real_data = np.array(real_data).squeeze()
        print(f"test input:{test_100[0]}")
        print(f"real_data[start]:{real_data[0]}")
        
        with torch.no_grad():
            # 假设你有一个输入序列 input_sequence_tensor
            predicted_sequence_100 = model_100(test_100).squeeze()
            predicted_sequence_110 = model_110(test_110).squeeze()
            predicted_sequence_111 = model_111(test_111).squeeze()
            vx,vy,w = nominal_calc(test_111)
            nominal_loss = nominal_loss_calc(vx, vy, w, real_data)

        # Record Losses
        for i, item in enumerate(predicted_sequence_100):
            loss = criterion(item, torch.tensor(real_data[i, :args.n_out]))
            losses_100.append(torch.sqrt(loss))
        for i, item in enumerate(predicted_sequence_110):
            loss = criterion(item, torch.tensor(real_data[i, :args.n_out]))
            losses_110.append(torch.sqrt(loss))
        for i, item in enumerate(predicted_sequence_111):
            loss = criterion(item, torch.tensor(real_data[i, :args.n_out]))
            losses_111.append(torch.sqrt(loss))
        # Show average loss for each model
        losses_100_mean = np.mean(losses_100)
        losses_110_mean = np.mean(losses_110)
        losses_111_mean = np.mean(losses_111)
        nominal_mean = np.mean(nominal_loss)
        print(f'mean loss for model ctrl:{losses_100_mean}')
        print(f'mean loss for model c+yaw:{losses_110_mean}')
        print(f'mean loss for model c+yaw+PI:{losses_111_mean}')
        print(f'mean loss for model Nominal:{nominal_mean}')


        # 绘制实际数据和预测数据
        predicted_sequence_100 = predicted_sequence_100.numpy()  # 预测数据
        predicted_sequence_110 = predicted_sequence_110.numpy()  # 预测数据
        predicted_sequence_111 = predicted_sequence_111.numpy()  # 预测数据
        print(f'size of actual data:{real_data.shape[0]}')
        print(f'size of predict data:{predicted_sequence_100.shape[0]}')
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)  # 第一个子图用于损失曲线
        plt.plot(losses_100, label="ctrl loss", marker='o', color='green')
        plt.plot(losses_110,label="c+yaw  loss", marker='x', color='orange')
        plt.plot(losses_111,label="c+yaw+PI loss", marker='*', color='purple')
        plt.plot(nominal_loss,label="Nominal loss", marker='^', color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("testing Loss")
        plt.legend()
        # 第二个子图用于实际和预测数据
        plt.subplot(2, 2, 2)  
        plt.plot(real_data[:, 0], label="Actual x", marker='o', color='blue')
        if '1' in str(selection):
            plt.plot(predicted_sequence_100[:, 0], linestyle='--', label="ctrl Predicted x", marker='x',color='green')
        if '2' in str(selection):
            plt.plot(predicted_sequence_110[:, 0], linestyle='--', label="c+yaw Predicted x", marker='x',color='orange')
        if '3' in str(selection):
            plt.plot(predicted_sequence_111[:, 0], linestyle='--', label="c+yaw+PI Predicted x", marker='x',color='purple')
        plt.plot(vx.numpy(), linestyle='--', label="Nominal x", marker='*',color='red')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Actual vs. Predicted Data (x)")
        plt.legend()

        plt.subplot(2, 2, 3) 
        plt.plot(real_data[:, 1], label="Actual y", marker='o',color='blue')
        if '1' in str(selection):
            plt.plot(predicted_sequence_100[:, 1], linestyle='--', label="ctrl Predicted y", marker='x',color='green')
        if '2' in str(selection):
            plt.plot(predicted_sequence_110[:, 1], linestyle='--', label="c+yaw Predicted y", marker='x',color='orange')
        if '3' in str(selection):
            plt.plot(predicted_sequence_111[:, 1], linestyle='--', label="c+yaw+PI Predicted y", marker='x',color='purple')
        plt.plot(vy.numpy(), linestyle='--', label="Nominal y", marker='*',color='red')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Actual vs. Predicted Data (y)")
        plt.legend()


        plt.subplot(2, 2, 4) 
        plt.plot(real_data[:, 2], label="Actual z", marker='o', color='blue')
        if '1' in str(selection):
            plt.plot(predicted_sequence_100[:, 2], linestyle='--', label="ctrl Predicted z", marker='x', color='green')
        if '2' in str(selection):
            plt.plot(predicted_sequence_110[:, 2], linestyle='--', label="c+yaw Predicted z", marker='x', color='orange')
        if '3' in str(selection):
            plt.plot(predicted_sequence_111[:, 2], linestyle='--', label="c+yaw+PI Predicted z", marker='x', color='purple')
        plt.plot(w.numpy(), linestyle='--', label="Nominal w", marker='*', color='red')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Actual vs. Predicted Data (z)")
        plt.legend()


        plt.suptitle(f"start:{start}, end:{start+test_num}")
        plt.tight_layout()  # 调整子图布局，使其更清晰
        plt.show()



if __name__ == "__main__":
    test()