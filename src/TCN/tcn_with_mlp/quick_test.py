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

args = parser.parse_args()
print(args)
lambda_v = args.lambda_value
print(lambda_v)

cur_path = os.getcwd()
print(f'cur_path:{cur_path}')
data_path = os.path.join(cur_path + '/data/test/tele_random.csv')
pt_path = os.path.join(cur_path + '/results/tcn_withyaw.pt')
print(f'data_path:{data_path}')
print(f'pt_path:{pt_path}')
saved_dict = torch.load(pt_path)


def nominal_calc(input):
    print(f'input:{input}')
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
        time_sequence = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w, raw_yaw))# with control and yaw
        print(f'time sequence :{time_sequence}')
        data_length = 1000 # whole data length
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
        # print(saved_dict['num_channels'])
        
        # model = TCN(input_size=saved_dict['input_size'], output_size=saved_dict['output_size'], 
        #             num_channels=saved_dict['num_channels'], kernel_size=saved_dict['kernel_size'], dropout=saved_dict['dropout'])
        
        model = TCN_withMLP(input_size=saved_dict['input_size'], output_size=saved_dict['output_size'], 
                            num_channels=saved_dict['num_channels'], kernel_size=saved_dict['kernel_size'], dropout=saved_dict['dropout'])
        
        model.load_state_dict(saved_dict['state_dict'])
        # Set evaluate mode
        model.eval()  
        N = 100

        criterion = nn.MSELoss()
        # getting slice
        start = random.randint(0, time_sequence.shape[0]-sequence_length-1)
        test_num = 100
        test = []
        losses = []

        test_output= time_sequence[start + input_size: start + input_size + test_num]
        actual_data = test_output  # 实际数据
        # print(f"actual_data:{actual_data}")

        real_data = []
        for i in range(test_num):
            test_input,real_out = data_traj[i + start]
            # print(f"test input:{test_input}")
            input_sequence_tensor = test_input
            real_data.append(real_out)
            test.append(input_sequence_tensor)
        test = torch.tensor(np.array(test))
        real_data = np.array(real_data).squeeze()
        print(f"test:{test}")
        print(f"data_traj[start]:{data_traj[start]}")
        print(f"real_data[start]:{real_data[0]}")
        print(f'actual_data[start]:{actual_data[0]}')
        
        with torch.no_grad():
            # 假设你有一个输入序列 input_sequence_tensor
            predicted_sequence = model(test).squeeze()
            # print(f"Predicted Sequence:{predicted_sequence}")
            # print(f"Predicted output:{predicted_sequence}")\
        # Record Losses
        for i, item in enumerate(predicted_sequence):
            loss = criterion(item, torch.tensor(test_output[i, :args.n_out]))
            # print(f'i {i}')
            # print(f'predict:{item}')
            # print(f'actual value:{test_output[i][:3]}')
            # print(f'losses:{loss}')
            losses.append(torch.sqrt(loss))
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)  # 第一个子图用于损失曲线
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        # 绘制实际数据和预测数据
        predicted_data = predicted_sequence.numpy()  # 预测数据
        print(f'size of actual data:{real_data.shape[0]}')
        print(f'size of predict data:{predicted_data.shape[0]}')
        vx,vy,w = nominal_calc(test)
        # print(f"Predicted output:{predicted_data}")
        plt.subplot(2, 2, 2)  # 第二个子图用于实际和预测数据
        plt.plot(real_data[:, 0], label="Actual x", marker='o')
        plt.plot(predicted_data[:, 0], linestyle='--', label="Predicted x", marker='x')
        plt.plot(vx.numpy(), linestyle='--', label="Nominal x", marker='*')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Actual vs. Predicted Data (x)")

        plt.subplot(2, 2, 3) 
        plt.plot(real_data[:, 1], label="Actual y", marker='o')
        plt.plot(predicted_data[:, 1], linestyle='--', label="Predicted y", marker='x')
        plt.plot(vy.numpy(), linestyle='--', label="Nominal y", marker='*')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Actual vs. Predicted Data (y)")


        plt.subplot(2, 2, 4) 
        plt.plot(real_data[:, 2], label="Actual z", marker='o')
        plt.plot(predicted_data[:, 2], linestyle='--', label="Predicted z", marker='x')
        plt.plot(w.numpy(), linestyle='--', label="Nominal w", marker='*')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Actual vs. Predicted Data (z)")
        plt.legend()


        plt.suptitle(f"start:{start}, end:{start+test_num}")
        plt.tight_layout()  # 调整子图布局，使其更清晰
        plt.show()



if __name__ == "__main__":
    test()