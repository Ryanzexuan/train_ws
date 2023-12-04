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
parser.add_argument('--n_input', type=int, default=3,
                    help='input dimension (default: 3)')
parser.add_argument('--n_out', type=int, default=1,
                    help='output dimension (default: 1)')
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
data_path = os.path.join(cur_path + '/data/data.csv')
print(f'data_path:{data_path}')
## To be done
def physics_loss(out, input):
    return 0

def train():
    rec_file = data_path
    raw_data  = pd.read_csv(rec_file)
    idx_data = raw_data['x_position_input']
    idx = np.array(np.where(idx_data == 0))
    print(f"idx:{idx}")
    raw_x = np.array(raw_data['x_position_input'])
    raw_y = np.array(raw_data['y_position_input'])
    raw_yaw = np.array(raw_data['yaw_input'])

    time_sequence = np.column_stack((raw_x, raw_y, raw_yaw))


    data_length = 1000 # whole data length
    sequence_length = 3 # net input length

    data_traj = TrajectoryDataset(time_sequence, sequence_length, idx)

    print(f"data_traj :{data_traj[0]}")

    data_loader = DataLoader(data_traj, batch_size=128, shuffle=True)


    # init training model
    input_size=3
    output_size=1
    hidden_size=64
    num_layers=3


    model = TCN_withMLP(input_size=3, output_size=1, num_channels=[16,16,16], kernel_size=3, dropout=0.3)
    # num_inputs:特征数
    # num_channels:每一层的输出特征数 


    # init Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 100
    losses = []  
    loss_infos = []

    bar = tqdm(range(num_epochs))

    for epoch in bar:
        for input_data, target_data in data_loader:
            # print(f"input data:{input_data}\n")
            # print(f"out value: {target_data}\n")
            optimizer.zero_grad()
            output = model(input_data)
            # print(f"out: {output}\n")
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
                'hidden_size': hidden_size,
                'output_size': output_size,
                'hidden_layers': num_layers
            }
    save_file_path ="/home/ryan/raigor/train_ws/results/model_fitting/TCN/tcn.pt"

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
    # print(f"test:{test}")
    # 
    # 
    # with torch.no_grad():
    #     # 假设你有一个输入序列 input_sequence_tensor
    #     predicted_sequence = model(test).squeeze()
    #     #print("Predicted Sequence:")
    #     # print(f"Predicted output:{predicted_sequence}")

    # # 绘制损失曲线
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 2, 1)  # 第一个子图用于损失曲线
    # plt.plot(losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training Loss")

    # # 绘制实际数据和预测数据
    # test_output= time_sequence[start + input_size: start + input_size + test_num]
    # actual_data = test_output  # 实际数据
    # predicted_data = predicted_sequence.numpy()  # 预测数据
    # # print(f"Predicted output:{predicted_data}")
    # plt.subplot(2, 2, 2)  # 第二个子图用于实际和预测数据
    # plt.plot(actual_data[:, 0], label="Actual x", marker='o')
    # plt.plot(predicted_data[:, 0], linestyle='--', label="Predicted x", marker='x')
    # plt.xlabel("Time Step")
    # plt.ylabel("Value")
    # plt.title("Actual vs. Predicted Data (x)")

    # plt.subplot(2, 2, 3) 
    # plt.plot(actual_data[:, 1], label="Actual y", marker='o')
    # plt.plot(predicted_data[:, 1], linestyle='--', label="Predicted y", marker='x')
    # plt.xlabel("Time Step")
    # plt.ylabel("Value")
    # plt.title("Actual vs. Predicted Data (y)")


    # plt.subplot(2, 2, 4) 
    # plt.plot(actual_data[:, 2], label="Actual z", marker='o')
    # plt.plot(predicted_data[:, 2], linestyle='--', label="Predicted z", marker='x')
    # plt.xlabel("Time Step")
    # plt.ylabel("Value")
    # plt.title("Actual vs. Predicted Data (z)")
    # plt.legend()



    # plt.tight_layout()  # 调整子图布局，使其更清晰
    # plt.show()



    # ## eval mode


    # # with torch.no_grad():
    # #     model.eval()
    # #     # print(f'input:{torch.randn(16,27,20)}')
    # #     print(f'model2 out:{model(torch.randn(16,27,20)).shape}')

if __name__ == "__main__":
    train()