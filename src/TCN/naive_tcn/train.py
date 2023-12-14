import torch
import os
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tcn_common_for1 import TCNModeltrain, TrajectoryDataset
from tqdm import tqdm




data_length = 1000
sequence_length = 3
traj_num = 1


cur_path = os.getcwd()
print(f'cur_path:{cur_path}')
data_path = os.path.join(cur_path + '/data/data.csv')
print(f'data_path:{data_path}')
raw_data  = pd.read_csv(data_path)
idx_data = raw_data['x_position_input']
idx = np.array(np.where(idx_data == 0))
print(f"idx:{idx}")
raw_x = np.array(raw_data['vel_x_input'])
raw_y = np.array(raw_data['vel_y_input'])
raw_yaw = np.array(raw_data['vel_w_input'])

time_sequence = np.column_stack((raw_x[:1000], raw_y[:1000], raw_yaw[:1000]))

data_traj = TrajectoryDataset(time_sequence, sequence_length, idx)

print(f"data_traj :{data_traj[0]}")
data_loader = DataLoader(data_traj, batch_size=64, shuffle=True)


# 创建 TCN 模型
input_size=3
output_size=1 # not means channels, channels change through tcn
hidden_size=64
num_layers=3

model = TCNModeltrain(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
# 训练模型
num_epochs = 100
losses = []  # 用于保存每个epoch的损失值


for epoch in tqdm(range(num_epochs), desc = 'processing'):
    for input_data, target_data in data_loader:
        optimizer.zero_grad()
        output = model(input_data)
        print(f"input data shape:{input_data.shape}\n")
        print(f"true out value shape : {target_data.shape}\n")
        print(f"out shape: {output.shape}\n")
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

save_dict = {
            'state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'hidden_layers': num_layers
        }
save_file_path = os.path.join(cur_path + '/results/tcn.pt')

torch.save(save_dict, save_file_path)

# 使用训练好的模型进行预测
model.eval()  # 设置为评估模式
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
with torch.no_grad():
    # 假设你有一个输入序列 input_sequence_tensor
    predicted_sequence = model(test).squeeze()
    #print("Predicted Sequence:")
    # print(f"Predicted output:{predicted_sequence}")

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)  # 第一个子图用于损失曲线
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

# 绘制实际数据和预测数据
test_output= time_sequence[start + input_size: start + input_size + test_num]
actual_data = test_output  # 实际数据
predicted_data = predicted_sequence.numpy()  # 预测数据
# print(f"Predicted output:{predicted_data}")
plt.subplot(2, 2, 2)  # 第二个子图用于实际和预测数据
plt.plot(actual_data[:, 0], label="Actual x", marker='o')
plt.plot(predicted_data[:, 0], linestyle='--', label="Predicted x", marker='x')
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("Actual vs. Predicted Data (x)")

plt.subplot(2, 2, 3) 
plt.plot(actual_data[:, 1], label="Actual y", marker='o')
plt.plot(predicted_data[:, 1], linestyle='--', label="Predicted y", marker='x')
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("Actual vs. Predicted Data (y)")


plt.subplot(2, 2, 4) 
plt.plot(actual_data[:, 2], label="Actual z", marker='o')
plt.plot(predicted_data[:, 2], linestyle='--', label="Predicted z", marker='x')
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("Actual vs. Predicted Data (z)")
plt.legend()



plt.tight_layout()  # 调整子图布局，使其更清晰
plt.show()



