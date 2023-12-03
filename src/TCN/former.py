import torch
import os
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tcn_common import TCNModeltrain, TrajectoryDataset




data_length = 1000
sequence_length = 3

rec_file = "/home/ryan/raigor/train_ws/data/simplified_sim_dataset/train/dataset_addref.csv"
raw_data  = pd.read_csv(rec_file)
idx_data = raw_data['x_position_input']
idx = np.array(np.where(idx_data == 0))
print(f"idx:{idx}")
raw_x = np.array(raw_data['x_position_input'])
raw_y = np.array(raw_data['y_position_input'])
raw_yaw = np.array(raw_data['yaw_input'])




data = np.random.rand(6, data_length, 3).astype(np.float32)
# 迭代处理原始数据
j = 0
last_idx = 0
for i in idx[0]:
    if i == 0:
        continue
    time_sequence = np.column_stack((raw_x[last_idx:i], raw_y[last_idx:i], raw_yaw[last_idx:i]))
    time_sequence_length = min(sequence_length, time_sequence.shape[0])
    data[j, :time_sequence_length, :] = time_sequence[:time_sequence_length]
    j = j + 1
    last_idx = i

# 创建示例数据
# first row is which sequence, second row is which timestamp within one sequence
data = np.random.rand(sequence_length, data_length, 3).astype(np.float32)
input_sequence = data[:, :-1, :]  # 历史数据作为输入序列  
# print(f"input_sequence:{input_sequence.shape[0]}")
target_sequence = data[:, 1:, :]   # 下一时刻的数据作为目标序列
# print(f"target_sequence:{target_sequence}")

# 转换数据为 PyTorch 张量
input_sequence = torch.from_numpy(input_sequence)
target_sequence = torch.from_numpy(target_sequence)
print(f"pre item:{input_sequence}\n,{target_sequence}\n")
# 创建数据加载器
dataset = TensorDataset(input_sequence, target_sequence)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"dataset is: {dataset}")

# 创建 TCN 模型
input_size=3
output_size=3
hidden_size=64
num_layers=3

model = TCNModeltrain(input_size=3, output_size=3, hidden_size=64, num_layers=3)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
# 训练模型
num_epochs = 100
losses = []  # 用于保存每个epoch的损失值

for epoch in range(num_epochs):
    for input_data, target_data in data_loader:
        # print(f"input data:{input_data}\n")
        # print(f"out value: {target_data}\n")
        optimizer.zero_grad()
        output = model(input_data)
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
save_file_path ="/home/ryan/raigor/train_ws/results/model_fitting/TCN/tcn.pt"

torch.save(save_dict, save_file_path)

# 使用训练好的模型进行预测
model.eval()  # 设置为评估模式
N = 100
with torch.no_grad():
    # 假设你有一个输入序列 input_sequence_tensor
    input_sequence_tensor = input_sequence[0:1,46:49]  # 选择第一个序列作为示例
    predicted_sequence = model(input_sequence_tensor)
    #print("Predicted Sequence:")
    #print(predicted_sequence)

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)  # 第一个子图用于损失曲线
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

# 绘制实际数据和预测数据
actual_data = input_sequence[0,46:50].numpy()  # 实际数据
predicted_data = predicted_sequence[0].numpy()  # 预测数据

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



