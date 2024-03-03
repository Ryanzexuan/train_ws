import torch
import os
from model import TCN_withMLP, TCN, NormalizedTCN
import casadi as cs
import numpy as np
from ml_casadi.torch.modules import TorchMLCasadiModule
from ml_casadi.torch.modules.nn import Linear
from ml_casadi.torch.modules.nn import activation as activations
from tcn_common_for_multi import TemporalBlock, TemporalConvNet, MLP
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import casadi as cs
import os
from nav_msgs.msg import Odometry
import rospy
from turtlesim.msg import Pose  # 导入 Pose 消息类型
from casadi_common import Normalized_TCN


# # 加载模型
# ws_path = os.getcwd()
# pt_name = 'ros_mlp.pt'
# path = os.path.join(ws_path + '/../results/raigor_pi.pt')
# print(f"path is {path}")
# saved_dict = torch.load(path)
# learn_model = TCN_withMLP(input_size=saved_dict['input_size'], output_size=saved_dict['output_size'], 
#                             num_channels=saved_dict['num_channels'], kernel_size=saved_dict['kernel_size'], dropout=saved_dict['dropout'])
# learn_model.load_state_dict(saved_dict['state_dict'])


# param = learn_model.tcn.network[0].conv1.weight_v

# # 创建两个向量
# vector1 = cs.MX([1, 2, 3])
# vector2 = cs.MX([4, 5, 6])

# # 使用 cs.dot 计算点积
# dot_product = cs.dot(vector1, vector2)
# # print(dot_product)


# # 将参数的值转换为 NumPy 数组
# param_values = param.data.numpy()

# # 打印参数的值
# # print(f'param values yzx get:{param_values}')


# # network0 = learn_model.tcn.network[0]
# # conv1_weight_v = network0.conv1.weight_v
# # conv1_weight_g = network0.conv1.weight_g
# # conv2_weight_v = network0.conv2.weight_v
# # conv2_weight_g = network0.conv2.weight_g
# # conv3_downsample = network0.downsample.weight
# # conv1_bias = network0.conv1.bias
# # conv2_bias = network0.conv2.bias
# # # print(f'conv_v:{conv1_bias}')
# # # print(f'conv_g:{conv1_weight_g.shape}')
# # # print(f'conv_v:{conv1_weight_v.shape}') 
# # print(f'conv3_downsample:{network0.downsample.weight.shape}') 
# # conv1_weight = conv1_weight_v * conv1_weight_g
# # conv2_weight = conv2_weight_v * conv2_weight_g
# # print(f'conv_weight:{conv1_weight.data.numpy().shape}')
# # # 复制 network1
# # network1 = learn_model.tcn.network[1]
# # conv1_weight_v1 = network1.conv1.weight_v
# # conv1_weight_g1 = network1.conv1.weight_g
# # conv2_weight_v1 = network1.conv2.weight_v
# # conv2_weight_g1 = network1.conv2.weight_g
# # conv1_bias1 = network1.conv1.bias
# # conv2_bias1 = network1.conv2.bias
# # conv1_weight1 = conv1_weight_v1 * conv1_weight_g1.unsqueeze(-1).unsqueeze(-1)
# # conv2_weight1 = conv2_weight_v1 * conv2_weight_g1.unsqueeze(-1).unsqueeze(-1)

# # # 复制 network2
# # network2 = learn_model.tcn.network[2]
# # conv1_weight_v2 = network2.conv1.weight_v
# # conv1_weight_g2 = network2.conv1.weight_g
# # conv2_weight_v2 = network2.conv2.weight_v
# # conv2_weight_g2 = network2.conv2.weight_g
# # conv1_bias2 = network2.conv1.bias
# # conv2_bias2 = network2.conv2.bias
# # conv1_weight2 = conv1_weight_v2 * conv1_weight_g2.unsqueeze(-1).unsqueeze(-1)
# # conv2_weight2 = conv2_weight_v2 * conv2_weight_g2.unsqueeze(-1).unsqueeze(-1)


# # 打印找到的卷积层
# # 查看模型参数
# for name, param in learn_model.named_parameters():
#     print(name, param)
# conv_layers = []
# for name, module in learn_model.named_children():
#     if isinstance(module, torch.nn.Conv1d):
#         conv_layers.append(name)
# print("Convolutional Layers:")
# print(conv_layers)
# print("Total Convolutional Layers:", len(conv_layers))


# print(f'linear get:{learn_model.linear.output_layer.weight.data.detach().numpy().shape}')




############# Be parallel to discrete_tcn_try2
t = np.linspace(0, 100, 50)
# print(len(t))
ref_len = len(t)
y_xpos =  (7 * np.sin(0.0156*t)).reshape(-1,1)
y_ypos = (7 - 7 * np.cos(0.0156*t)).reshape(-1,1)
y_yaw = (0.0156 * t).reshape(-1,1)
print(y_xpos.shape)

yref = np.hstack((np.zeros((ref_len,4)),y_xpos,np.zeros((ref_len,4)),y_ypos,np.zeros((ref_len,4)),y_yaw,np.zeros((ref_len,25))))
# yref = np.array([y_xpos,y_ypos, y_yaw]).T
print(yref.shape)
x_ref = []
for t, ref in enumerate(yref):
    x_ref.append(ref)

cur_rec_state_set = np.zeros(7)
cur_cmd = np.zeros(2)

# def Callback_base_turtle(msg):
#     rospy.loginfo("msg got~!!!!!")
#     # quaternion = msg.pose.pose.orientation
#     # rospy.loginfo(f"x pose{msg.pose.pose.position.x}")
#     cur_rec_state_set[0] = msg.x 
#     cur_rec_state_set[1] = msg.y 
#     cur_rec_state_set[2] = msg.theta
#     cur_rec_state_set[3] = rospy.Time.now().to_sec()
#     cur_rec_state_set[4] = msg.linear_velocity * np.cos(msg.theta)
#     cur_rec_state_set[5] = msg.linear_velocity * np.sin(msg.theta)
#     cur_rec_state_set[6] = msg.angular_velocity
#     rospy.loginfo(f'cur state:{cur_rec_state_set}')

# rospy.init_node("acados", anonymous=True)
# rospy.Subscriber("/turtle1/pose", Pose, Callback_base_turtle)
# rate = rospy.Rate(1)
# while not rospy.is_shutdown():
#     rate.sleep()






data = """0.335816002016689 0.010920714489099 0.106576585282543 0.6 0.2 0.01371263412115
0.664412777529619 0.020956249355815 0.181341725096883 0.6 0.2 0.033083066465925
0.699820755077229 0.041295144872296 0.112634651766008 0.6 0.2 0.057292978194316
0.696839677180168 0.049609039782608 0.122569128851088 0.6 0.2 0.076999218682932
0.675622452496484 0.065449401803523 0.152111546326436 0.6 0.2 0.099793699296486"""

data1 = """0.440361258563412 1.03508325263723 0.102814795592509 1 0.2 1.1722245749658
0.424147673651639 1.04007923898172 0.089554032089423 1.2 0.2 1.18518376801384
0.395005956853244 1.07431410662754 0.112769542820807 1.2 0.2 1.20104034177755
0.355754551910453 1.07196123927693 0.126837985774625 1.2 0.2 1.23133036455249
0.338797781283185 1.07769756858737 0.065586269263586 1.2 0.2 1.25531679606128"""

# 将文本数据转换为NumPy数组
array_data = np.array([list(map(float, line.split())) for line in data.split('\n')])
array_data1 = np.array([list(map(float, line.split())) for line in data1.split('\n')])

print(f'data gen!!!!!!!!!!!!!!!!!!!!!!!!')
print(array_data.shape)
print(array_data1.shape)


# Get git commit hash for saving the model
ws_path = os.getcwd()
pt_name = 'ros_mlp.pt'
path = os.path.join(ws_path + '/../results/raigor_pi.pt')
# print(f"path is {path}")
saved_dict = torch.load(path)

learn_model = TCN_withMLP(input_size=saved_dict['input_size'], output_size=saved_dict['output_size'], 
                        num_channels=saved_dict['num_channels'], kernel_size=saved_dict['kernel_size'], dropout=saved_dict['dropout'])
learn_model.load_state_dict(saved_dict['state_dict'])
learn_model.eval()
# Set evaluate mode
# model_100.eval()
# basic info  
# 示例用法
num_channels = saved_dict['num_channels']
sequence_len = 5
num_inputs = 6
num_outputs = 16
kernel_size = 3
stride = 1
dilation = 1
padding = (kernel_size-1) * dilation
dropout = 0.3

print(f'first step:Normalized_TCN')
tcn_model = Normalized_TCN(learn_model, num_inputs, num_channels, kernel_size, stride, dropout, saved_dict)
print(f'second step:DoubleIntegratorWithLearnedDynamics')
with torch.no_grad():
    tensor = torch.tensor(array_data, dtype=torch.float32).unsqueeze(0)
    print(f'tensor:{tensor.shape}')
    print(f'torch now!!!!!!!!!!!!!!!!!!!')
    print(f'torch tcn output:{learn_model(tensor)}')
print(f'tcn output1!!!!!!!!!!!!!!!!!!!')
print(f'{tcn_model.forward(array_data1)}')
print(f'tcn output!!!!!!!!!!!!!!!!!!!')
print(f'{tcn_model.forward(array_data)}')