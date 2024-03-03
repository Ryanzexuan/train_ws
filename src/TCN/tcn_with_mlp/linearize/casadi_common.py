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
from model import TCN_withMLP, TCN, NormalizedTCN


def weight_norm_conv1d(x, kernel, bias=None, stride=1, dilation=1, padding=0):
    """
    一维卷积的NumPy实现，包含了stride、padding和dilation参数。

    参数：
    - x: 输入数据，numpy 数组，形状为 (batch_size, in_channels, input_length)
    - kernel: 卷积核，numpy 数组，形状为 (out_channels, in_channels, kernel_size)
    - bias: 偏置，numpy 数组，形状为 (out_channels,)
    - stride: 步幅，默认为 1
    - dilation: 膨胀率，默认为 1
    - padding: 填充，默认为 0

    返回：
    - 卷积结果，numpy 数组，形状为 (batch_size, out_channels, output_length)
    """
    # batch_size, in_channels, input_length = x.shape
    in_channels, input_length = x.shape
    out_channels, _, kernel_size = kernel.shape

    # 计算输出长度
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    # print(f'in channel:{in_channels}')
    # print(f'input_len:{input_length}')
    # print(f'output_len:{output_length}')
    # print(f'out_channel:{out_channels}')
    
    # 填充输入数据
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')

    # 初始化输出结果
    # output = np.zeros((batch_size, out_channels, output_length))
    output = np.zeros((out_channels, output_length))

    # 进行卷积计算
    for i in range(0, input_length - dilation * (kernel_size - 1) + 1, stride):
        window = x_padded[:, :, i:i + dilation * (kernel_size - 1) + 1:dilation]
        output[:, :, i  // stride + padding] = np.sum(window * kernel, axis=(1, 2))
    print(f'out:{output.shape}')
    # 添加偏置
    if bias is not None:
        bias = np.random.normal(0, 0.01, size=(out_channels, output_length))
        bias = bias[np.newaxis, :, :]
        print(f'bias:{bias.shape}')
        output += bias
        # output += bias[:, np.newaxis, np.newaxis] 

    return output

def conv1d_casadi(x, kernel, bias=None, stride=1, dilation=1, padding=0):
    # print(f'kernel shape:{kernel.shape}')
    input_length, in_channels = x.shape
    out_channels, _, kernel_size = kernel.shape
    kernel_origin = kernel
    # print(f'x shape:{x.shape}')
    # print(f'kernel var:{kernel.shape}')
    # print(f'padding size:{padding}')
    # print(f'dilation size:{dilation}')
    # kernel = kernel[1]
    # 计算输出长度
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    # print(f'in channel:{in_channels}')
    # print(f'input_len:{input_length}')
    # print(f'output_len:{output_length}')
    # print(f'kernel shape:{kernel[2].shape}')
    # 填充输入数据
    # print(f'cscsc:{cs.DM.zeros((padding, in_channels)).shape}')
    x_padded = cs.transpose(cs.vertcat(cs.DM.zeros((padding, in_channels)),
                          x,
                          cs.DM.zeros((padding, in_channels))))
   
    # 初始化输出结果
    output = cs.DM.zeros((out_channels, output_length))
    # 初始化输出结果
    output_mx = cs.MX.zeros((out_channels, output_length))
    # kernel_mx = cs.DM.zeros(out_channels*in_channels*kernel_size,1)
    # kernel_mx = cs.reshape(kernel_mx, (out_channels, in_channels, kernel_size))
    # print(f'output_mx:{output_mx.shape}')
    # print(f'x_padded shape after transpose:{x_padded.shape}')
    # print(f'range:{x_padded.shape[1] - dilation * (kernel_size - 1) + 1}')
    # 进行卷积计算
    for j in range(out_channels):
        # print(f'j:{j}')
        kernel = kernel_origin[j]
        # print(f'kernel:{kernel}')
        for i in range(0, x_padded.shape[1] - dilation * (kernel_size - 1), stride):
            # print(f'kernel first shape:{kernel.shape}')
            # print(f'i:{i}')
            # print(f'x_padded:{x_padded.shape}')
            if (i + dilation * (kernel_size - 1) + 1) > x_padded.shape[1]:
                window_mx = x_padded[:, i::dilation]
            else:
                window_mx = x_padded[:, i:i + dilation * (kernel_size - 1) + 1:dilation]  # 保留所有通道的时间窗口
            # print(f'window_mx shape :{window_mx.shape}')
            # print(f'kernel shape:{kernel.shape}')
            # print(f'kernel type:{window_mx.dtype}')
            cs_dot = cs.dot(kernel, window_mx)
            # print(f'csdot:{cs_dot}')
            output_mx[0, i // stride] = cs_dot
            # print(f'output_mx:{output_mx.shape}')
    # print(f'output_mx:{output_mx.shape}')
    out = output_mx[1,:] + 0.1
    print(f'out shape:{out.shape}')
    print(f'out:{out}')
    # print(f'out:{output_mx[1,:]}')
    # print(f'test:{out}')
    # 添加偏置
    if bias is not None:
        # print(f'bias shape:{bias.shape}')
        for k in range(bias.shape[0]):
            out[k,:] += bias[k]
        # bias_add = cs.vertcat(bias, cs.DM.zeros((out_channels - 1, output_length)))
        # print(f'bias add shape:{bias.shape}')
    # print(f'conv before chomp:{output_mx.shape}')
    return out
# def relu(x):
#     return np.maximum(0, x)





class Chomp1d:
    def __init__(self, chompsize):
        self.chomp_size = chompsize
    
    def forward(self, x):
        print(f'x in chompd:{x}')
        out_channels, output_length = x.shape
        out = cs.MX.zeros((out_channels, output_length))
        out = x[:, :-self.chomp_size]
        return out

class TemporalBlock_numpy():
    def __init__(self, n_inputs, n_outputs, stride, dilation, padding, tcn_params, dropout=0.2):
        super().__init__()
        self.n_input = n_inputs
        self.n_output = n_outputs
        # print(f'params:{tcn_params}')

        self.conv1_weight = tcn_params['conv1_weight']
        self.conv1_bias = tcn_params['conv1_bias']
        self.conv2_weight = tcn_params['conv2_weight']
        self.conv2_bias = tcn_params['conv2_bias']
        self.downsample_weight = tcn_params['downsample_conv'] if tcn_params['downsample_conv'] is not None else None
        self.downsample_bias = tcn_params['downsample_bias'] if tcn_params['downsample_bias'] is not None else None


        self.chomp_size = padding 
        self.chomp1 = Chomp1d(self.chomp_size)
        self.chomp2 = Chomp1d(self.chomp_size)
        self.act = getattr(activations, 'ReLU')()
        self.dropout = dropout
        self.padding = padding
        self.stride = stride
        self.dilation = dilation


    def cs_forward(self, x):
        # print(f'conv 1')
        print(f'x:{x}')
        out = conv1d_casadi(x, self.conv1_weight, self.conv1_bias, 
                                  stride=self.stride, dilation=self.dilation, padding=self.padding)
        print(f'out in between before chomp:{out}')
        out = self.chomp1.forward(out)
        # print(f'chomp1_shape:{out.shape}')
        dropout_shape = out.shape
        print(f'out in between:{out}')
        out_1 = self.act(out)
        droupout_mask = 1-self.dropout
        out = out_1 * droupout_mask
        # print(f'conv1 out shape:{out.shape}')
        # first conv finally done!!
        # print(f'conv 2')
        out = conv1d_casadi(cs.transpose(out), self.conv2_weight, self.conv2_bias,
                                  stride=self.stride, dilation=self.dilation, padding=self.padding)
        out = self.chomp2.forward(out)
        out_2 = self.act(out)
        droupout_mask = 1-self.dropout
        out = out_2 * droupout_mask
        # print(f'conv2 out shape:{out.shape}')
        if self.n_input != self.n_output:
            res = conv1d_casadi(x, self.downsample_weight, self.downsample_bias)
        else:
            print(f'not first block!!!!!!')
            res = cs.transpose(x)
        # print(f'res shape:{res.shape}')
        # print(f'block out shape:{self.act(out + res).shape}')
        # print(res)
        return self.act(out + res)


class Linear_tcn(TorchMLCasadiModule, torch.nn.Linear):
    def __init__(self, input_size, hidden_size, w, bias) -> None:
        self.weight = w
        self.bias = bias

    def cs_forward(self, x):
        # print(f'x cs_forward shape:{x.shape}')
        # print(f'weight shape:{self.weight.shape}')
        # assert x.shape[1] == 1, 'Casadi can not handle batches.'
        # print(x.shape)
        # print(f'weight:{self.weight.detach().numpy().shape}')
        y = cs.mtimes(self.weight, x)
        if self.bias is not None:
            y = y + self.bias
        return y




class Normalized_TCN(TorchMLCasadiModule):
    def __init__(self, learn_model, num_inputs, num_channels, kernel_size, stride, dropout, net_dict):
        super().__init__()
        dilation_size = np.zeros(len(num_channels)).astype(int)
        padding_size = np.zeros(len(num_channels)).astype(int)

        # print(f'num_channels:{num_channels}')
        network0 = learn_model.tcn.network[0]
        conv1_weight_v = network0.conv1.weight_v
        conv1_weight_g = network0.conv1.weight_g
        conv2_weight_v = network0.conv2.weight_v
        conv2_weight_g = network0.conv2.weight_g
        network0_conv1_bias = (network0.conv1.bias).data.detach().numpy()
        network0_conv2_bias = (network0.conv2.bias).data.detach().numpy()
        network0_conv1_weight = (conv1_weight_v * conv1_weight_g).data.detach().numpy() 
        network0_conv2_weight = (conv2_weight_v * conv2_weight_g).data.detach().numpy() 
        network0_downsample_conv = network0.downsample.weight.data.detach().numpy() 
        network0_downsample_bias = network0.downsample.bias .data.detach().numpy() 
        # network1
        network1 = learn_model.tcn.network[1]
        conv1_weight_v1 = network1.conv1.weight_v
        conv1_weight_g1 = network1.conv1.weight_g
        conv2_weight_v1 = network1.conv2.weight_v
        conv2_weight_g1 = network1.conv2.weight_g
        network1_conv1_bias = (network1.conv1.bias).data.detach().numpy()
        network1_conv2_bias = (network1.conv2.bias).data.detach().numpy()
        network1_conv1_weight = (conv1_weight_v1 * conv1_weight_g1).data.detach().numpy() 
        network1_conv2_weight = (conv2_weight_v1 * conv2_weight_g1).data.detach().numpy() 

        # network2
        network2 = learn_model.tcn.network[2]
        conv1_weight_v2 = network2.conv1.weight_v
        conv1_weight_g2 = network2.conv1.weight_g
        conv2_weight_v2 = network2.conv2.weight_v
        conv2_weight_g2 = network2.conv2.weight_g
        network2_conv1_bias = (network2.conv1.bias).data.detach().numpy()
        network2_conv2_bias = (network2.conv2.bias).data.detach().numpy()
        network2_conv1_weight = (conv1_weight_v2 * conv1_weight_g2).data.detach().numpy() 
        network2_conv2_weight = (conv2_weight_v2 * conv2_weight_g2).data.detach().numpy() 
        # print(f'TCN init')

        # mlp params
        fc1_w = learn_model.linear.fc1.weight.detach().numpy()
        fc1_bias = learn_model.linear.fc1.bias.detach().numpy()
        fc2_w = learn_model.linear.fc2.weight.detach().numpy()
        fc2_bias = learn_model.linear.fc2.bias.detach().numpy()
        fc3_w = learn_model.linear.fc3.weight.detach().numpy()
        fc3_bias = learn_model.linear.fc3.bias.detach().numpy()
        output_layer_w = learn_model.linear.output_layer.weight.detach().numpy()
        output_layer_bias = learn_model.linear.output_layer.bias.detach().numpy()

        # TCN params
        network0_parameters = {
            'conv1_bias': network0_conv1_bias,
            'conv2_bias': network0_conv2_bias,
            'conv1_weight': network0_conv1_weight,
            'conv2_weight': network0_conv2_weight,
            'downsample_conv': network0_downsample_conv,
            'downsample_bias': network0_downsample_bias
        }
        network1_parameters = {
            'conv1_bias': network1_conv1_bias,
            'conv2_bias': network1_conv2_bias,
            'conv1_weight': network1_conv1_weight,
            'conv2_weight': network1_conv2_weight,
            'downsample_conv': None,
            'downsample_bias': None
        }
        network2_parameters = {
            'conv1_bias': network2_conv1_bias,
            'conv2_bias': network2_conv2_bias,
            'conv1_weight': network2_conv1_weight,
            'conv2_weight': network2_conv2_weight,
            'downsample_conv': None,
            'downsample_bias': None
        }

        # MLP
        mlp_input_size = net_dict['mlp_inputsize']
        mlp_output_size = net_dict['mlp_outputsize']
        mlp_hidden_size = net_dict['hidden_size']
        mlp_n_hidden = net_dict['hidden_layers']
        mlp_activation = 'ReLU'

        # TCN net init
        for i in range(len(num_channels)):
            dilation_size[i] = 2**i
            padding_size[i] = int((kernel_size-1) * dilation_size[i])
        # print(f'padding size:{padding_size}')
        # print(f'paramstcn:{network0_parameters}')
        self.block1 = TemporalBlock_numpy(num_inputs, num_channels[0], stride, dilation_size[0], padding_size[0], network0_parameters, dropout)
        self.block2 = TemporalBlock_numpy(num_channels[0], num_channels[1], stride, dilation_size[1], padding_size[1], network1_parameters, dropout)
        self.block3 = TemporalBlock_numpy(num_channels[1], num_channels[2], stride, dilation_size[2], padding_size[2], network2_parameters, dropout)
        
        
        # MLP init
        assert mlp_n_hidden >= 1, 'There must be at least one hidden layer'
        self.input_size = mlp_input_size
        self.output_size = mlp_output_size
        self.input_layer = Linear_tcn(mlp_input_size, mlp_hidden_size[0], fc1_w, fc1_bias)

        hidden = []
        
        hidden.append((Linear_tcn(mlp_hidden_size[0], mlp_hidden_size[1], fc2_w, fc2_bias)))
        hidden.append((Linear_tcn(mlp_hidden_size[1], mlp_hidden_size[2], fc3_w, fc3_bias)))

        self.hidden_layers = torch.nn.ModuleList(hidden)

        self.output_layer = Linear_tcn(mlp_hidden_size[-1], mlp_output_size, output_layer_w, output_layer_bias)

        if mlp_activation is None:
            self.act = lambda x: x
        elif type(mlp_activation) is str:
            # print(f'getting activation part!!')
            self.act = getattr(activations, mlp_activation)()
        else:
            self.act = mlp_activation

    def acceleration_vel(self, x, y):
        time_horizon = 5
        vel_type = 3 
        ##  casadi version
        new_x = cs.MX.zeros((2*x.shape[0],x.shape[1])) # x 5*6
        acc = cs.MX.zeros((x.shape[0],vel_type))
        vel = cs.MX.zeros((x.shape[0],vel_type))
        ## numpy version
        # new_x = np.zeros((2*x.shape[0],x.shape[1]))
        # acc = np.zeros((x.shape[0],vel_type))
        # vel = np.zeros((x.shape[0],vel_type))

        print(f'new_x shape:{acc.shape}')
        print(f'x shape:{x.shape}')
        print(f'y shape:{y.shape}')
        print(f'slice_x:{x[:,1]}')
        for i in range(x.shape[0]-1):
            acc[i,:] = x[i+1,:vel_type] - x[i,:vel_type]
        acc[-1,:] = y - x[-1,:vel_type]
        print(f'new_x:{acc.shape}')
        # vel
        vel[:time_horizon-1,:] = x[1:,:vel_type]
        vel[time_horizon-1,:] = y
        print(f'vel:{vel}')

        # augment vel and acc
        new_x = cs.horzcat(vel,acc)
        print(f'x:{x[:,:vel_type]}')
        print(f'y:{y}')
        print(f'new_x:{new_x}')
        return new_x


    def forward(self, x):
        # TCN
        # print(f'block1 start')
        out = self.block1.cs_forward(x)
        # print(f'block2 start')
        out = self.block2.cs_forward(cs.transpose(out))
        # print(f'block3 start')
        out = self.block2.cs_forward(cs.transpose(out))
        # MLP
        out = self.input_layer(out)
        out = self.act(out)
        for layer in self.hidden_layers:
            out = layer(out)
            # print(f'try to use activation')
            out = self.act(out)
        y = self.output_layer(out)
        out = y[:,-1]
        print(f'tcn input :{x}')
        print(f'all tcn out shape:{out.shape}')
        # out = self.acceleration_vel(x,cs.transpose(out))
        return out


        


class Chomp1d_torch(nn.Module):
    def __init__(self, chompsize):
        super(Chomp1d_torch, self).__init__()
        self.chompsize = chompsize

    def forward(self, x):
        return x[:, :, :-self.chompsize].contiguous()




def gen_input(x, z, u):
    print(f'x:{x.shape}') # x,y,theta,x_dot,y_dot,theta_dot,
    print(f'z:{z.shape}')
    print(f'u:{u.shape}')
    # generate normal x and u for rk4
    x_pose_vel = cs.vertcat(cs.transpose(x),z[:,:6])
    
    # generate input for nn
    x_previous = z[:,3:6]
    x_all = cs.vertcat(cs.transpose(x[3:6]),x_previous)
    u_previous = z[:,6:]
    u_all = cs.vertcat(cs.transpose(u),u_previous)
    theta_all = cs.vertcat(cs.transpose(x[3]),z[:,3])
    input = cs.horzcat(x_all, u_all, theta_all)
    # update z
    u_x_cur = cs.horzcat(cs.transpose(x),cs.transpose(u))
    z_update = cs.vertcat(u_x_cur, z[:-1,:])

    print(f'x_pose_vel:{x_pose_vel.shape}')
    print(f'x_previouis:{x_previous.shape}')
    print(f'x_all:{x_all.shape}')
    print(f'u_all :{u_all.shape}')
    print(f'theta_all:{theta_all.shape}')
    print(f'input gen :{input.shape}')
    return x_pose_vel, u_all, input, z_update


# load model
ws_path = os.getcwd()
pt_name = 'ros_mlp.pt'
path = os.path.join(ws_path + '/../results/raigor_pi.pt')
print(f"path is {path}")
saved_dict = torch.load(path)
learn_model = TCN_withMLP(input_size=saved_dict['input_size'], output_size=saved_dict['output_size'], 
                            num_channels=saved_dict['num_channels'], kernel_size=saved_dict['kernel_size'], dropout=saved_dict['dropout'])
learn_model.load_state_dict(saved_dict['state_dict'])



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

# MX
v = cs.MX.sym('v',1,sequence_len)
omega = cs.MX.sym('omega',1,sequence_len)
u = cs.vertcat(v, omega)

x = cs.MX.sym('x',1,sequence_len) # state, can be position 
y = cs.MX.sym('y',1,sequence_len) # deriavtive of state, can be velocity
theta = cs.MX.sym('theta',1,sequence_len)
x_dot = cs.MX.sym('x_dot',1,sequence_len) # x linear vel
y_dot = cs.MX.sym('y_dot',1,sequence_len) # y linear vel
w_dot = cs.MX.sym('w_dot',1,sequence_len) # angular vel
# s_dot = cs.MX.sym('s_dot', 3, sequence_len)
previous_vel = cs.vertcat(x_dot, y_dot, w_dot)
previous_pos = cs.vertcat(x, y, theta)
input = cs.transpose(cs.vertcat(previous_vel, u, theta)) # len,channel
print(f'x_dot:{x_dot.shape}')
print(f'previous_vel:{previous_vel.shape}')
print(f'input:{input.shape}')
ctr_vel = cs.vertcat(v*cs.cos(theta), v*cs.sin(theta), omega)
# u_combine = cs.vertcat(v, omega, dt)
state = cs.vertcat(x, y, theta, x_dot, y_dot, w_dot)

# Calc
x_np = np.random.rand(1, num_inputs, sequence_len) # channel,len
# print(f'input:{x_np.shape}')
tcn_torch = TemporalBlock(num_inputs, num_outputs, kernel_size, stride=1, dilation=dilation,
                                     padding=(kernel_size-1) * dilation, dropout=dropout)
# TemporalBlock_numpy(num_inputs, num_outputs, kernel_size, stride, dilation, padding, dropout)

torch_input = torch.tensor(x_np, dtype=torch.float32)
tcn_torch_out = tcn_torch.forward(torch_input)
torch_conv = weight_norm(nn.Conv1d(num_inputs, num_outputs, kernel_size, 
                                           stride=stride, dilation=dilation, padding=padding))
print(f'torch.conv:{torch_conv(torch_input).shape}')
chomp1 = Chomp1d_torch(padding) 
# print(f'torch.chomp:{chomp1(torch_conv(torch_input)).shape}')\
print(f'start forwarding input')
# output_np = temporal_block_np.forward(input)
# print(output_np)
cs_tcn = Normalized_TCN(learn_model, num_inputs, num_channels, kernel_size, stride, dropout, saved_dict)
# print(f'{cs_tcn.forward(input)}')
# print("NumPy TemporalBlock Output:")


# it seems ok to me !?




print(f'discrete type!!!')

# pose init
pose_x = cs.MX.sym('pose_x')
pose_y = cs.MX.sym('pose_y')
pose_theta = cs.MX.sym('pose_theta')
pose = cs.vertcat(pose_x,pose_y,pose_theta)
# vel init
pose_x_dot = cs.MX.sym('pose_x_dot')
pose_y_dot = cs.MX.sym('pose_y_dot')
pose_theta_dot = cs.MX.sym('pose_theta_dot')
pose_dot = cs.vertcat(pose_x_dot,pose_y_dot,pose_theta_dot)
# init state x
x = cs.vertcat(pose, pose_dot) # include position and vel
# ctrl init
ctr_v = cs.MX.sym('ctr_v',1)
ctr_omega= cs.MX.sym('ctr_omega',1)
u = cs.vertcat(ctr_v, ctr_omega)
# z init
z = cs.MX.sym('z', 4, 8)




def acceleration_vel(x, y):
    # new_x = cs.MX.zeros((2*x.shape[0],x.shape[1])) # x 5*6
    # acc = cs.MX.zeros((x.shape[0],vel_type))
    # vel = cs.MX.zeros((x.shape[0],vel_type))

    time_horizon = 5
    vel_type = 3 
    new_x = np.zeros((2*x.shape[0],x.shape[1]))
    acc = np.zeros((x.shape[0],vel_type))
    vel = np.zeros((x.shape[0],vel_type))
    print(f'new_x shape:{acc.shape}')
    print(f'x shape:{x.shape}')
    print(f'y shape:{y.shape}')
    print(f'slice_x:{x[:,1]}')
    for i in range(x.shape[0]-1):
        acc[i,:] = x[i+1,:vel_type] - x[i,:vel_type]
    acc[-1,:] = y - x[-1,:vel_type]
    print(f'new_x:{acc.shape}')
    # vel
    vel[:time_horizon-1] = x[1:,:vel_type]
    vel[time_horizon-1] = y
    print(f'vel:{vel}')

    # augment vel and acc
    new_x = np.hstack([vel,acc])
    print(f'x:{x[:,:vel_type]}')
    print(f'y:{y}')
    print(f'new_x:{new_x}')
    return acc

def rk_permute(x):
    # x : [vel,acc]
    x_rk_frompermute = x[2:,:]
    print(f'x_rk_frompermute:{x_rk_frompermute.shape}')
    return x_rk_frompermute

array_5x6 = np.array([[1, 2, 3, 4, 5, 6],
                     [7, 8, 9, 10, 11, 12],
                     [13, 14, 15, 16, 17, 18],
                     [19, 20, 21, 22, 23, 24],
                     [25, 26, 27, 28, 29, 30]])

# 初始化一个3x1的数组，元素为确定性的数字（示例中使用31到33的整数）
array_3x1 = np.array([[1],
                      [2],
                      [3]])
print(f'test acceleration:{acceleration_vel(array_5x6,np.transpose(array_3x1))}')

# net input generation
x_pose_vel, u_rk4, input_fromgen, z_update = gen_input(x, z, u)
expl = cs_tcn.forward(input_fromgen)
# ode = cs.Function('ode', [input_fromgen], [expl])
# print(f'function ode:{ode(input_fromgen)}')
print(f'output_fromgen:{expl.shape}')
print(f'z_update:{z_update.shape}')
print(f'permuting for rk now!!!:{rk_permute(cs.transpose(expl)).shape}')





# net new input generation
def new_gen_input(x, u):
    print(f'new_gen_input!!!!')
    print(f'x:{x.shape}') # x,y,theta,x_dot,y_dot,theta_dot,
    print(f'u:{u.shape}')
 
    # generate input for nn
    x_all = cs.vertcat(cs.transpose(x[3:6,:]))
    u_previous = x[6:,:-1] # get the first 4 value from u, cause the fifth is nonsense
    u_all = cs.transpose(cs.horzcat(u_previous,u))
    theta_all = cs.vertcat(cs.transpose(x[2,:]))
    input = cs.horzcat(x_all, u_all, theta_all)

    # generate for RK4
    x_rk = cs.vertcat(cs.transpose(x_all),cs.transpose(theta_all))
    u_rk = u_all

    # generate u for integrate into new x
    u_for_integrate = u_all[1:,:] 
    u_for_integrate = cs.vertcat(u_for_integrate,u_all[0,:]) # add nonsense u_all[0,:]

    print(f'x_all:{x_all.shape}')
    print(f'u_previous:{u_previous.shape}')
    print(f'u_all :{u_all.shape}')
    print(f'theta_all:{theta_all.shape}')
    print(f'input gen :{input.shape}') # 5*6
    print(f'x_rk:{x_rk.shape}') # 4*5 [vel,theta]
    print(f'u_rk:{u_rk.shape}') # 5*2
    print(f'u_for_integrate:{u_for_integrate.shape}')
    return input,x_rk,u_rk

def gen_rk_input(x,u):
    print(f'gen rk input!!!!!!!!')
    x_all_fromrk = cs.transpose(x[:3,:]) # 5*3
    theta_all_fromrk  = cs.transpose(x[3,:])
    u__all_fromrk = u
    rk_input = cs.horzcat(x_all_fromrk,u__all_fromrk,theta_all_fromrk)

    print(f'x_all:{x_all_fromrk .shape}')
    print(f'u_all :{u__all_fromrk .shape}')
    print(f'theta_all:{theta_all_fromrk .shape}')
    print(f'input gen :{rk_input.shape}') # 5*6
    return rk_input

new_pose = cs.MX.sym('new_pose', 3, sequence_len)
new_vel = cs.MX.sym('new_vel', 3, sequence_len)
new_u_pst = cs.MX.sym('u_pst', 2, sequence_len)
new_x = cs.vertcat(new_pose,new_vel,new_u_pst)
print(f'newx reshape:{new_x.reshape((8*sequence_len,1)).shape}')
new_input,x_rk,u_rk = new_gen_input(new_x,u)
whatever = gen_rk_input(x_rk,u_rk)


def create_x4input(x):
    sequence_len = 5
    print(f'create x4input!!!!!!!')
    pose = x[:3*sequence_len].reshape((3,sequence_len))
    vel = x[3*sequence_len:3*sequence_len+3*sequence_len].reshape((3,sequence_len))
    u_pst = x[3*sequence_len+3*sequence_len:].reshape((2,sequence_len))
    print(f'pose:{pose.shape}')
    print(f'vel:{vel.shape}')
    print(f'u_pst:{u_pst.shape}')
    return 0

new_pose1 = cs.MX.sym('new_pose', 3*sequence_len)
new_vel1 = cs.MX.sym('new_vel', 3*sequence_len)
new_u_pst1 = cs.MX.sym('u_pst', 2*sequence_len)
new_x1 = cs.vertcat(new_pose1,new_vel1,new_u_pst1)
print(f'new_x1:{new_x1.shape}')
x4input = create_x4input(new_x1)




