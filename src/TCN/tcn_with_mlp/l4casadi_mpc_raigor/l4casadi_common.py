import torch
import numpy as np
import l4casadi as l4c
import casadi as cs
import sys
sys.path.append("/home/ryan/raigor/train_ws/src/")
import casadi as cs
import numpy as np
import torch
import torchvision.models as models
import os
# from draw import Draw_MPC_point_stabilization_v1
from model import TCN_withMLP, TCN, NormalizedTCN
import rospy
from turtlesim.msg import Pose  # 导入 Pose 消息类型
from geometry_msgs.msg import Twist

# load model
ws_path = os.getcwd()
pt_name = 'ros_mlp.pt'
path = os.path.join(ws_path + '/../results/raigor_pi.pt')
print(f"path is {path}")
saved_dict = torch.load(path)
learn_model = TCN_withMLP(input_size=saved_dict['input_size'], output_size=saved_dict['output_size'], 
                            num_channels=saved_dict['num_channels'], kernel_size=saved_dict['kernel_size'], dropout=saved_dict['dropout'])
learn_model.load_state_dict(saved_dict['state_dict'])


# x = np.random.randn(49).astype(np.float32)
model = learn_model
# y = model(torch.tensor(new_input)[None])
# print(f'Torch output: {y}')

l4c_model = l4c.L4CasADi(model, model_expects_batch_dim=False)
# print(f'input:{input.shape}')
# y_sym = l4c_model(input)
# print(f'L4CasADi Output: {y_sym}')

# Create a model with convolutional layers
class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = x.reshape(-1, 1, 7, 7)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# net new input generation
def gen_input(x, u):
    print(f'new_gen_input!!!!')
    print(f'x:{x.shape}') # x,y,theta,x_dot,y_dot,theta_dot,
    print(f'u:{u.shape}')

    # generate input for nn
    x_all = np.vstack(np.transpose(x[3:6,:]))
    u_previous = x[6:,:-1]
    u_all = np.transpose(np.hstack((u_previous,u)))
    theta_all = np.vstack(np.transpose(x[2,:]))
    input = np.hstack((x_all, u_all, theta_all))

    # generate for RK4
    x_rk = np.vstack((np.transpose(x_all),np.transpose(theta_all))) # vel,theta 4*5
    u_rk = u_all
    # generate u for integrate into new x
    u_for_integrate = u_all[1:,:] 
    u_for_integrate = np.transpose(np.vstack((u_for_integrate,u_all[0,:]))) # add nonsense u_all[0,:]
    
    print(f'x_all:{x_all.shape}')
    print(f'u_all :{u_all.shape}')
    print(f'theta_all:{theta_all.shape}')
    print(f'input gen :{input.shape}') # 5*6
    print(f'x_rk:{x_rk.shape}') # 4*5 vel,theta
    return input,x_rk,u_rk,u_for_integrate

def gen_rk_input(x,u):
    print(f'gen rk input!!!!!!!!')
    print(f'gen rk input:{x.shape}')
    x_all_fromrk = np.transpose(x[:3,:]) # 5*3
    theta_all_fromrk  = np.transpose(x[3,:]).reshape(5,-1)
    u__all_fromrk = u
    print(f'shape:{x_all_fromrk.shape,u__all_fromrk.shape,theta_all_fromrk.shape}')
    rk_input = np.hstack((x_all_fromrk,u__all_fromrk,theta_all_fromrk))

    print(f'x_all:{x_all_fromrk .shape}')
    print(f'u_all :{u__all_fromrk .shape}')
    print(f'theta_all:{theta_all_fromrk .shape}')
    print(f'input gen :{rk_input.shape}') # 5*6 [vel,u,theta]
    return rk_input

def rk_permute(x):
    # x : [vel,acc] 5*6
    # delete old vel and add theta in
    x = cs.transpose(x)
    x_rk_vel = x[-3:,:] # 3*5 actual acc for vel
    x_rk_theta = x[2,:] # 1*5 actual w for theta
    x_rk_frompermute = cs.vertcat(x_rk_vel,x_rk_theta)
    print(f'x_rk_frompermute:{x_rk_frompermute.shape}')
    return x_rk_frompermute


def create_x4input(x):
    sequence_len = 5
    print(f'create x4input!!!!!!!')
    pose = x[:3*sequence_len].reshape((3,sequence_len))
    vel = x[3*sequence_len:3*sequence_len+3*sequence_len].reshape((3,sequence_len))
    u_pst = x[3*sequence_len+3*sequence_len:].reshape((2,sequence_len))
    print(f'pose:{pose.shape}')
    print(f'vel:{vel.shape}')
    print(f'u_pst:{u_pst.shape}')
    new_x = np.vstack((pose,vel,u_pst))
    newx_no_u = np.vstack((pose,vel))
    return new_x,newx_no_u

sequence_len = 5

# # 定义三维的 CasADi 变量
# x = cs.MX.sym('x', 3, 4, 2)  # 三维变量，形状为 (3, 4, 2)

# # 打印变量的形状
# print("Shape of x:", x.shape)

# pose init
pose = cs.MX.sym('new_pose', 3*sequence_len) # 3*5
# vel init
pose_dot = cs.MX.sym('new_vel', 3*sequence_len) # 3*5
# past u init
u_pst = cs.MX.sym('u_pst', 2*sequence_len) # 2*5 one nonsense variable
# state x init
x = cs.vertcat(pose,pose_dot,u_pst)

# acc init
x_acc = cs.MX.sym('x_acc')
y_acc = cs.MX.sym('y_acc')
theta_acc = cs.MX.sym('theta_acc')
acc = cs.vertcat(x_acc,y_acc,theta_acc)
# init state x
xdot = cs.vertcat(pose_dot, acc) # include position and vel
# ctrl init
ctr_v = cs.MX.sym('ctr_v',1)
ctr_omega= cs.MX.sym('ctr_omega',1)
# u = cs.vertcat(ctr_v, ctr_omega)



data = """-0.086893975226788 0.000543302599104 0.01371263412115 0.335816002016689 0.010920714489099 0.106576585282543 0.6 0.2
0.026129363985015 0.005126101958987 0.033083066465925 0.664412777529619 0.020956249355815 0.181341725096883 0.6 0.2
0.155373345108982 0.012716666128385 0.057292978194316 0.699820755077229 0.041295144872296 0.112634651766008 0.6 0.2
0.299447939536588 0.023423730984981 0.076999218682932 0.696839677180168 0.049609039782608 0.122569128851088 0.6 0.2
0.436021657138206 0.036567799642723 0.099793699296486 0.675622452496484 0.065449401803523 0.152111546326436 0.6 0.2"""

data2 = """0.335816002016689	0.010920714489099	0.106576585282543	0.6	0.2	0.01371263412115
0.664412777529619	0.020956249355815	0.181341725096883	0.6	0.2	0.033083066465925
0.699820755077229	0.041295144872296	0.112634651766008	0.6	0.2	0.057292978194316
0.696839677180168	0.049609039782608	0.122569128851088	0.6	0.2	0.076999218682932
0.675622452496484	0.065449401803523	0.152111546326436	0.6	0.2	0.099793699296486"""

data1 = """0.335816002016689	0.010920714489099	0.106576585282543	0.6	0.2	0.01371263412115
0.664412777529619	0.020956249355815	0.181341725096883	0.6	0.2	0.033083066465925
0.699820755077229	0.041295144872296	0.112634651766008	0.6	0.2	0.057292978194316
0.696839677180168	0.049609039782608	0.122569128851088	0.6	0.2	0.076999218682932
0.675622452496484	0.065449401803523	0.152111546326436	0.6	0.2	0.099793699296486""" # vel ctrl theta 5*6

# 将文本数据转换为NumPy数组
array_data = np.array([list(map(float, line.split())) for line in data.split('\n')])
array_data1 = np.array([list(map(float, line.split())) for line in data1.split('\n')])
array_data2 = np.array([list(map(float, line.split())) for line in data2.split('\n')])
y_sy1 = l4c_model(array_data2)
print(f'L4CasADi Output: {y_sy1}')

u = np.array([[0.7,0.2]]).T
y_sy2 = np.array(model(torch.tensor(array_data2, dtype=torch.float32)))
print(f'L4CasADi Output: {y_sy2}')



# net input generation
x = (array_data.T).reshape(40,1)
print(f'x:{x.shape}')
x4_input,newx_no_u = create_x4input(x)
input,x_rk,u_rk,u_for_integrate = gen_input(x4_input, u)
print(f'--------------------------here is acados---------------------------------')
print(f'x_rk_1st:{x_rk}')
print(f'u_rk_1st:{u_rk}')
learned_dyn = l4c_model
# set up RK4
dT = 0.1
k1 = learned_dyn(gen_rk_input(x_rk,u_rk))
print(f'k1:{k1.shape}') # 5*6 vel,acc
print(f'k1_1st:{k1}') # 5*6 vel,acc
k1_new = rk_permute(k1)
print(f'k1_1st_after:{k1_new}')
k2 = learned_dyn(gen_rk_input(x_rk+dT/2*k1_new,u_rk))  # x_rk:[vel,theta]
print(f'k2_1st:{k2}') # 5*6 vel,acc
k2_new = rk_permute(k2)
print(f'k2_1st_after:{k2_new}')
k3 = learned_dyn(gen_rk_input(x_rk+dT/2*k2_new,u_rk))
print(f'k3_1st:{k3}') # 5*6 vel,acc
k3_new = rk_permute(k3)
print(f'k3_1st_after:{k3_new}')
k4 = learned_dyn(gen_rk_input(x_rk+dT*k3_new,  u_rk))
print(f'k4_1st:{k4}') # 5*6 vel,acc
print(f'newx_no_u:{newx_no_u}')
print(f'u_for_integrate:{u_for_integrate}')
xf = cs.transpose(newx_no_u) + dT/6 * (k1 + 2*k2 + 2*k3 + k4) # [pose,vel]
print(f'xf:{xf}')
xf_u = cs.transpose(cs.vertcat(cs.transpose(xf),u_for_integrate)).reshape((8*sequence_len,1))      
print(f'xf_u:{xf_u}')







