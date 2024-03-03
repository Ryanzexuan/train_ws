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
import l4casadi as l4c


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
parser.add_argument('--select', type=int, default=1234)

args = parser.parse_args()
print(args)
lambda_v = args.lambda_value
print(lambda_v)
selection = args.select

cur_path = os.getcwd()
print(f'cur_path:{cur_path}')
# uneven_manual, data, dataset_gzsim_nominal, tele_random, uneven, uneven.mpc 
data_path = os.path.join(cur_path + '/../data/test/data.csv') # 5198
pt_path_withoutYaw = os.path.join(cur_path + '/../results/tcn.pt')
# pt_path_withPI = os.path.join(cur_path + '/../results/raigor_pi.pt')
pt_path_withPI = os.path.join(cur_path + '/../results/sim/PItcn_withyaw.pt')
pt_path_withoutPI = os.path.join(cur_path + '/../results/tcn_withyaw.pt')
pt_path_with_dt = os.path.join(cur_path + '/../results/tcn_withyaw_dt1.pt')
print(f'data_path:{data_path}')
if torch.cuda.is_available():
        print(f'cuda is available')
# print(f'pt_path:{pt_path}')
### ctrl + yaw + PI loss : 111 means has all
# saved_dict_100 = torch.load(pt_path_withoutYaw)
# saved_dict_110 = torch.load(pt_path_withoutPI)
saved_dict_111 = torch.load(pt_path_withPI)
# saved_dict_000 = torch.load(pt_path_with_dt)

class data_character():
    def __init__(self, sequence_len, input_size, output_size):
        super(data_character, self).__init__()
        print(f'inputsize:{input_size}')
        self.row_len = sequence_len
        self.col_len = input_size
        self.out_len = output_size

def divide_params(params, data):
    row_len = data.row_len
    col_len = data.col_len
    out_len = data.out_len
    divide1 = col_len
    divide2 = col_len + out_len
    print(f'divide1:{divide1}')
    print(f'divide2:{divide2}')
    a = params[:, :col_len*row_len].reshape(row_len, col_len)
    f_a = params[:, col_len*row_len : col_len*row_len + out_len]
    df_a = params[:, col_len*row_len + out_len:].reshape(out_len, row_len, col_len)
    print(f'a:{a}')
    print(f'fa:{f_a}')
    print(f'dfa:{df_a}')
    return a, f_a, df_a

def linear_calc(a, f_a, df_a, x):
    delta = x - a
    original_f_delta = df_a * delta
    f_delta = np.sum(df_a * delta, axis=(1, 2))
    f = f_a + f_delta
    # print(f'original f_delta:{df_a * delta}')
    # print(f'f_delta:{f_delta}')
    return f


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


def calc_diff_time(t, start_dice):
    data = t
    if np.ndim(start_dice) == 2:
        start_dice = start_dice.ravel()
    if np.size(start_dice) == 0 or (np.size(start_dice) > 0 and start_dice[0] != 0):
        start_indices_noend = np.insert(start_dice, 0, 0)
        flag = True  
    else: 
        start_indices_noend = start_dice
    start_indices = np.insert(start_indices_noend, len(start_indices_noend), data.shape[0]-1)
    raw_dt = t#
    dt = np.diff(t)
    print(f't:{dt}')
    raw_dt[:-1] = dt
    raw_dt[-1] = 0
    print(f'start_indices:{start_indices}')
    for i in start_indices.squeeze():
        # print(f'i{i}')
        raw_dt[i] = 0
        if i-1 >= 0:
            raw_dt[i-1] = 0
    return raw_dt

# net new input generation
def gen_input(x, u):
    # print(f'new_gen_input!!!!')
    # print(f'x:{x.shape}') # x,y,theta,x_dot,y_dot,theta_dot,
    # print(f'u:{u.shape}')

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
    
    # print(f'x_all:{x_all.shape}')
    # print(f'u_all :{u_all.shape}')
    # print(f'theta_all:{theta_all.shape}')
    # print(f'input gen :{input.shape}') # 5*6
    # print(f'x_rk:{x_rk.shape}') # 4*5 vel,theta
    return input,x_rk,u_rk,u_for_integrate

def gen_xrk_urk_fromtorch(x):
    # x is 50*5*8 x,y,vel*3,u*2,theta
    x_rk = x[:,[2,3,4,-1]] # vel,theta
    u_rk = x[:,[5,6]]
    x_rk = x_rk.T
    # print(f'x_rk:{x_rk.shape}')
    # generate u for integrate into new x
    u_all = u_rk
    u_for_integrate = u_all[1:,:] 
    u_for_integrate = np.transpose(np.vstack((u_for_integrate,u_all[0,:]))) # add nonsense u_all[0,:]

    newx_no_u = x[:,[0,1,-1,2,3,4]]

    return x_rk,u_rk,u_for_integrate,newx_no_u


def gen_rk_input(x,u):
    # print(f'gen rk input!!!!!!!!')
    # print(f'gen rk input:{x.shape}')
    # x = x.T
    x_all_fromrk = np.transpose(x[:3,:]) # 5*3
    theta_all_fromrk  = np.transpose(x[3,:]).reshape(5,-1)
    u__all_fromrk = u
    # print(f'shape:{x_all_fromrk.shape,u__all_fromrk.shape,theta_all_fromrk.shape}')
    rk_input = np.hstack((x_all_fromrk,u__all_fromrk,theta_all_fromrk))

    # print(f'x_all:{x_all_fromrk .shape}')
    # print(f'u_all :{u__all_fromrk .shape}')
    # print(f'theta_all:{theta_all_fromrk .shape}')
    # print(f'input gen :{rk_input.shape}') # 5*6 [vel,u,theta]
    return rk_input

def rk_permute(x):
    # x : [vel,acc] 5*6
    # delete old vel and add theta in
    x = x.T
    x_rk_vel = x[-3:,:] # 3*5 actual acc for vel
    x_rk_theta = x[2,:] # 1*5 actual w for theta
    x_rk_frompermute = np.vstack((x_rk_vel,x_rk_theta))
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

def calc_pose(x,l4c_model):
    sequence_len = 5
    x_rk,u_rk,u_for_integrate,newx_no_u = gen_xrk_urk_fromtorch(x)
    # print(f'--------------------------here is acados---------------------------------')
    # print(f'x_rk_1st:{x_rk}')
    # print(f'u_rk_1st:{u_rk}')
    learned_dyn = l4c_model
    # set up RK4
    dT = 0.1
    k1 = learned_dyn(gen_rk_input(x_rk,u_rk))
    # print(f'k1:{k1.shape}') # 5*6 vel,acc
    # print(f'k1_1st:{k1}') # 5*6 vel,acc
    k1_new = rk_permute(k1)
    # print(f'k1_1st_after:{k1_new}')
    k2 = learned_dyn(gen_rk_input(x_rk+dT/2*k1_new,u_rk))  # x_rk:[vel,theta]
    # print(f'k2_1st:{k2}') # 5*6 vel,acc
    k2_new = rk_permute(k2)
    # print(f'k2_1st_after:{k2_new}')
    k3 = learned_dyn(gen_rk_input(x_rk+dT/2*k2_new,u_rk))
    # print(f'k3_1st:{k3}') # 5*6 vel,acc
    k3_new = rk_permute(k3)
    # print(f'k3_1st_after:{k3_new}')
    k4 = learned_dyn(gen_rk_input(x_rk+dT*k3_new,  u_rk))
    # print(f'k4_1st:{k4}') # 5*6 vel,acc
    # print(f'newx_no_u:{newx_no_u}')
    # print(f'u_for_integrate:{u_for_integrate}')
    xf = newx_no_u + dT/6 * (k1 + 2*k2 + 2*k3 + k4) # [pose,vel]
    # print(f'xf:{xf.shape}')
    # xf_u = np.transpose(np.vstack((np.transpose(xf),u_for_integrate))).reshape((8*sequence_len,1))      
    # print(f'xf_u:{xf_u}')
    return xf


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
        raw_x = np.array(raw_data['x_position_input'])
        raw_y = np.array(raw_data['y_position_input'])
        raw_yaw = np.array(raw_data['yaw_input'])
        # state dot
        raw_x_dot = np.array(raw_data['vel_x_input'])
        raw_y_dot = np.array(raw_data['vel_y_input'])
        raw_yaw_dot = np.array(raw_data['vel_w_input']) # vel_w too small
        # print(f'raw x dot{raw_x_dot}')
        # dt
        raw_con_t = np.array(raw_data['con_time'])
        raw_dt = calc_diff_time(raw_con_t, idx)
        # control 
        raw_u_v = np.array(raw_data['con_x_input'])
        raw_u_w = np.array(raw_data['con_z_input'])
        # print(f'raw x dot{raw_u_v}')

        # print(f'raw x:{raw_y}')
        # time_sequence = np.column_stack((raw_x, raw_y, raw_yaw))
        # time_sequence = np.column_stack((raw_x_dot[:1000], raw_y[:1000], raw_yaw[:1000]))
        # time_sequence = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w))# with control
        time_sequence_withyaw = np.column_stack((raw_x,raw_y,raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w, raw_yaw))# with control and yaw
        # time_sequence_noyaw = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w))
        # time_sequence_with_dt = np.column_stack((raw_x_dot, raw_y_dot, raw_yaw_dot, raw_u_v, raw_u_w, raw_yaw, raw_dt))
        print(f'time sequence :{time_sequence_withyaw}')
        # data_length = 1000 # whole data length
        sequence_length = 5 # net input length

        data_traj_111 = TrajectoryDataset(time_sequence_withyaw, sequence_length, idx, saved_dict_111['output_size'])
        data_loader_111 = DataLoader(data_traj_111, batch_size=128, shuffle=True)


        ## model 111 read
        model_111 = TCN_withMLP(input_size=saved_dict_111['input_size'], output_size=saved_dict_111['output_size'], 
                            num_channels=saved_dict_111['num_channels'], kernel_size=saved_dict_111['kernel_size'], dropout=saved_dict_111['dropout'])
        
        model_111.load_state_dict(saved_dict_111['state_dict'])
        
        # Set evaluate mode
        model_111.eval()  
        model = model_111
        l4c_model = l4c.L4CasADi(model, model_expects_batch_dim=False)
        # ## model 000 read
        # model_000 = TCN_withMLP(input_size=saved_dict_000['input_size'], output_size=saved_dict_000['output_size'], 
        #                     num_channels=saved_dict_000['num_channels'], kernel_size=saved_dict_000['kernel_size'], dropout=saved_dict_000['dropout'])
        
        # model_000.load_state_dict(saved_dict_000['state_dict'])
        # # Set evaluate mode
        # model_000.eval()  

        # N = 100

        criterion = nn.MSELoss()
        # # getting slice
        test_num = 50
        start = random.randint(0, time_sequence_withyaw.shape[0]-test_num-1)
        # test_100 = []
        # test_110 = []
        test_111 = []
        # test_000 = []

        # losses_100 = []
        # losses_110 = []
        losses_111 = []
        # losses_000 = []

        # # print(f"actual_data:{actual_data}")

        real_data = []
        for i in range(test_num):
            print(f'start:{start}')
            test_input_100,real_out = data_traj_111[i + start]
            test_input_111,_ = data_traj_111[i + start]
            print(f'test_input_111 single shape:{test_input_111.shape}')
            real_data.append(real_out)
            test_111.append(test_input_111)
        test_111 = torch.tensor(np.array(test_111))
        real_data = np.array(real_data).squeeze()
        print(f"test_111 shape:{test_111.shape}")
        print(f"real_data[start]:{real_data[0]}")
        test_111_np = np.array(test_111)
        
        predicted_sequence_111 = []
        print(test_111_np[0])
        x_0 = test_111_np[0,:,[0,1,-1,2,3,4]]
        input = x_0.T # 5*6
        print(f'input:{input}')
        with torch.no_grad():
            # 假设你有一个输入序列 input_sequence_tensor
            for i,item in enumerate(test_111_np):
                print(f'item:{item.shape}')
                print(f'input:{input.shape}')
                pose_vel = input[:,[0,1,3,4,5]]
                theta = input[:,2].reshape((5,1))
                input = np.hstack((pose_vel,item[:,[5,6]],theta))
                next_input = calc_pose(input,l4c_model)
                predicted_sequence_111.append(next_input)
                input = next_input
                # predicted_sequence_111 = model_111(item).squeeze()
            
            vx,vy,w = nominal_calc(test_111)
            nominal_loss = nominal_loss_calc(vx, vy, w, real_data)


        # for i, item in enumerate(predicted_sequence_111):
        #     loss = criterion(torch.tensor(item), torch.tensor(real_data[i, :args.n_out]))
        #     losses_111.append(torch.sqrt(loss))

        # losses_111_mean = np.mean(losses_111)
        
        nominal_mean = np.mean(nominal_loss)

        # print(f'mean loss for model c+yaw+PI:{losses_111_mean}')
        #print(f'mean loss for model c+yaw+PI+time:{losses_000_mean}')
        print(f'mean loss for model Nominal:{nominal_mean}')


        # 绘制实际数据和预测数据
        # predicted_sequence_100 = predicted_sequence_100.numpy()  # 预测数据
        # predicted_sequence_110 = predicted_sequence_110.numpy()  # 预测数据
        # predicted_sequence_111 = predicted_sequence_111.numpy()  # 预测数据
        #predicted_sequence_000 = predicted_sequence_000.numpy()
        real_data = real_data[:,[0,1,-1]]
        predicted_sequence_111 = np.array(predicted_sequence_111)
        predicted_sequence_111 = predicted_sequence_111[:,-1,:]
        print(f'size of actual data:{real_data.shape}')
        print(f'size of predict data:{predicted_sequence_111.shape}')
        print(f'size of predict data:{predicted_sequence_111}')
        #print(f'size of predict data:{predicted_sequence_100.shape[0]}')
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)  # 第一个子图用于损失曲线
        # plt.plot(losses_100, label="ctrl loss", marker='o', color='green')
        # plt.plot(losses_110,label="c+yaw  loss", marker='x', color='orange')
        plt.plot(losses_111,label="c+yaw+PI loss", marker='*', color='purple')
        #plt.plot(losses_000,label="c+yaw+PI+time loss", marker='*', color='black')
        plt.plot(nominal_loss,label="Nominal loss", marker='^', color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("testing Loss")
        plt.legend()
        # 第二个子图用于实际和预测数据
        plt.subplot(2, 2, 2)  
        plt.plot(real_data[:, 0], label="Actual x", marker='o', color='blue')
        # if '1' in str(selection):
        #     plt.plot(predicted_sequence_100[:, 0], linestyle='--', label="ctrl Predicted x", marker='x',color='green')
        # if '2' in str(selection):
        #     plt.plot(predicted_sequence_110[:, 0], linestyle='--', label="c+yaw Predicted x", marker='x',color='orange')
        if '3' in str(selection):
            plt.plot(predicted_sequence_111[:, 0], linestyle='--', label="c+yaw+PI Predicted x", marker='x',color='purple')
        #if '4' in str(selection):
            # plt.plot(predicted_sequence_000[:, 1], linestyle='--', label="c+yaw+PI+time Predicted x", marker='x',color='black')
        plt.plot(vx.numpy(), linestyle='--', label="Nominal x", marker='*',color='red')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Actual vs. Predicted Data (x)")
        plt.legend()

        plt.subplot(2, 2, 3) 
        plt.plot(real_data[:, 1], label="Actual y", marker='o',color='blue')
        # if '1' in str(selection):
        #     plt.plot(predicted_sequence_100[:, 1], linestyle='--', label="ctrl Predicted y", marker='x',color='green')
        # if '2' in str(selection):
        #     plt.plot(predicted_sequence_110[:, 1], linestyle='--', label="c+yaw Predicted y", marker='x',color='orange')
        # 
        if '3' in str(selection):
            plt.plot(predicted_sequence_111[:, 1], linestyle='--', label="c+yaw+PI Predicted y", marker='x',color='purple')
        # if '4' in str(selection):
        #     plt.plot(predicted_sequence_000[:, 1], linestyle='--', label="c+yaw+PI+time Predicted y", marker='x',color='black')
        # 
        plt.plot(vy.numpy(), linestyle='--', label="Nominal y", marker='*',color='red')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Actual vs. Predicted Data (y)")
        plt.legend()


        plt.subplot(2, 2, 4) 
        plt.plot(real_data[:, 2], label="Actual z", marker='o', color='blue')
        # if '1' in str(selection):
        #     plt.plot(predicted_sequence_100[:, 2], linestyle='--', label="ctrl Predicted z", marker='x', color='green')
        # if '2' in str(selection):
        #     plt.plot(predicted_sequence_110[:, 2], linestyle='--', label="c+yaw Predicted z", marker='x', color='orange')
        # 
        if '3' in str(selection):
            plt.plot(predicted_sequence_111[:, 2], linestyle='--', label="c+yaw+PI Predicted z", marker='x', color='purple')
        # if '4' in str(selection):
        #     plt.plot(predicted_sequence_000[:, 2], linestyle='--', label="c+yaw+PI+time Predicted z", marker='x',color='black')
        # 
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