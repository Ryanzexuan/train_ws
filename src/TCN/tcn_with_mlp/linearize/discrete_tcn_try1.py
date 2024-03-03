import sys
sys.path.append("/home/ryan/raigor/train_ws/src/")
import rospy
import casadi as cs
from casadi_common import Normalized_TCN
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import ml_casadi.torch as mc
from tqdm import tqdm
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
import time
import os
import math
import subprocess
import tf.transformations as tf
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy.linalg
# from draw import Draw_MPC_point_stabilization_v1
from model import TCN_withMLP, TCN, NormalizedTCN
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MODEL_ARCH = 'MLP'  # On of 'MLP' or 'CNNRESNET'
# MLP: Dense Network of 32 layers 256 neurons each
# CNNRESNET: ResNet model with 18 Convolutional layers

USE_GPU = False
cur_rec_state_set = np.zeros(7)
cur_cmd = np.zeros(2)

class CNNResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net = models.resnet18()

        self.linear = torch.nn.Linear(1000, 2, bias=False)

        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.linear.weight.fill_(0.)

    def forward(self, x):
        # Tile input such that it fits the expected ResNet Input
        x = x[:, None, None, :].repeat(1, 3, 64, 32)
        return self.linear(self.res_net(x))


class MLP(mc.nn.MultiLayerPerceptron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.output_layer.bias.fill_(0.)
            self.output_layer.weight.fill_(0.)


class DoubleIntegratorWithLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn
        self.Bx = np.array([[0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1]])
        self.sequence_len = 5

    def function(self, x):
        print(f'function found')
        return 0

    def acceleration(self, x, y):
        new_x = cs.MX.zeros(x.shape)
        print(f'new_x shape:{new_x.shape}')
        print(f'x shape:{x.shape}')
        print(f'y shape:{y.shape}')
        print(f'slice_x:{x[:,1]}')
        for i in range(new_x.shape[1] - 1):
            new_x[i,:] = x[i+1,:] - x[i,:]
        new_x[-1,:] = y - x[-1,:]
        print(f'new_x:{new_x.shape}')
        return new_x

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
    def model(self):
        s_dot_dim = self.learned_dyn.output_size
        x_dim = self.learned_dyn.input_size
        # pose init
        pose_x = cs.MX.sym('pose_x')
        pose_y = cs.MX.sym('pose_y')
        pose_theta = cs.MX.sym('pose_theta')
        x = cs.vertcat(pose_x,pose_y,pose_theta)
        # vel init
        pose_x_dot = cs.MX.sym('pose_x_dot')
        pose_y_dot = cs.MX.sym('pose_y_dot')
        pose_theta_dot = cs.MX.sym('pose_theta_dot')
        pose_dot = cs.vertcat(pose_x_dot,pose_y_dot,pose_theta_dot)
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
        u = cs.vertcat(ctr_v, ctr_omega)
        # z init
        z = cs.MX.sym('z', 4, 8)

        # net input generation
        x_rk4, u_rk4, input, z_update = self.gen_input(x, z, u)

        
        u_past = cs.MX.sym('u_past',1,self.sequence_len-1)
        # x = cs.MX.sym('x',1,self.sequence_len) # state, can be position 
        y = cs.MX.sym('y',1,self.sequence_len) # deriavtive of state, can be velocity
        theta = cs.MX.sym('theta',1,self.sequence_len)
        # k moment
        x_dot_k = cs.MX.sym('x_dot_k',1) # x linear vel
        y_dot_k = cs.MX.sym('y_dot_k',1) # y linear vel
        w_dot_k = cs.MX.sym('w_dot_k',1) # angular vel
        # 
        x_dot_k_minus_1 = cs.MX.sym('x_dot_k_minus_1',1) # x linear vel
        y_dot_k_minus_1 = cs.MX.sym('y_dot_k_minus_1',1) # y linear vel
        w_dot_k_minus_1 = cs.MX.sym('w_dot_k_minus_1',1) # angular vel
        # 
        x_dot_k_minus_2 = cs.MX.sym('x_dot_k_minus_2',1) # x linear vel
        y_dot_k_minus_2 = cs.MX.sym('y_dot_k_minus_2',1) # y linear vel
        w_dot_k_minus_2 = cs.MX.sym('w_dot_k_minus_2',1) # angular vel
        # 
        x_dot_k_minus_3 = cs.MX.sym('x_dot_k_minus_3',1) # x linear vel
        y_dot_k_minus_3 = cs.MX.sym('y_dot_k_minus_3',1) # y linear vel
        w_dot_k_minus_3 = cs.MX.sym('w_dot_k_minus_3',1) # angular vel
        # 
        x_dot_k_minus_4 = cs.MX.sym('x_dot_k_minus_4',1) # x linear vel
        y_dot_k_minus_4 = cs.MX.sym('y_dot_k_minus_4',1) # y linear vel
        w_dot_k_minus_4 = cs.MX.sym('w_dot_k_minus_4',1) # angular vel

        
        # s_dot = cs.MX.sym('s_dot', 3, self.sequence_len)
        vel_k = cs.vertcat(x_dot_k, y_dot_k, w_dot_k)
        vel_k_minus_1 = cs.vertcat(x_dot_k_minus_1, y_dot_k_minus_1, w_dot_k_minus_1)
        vel_k_minus_2 = cs.vertcat(x_dot_k_minus_2, y_dot_k_minus_2, w_dot_k_minus_2)
        vel_k_minus_3 = cs.vertcat(x_dot_k_minus_3, y_dot_k_minus_3, w_dot_k_minus_3)
        vel_k_minus_4 = cs.vertcat(x_dot_k_minus_4, y_dot_k_minus_4, w_dot_k_minus_4)

        
        # previous_pos = cs.vertcat(x, y, theta)
        # input = cs.vertcat(previous_vel, u, theta)
        # input = cs.transpose(cs.vertcat(vel, u, theta)) # len,channel
        print(f'x_dot:{x_dot_k.shape}')
        print(f'previous_vel:{vel_k.shape}')
        print(f'input:{input.shape}')

        # cur_acc_vel = self.learned_dyn.forward(input)
        # vel_col = cs.vertsplit(previous_vel)

        # set up RK4
        dT = 0.1
        k1 = self.learned_dyn.forward(x_rk4,       u_rk4)
        k2 = self.learned_dyn.forward(x+dT/2*k1,u_rk4)
        k3 = self.learned_dyn.forward(x+dT/2*k2,u_rk4)
        k4 = self.learned_dyn.forward(x+dT*k3,  u_rk4)
        xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)
        # change z
        z_next = z_update               

        
        # ctr_vel = cs.vertcat(v*cs.cos(theta), v*cs.sin(theta), omega)
        # u_combine = cs.vertcat(v, omega, dt)
        # state = cs.vertcat(x, y, theta, x_dot, y_dot, w_dot)
        
        # res_model = self.learned_dyn(input)
        # print(f"res model :{res_model}")
        # print(f"res model shape is :{res_model.shape}")
        # dynamics = cs.Function('vel_dot', [input], [res_model])
        # self.dynamics = dynamics

        f_expl = cs.vertcat(xf,z_next)  

        # print(f"state shape : {state.shape}")
        x_start = np.zeros((x_dim))

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = x  # contains {x, y, theta, x_dot, y_dot, w_dot}
        model.xdot = xdot
        # model.u = u_combine # with dt
        model.u = u
        # model.dt = dt
        model.z = z
        # model.p = cs.vertcat([])
        # model.parameter_values = cs.vertcat([])
        model.disc_dyn_expr = f_expl
        
        model.f_expl = f_expl
        model.f_impl = xdot - f_expl
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "wr"


        return model


class MPC:
    def __init__(self, model, N):
        self.N = N
        self.model = model

    @property
    def simulator(self):
        return AcadosSimSolver(self.ocp())

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def sim(self):
        model = self.model

        t_horizon = 1.
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        # model_ac.p = model.p

        # Dimensions
        nx = 6
        nu = 2
        ny = nx + nu

        # Create OCP object to formulate the optimization
        sim = AcadosSim()
        sim.model = model_ac
        sim.dims.N = N
        sim.dims.nx = nx
        sim.dims.nu = nu
        sim.dims.ny = ny
        sim.solver_options.tf = t_horizon

        # Solver options
        sim.solver_options.Tsim = 1./ 10.
        sim.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        sim.solver_options.hessian_approx = 'GAUSS_NEWTON'
        sim.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 0
        sim.solver_options.nlp_solver_type = 'SQP_RTI'

        return sim

    def ocp(self):
        model = self.model

        t_horizon = 1.
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)

        # Dimensions
        nx = 6
        nu = 2
        ny = nx + nu

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # Initialize cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # original QR 
        # 已有的 Q 矩阵
        Q_existing = np.array([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 2]])

        # 新增变量的权重为 0
        Q_additional = np.zeros((3, 3))

        # 将 Q_existing 和 Q_additional 合并成一个 6x6 的矩阵
        Q = np.block([[Q_existing, np.zeros((3, 3))],
                    [np.zeros((3, 3)), Q_additional]])
        R = np.array([[0.5, 0.0], [0.0, 0.5]])

        # Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]])
        # R = np.array([[0.5, 0.0], [0.0, 0.05]])

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q # not understand ???????
        
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vz = np.array([[]])
        ocp.cost.Vx_e = np.eye(nx)
        


        # Initial reference trajectory (will be overwritten)
        x_ref = np.zeros(nx)
        u_ref = np.zeros(nu)
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start

        # Set constraints
        constraint = cs.types.SimpleNamespace()
        constraint.v_max = 2
        constraint.v_min = -2
        constraint.omega_max = 2
        constraint.omega_min = -2
        constraint.x_min = -2.
        constraint.x_max = 50.
        constraint.y_min = -2.
        constraint.y_max = 50.
    
        ocp.constraints.lbu = np.array([constraint.v_min, constraint.omega_min])
        ocp.constraints.ubu = np.array([constraint.v_max, constraint.omega_max])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbx = np.array([constraint.x_min, constraint.y_min])
        ocp.constraints.ubx = np.array([constraint.x_max, constraint.y_max])
        ocp.constraints.idxbx = np.array([0, 1])
        # new add
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref
        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'


        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.f_impl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        return model_ac



def run():
    N = 10  # YZX notes:true predict horizon 
    Nsim = 100
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
   

    if USE_GPU:
        print('Moving Model to GPU')
        tcn_model = tcn_model.to('cuda:0')
        print('Model is on GPU')

    
    print(f'second step:DoubleIntegratorWithLearnedDynamics')
    model = DoubleIntegratorWithLearnedDynamics(tcn_model)
    print("third step: MPC")
    MPC_object = MPC(model=model.model(), N=N)
    MPC_object.model.function(1)
    solver = MPC_object.solver
    
    # # integrator = MPC(model=model.model(), N=N).simulator

    # print('Warming up model...')
    # x_l = []
    # u_l = []
    # input_l = []
    # for i in range(N):
    #     x_l.append(solver.get(i, "x"))
    #     u_l.append(solver.get(i, "u"))
    #     input_l = np.hstack((x_l, u_l))

    # print('Warmed up!')

    # x = []
    # y_ref = []
    # ts = 1. / N
    # xt = np.zeros(6)# boundary for x 
    # opt_times = []
    # # solver.print_statistics()

    # simX = []
    # simU = []
    # x_current = xt
    # simX.append(xt)
    # x_draw = []
    # y_ref_draw = []
    
    
    # ## yzx init yref
    # xs = np.array([7.,5.,0])
    # t = np.linspace(0, 100, 100)
    #     # print(t)
    # y_xpos =  7* np.sin(0.0156*t)
    # y_ypos = 7 - 7 * np.cos(0.0156*t)
    # y_yaw = 0.0156 * t
    # yref = np.array([y_xpos,y_ypos, y_yaw]).T
    # x_ref = []
    # for t, ref in enumerate(yref):
    #     x_ref.append(ref)
    # Nsim = 0
    # i = 0
    # goal = False
    # # print(x_ref)
    # while(goal is False):
    #     # solve ocp
    #     now = time.time()
    #     # find near ref path
    #     # x_ref = data_ref_sim
    #     index = findnearestIndex(x_current, x_ref)
    #     print(f"nearest index:{index}")

    #     # set y_ref
    #     y_ref_draw.append(x_ref[index])
    #     y_cur_ref = Set_y_ref(x_ref, index, N)
    #     if(y_cur_ref == -1):
    #         print("find goal!!!")
    #         goal = True
    #         break
    #     y_cur_ref = np.array(y_cur_ref)
    #     new_col = np.zeros((y_cur_ref.shape[0], 3))
    #     y_cur_ref = np.hstack((y_cur_ref, new_col))
    #     print(f"len(y_cur_ref):{len(y_cur_ref)}")
    #     y_e = y_cur_ref[len(y_cur_ref)-1]
    #     solver.set(N, 'yref', y_cur_ref[len(y_cur_ref)-1])
    #     for j in range(len(y_cur_ref)-1):
    #         xs_between = np.concatenate((y_cur_ref[j], np.zeros(2)))
    #         solver.set(j, "yref", xs_between)
    #     if(N-len(y_cur_ref) > 0):
    #         for k in range(N-len(y_cur_ref)):
    #             solver.set(len(y_cur_ref)+k, "yref", np.concatenate((y_e, np.zeros(2))))
        
    #     ##  set inertial (stage 0)
    #     # x_current = data_pose_sim[i] # let real sim data be the pose
    #     solver.set(0, 'lbx', x_current)
    #     solver.set(0, 'ubx', x_current)
    #     status = solver.solve()
    #     # solver.get_residuals()
    #     # solver.print_statistics()
    #     if status != 0 :
    #         raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
        
    #     # print(solver.get(0, 'u'))
    #     simU.append(solver.get(0, 'u'))
    #     print(f"solver get u: {solver.get(0, 'u')}")
    #     # simulate system
    #     integrator.set('x', x_current)
    #     integrator.set('u', simU[i])

    #     status_s = integrator.solve()
    #     if status_s != 0:
    #         raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    #     # update
        
    #     x_current = integrator.get('x')
    #     simX.append(x_current)        
    #     x_draw.append(x_current)

    #     # update NN params into ocp solver params
    #     # update MLP part
    #     x_l = []
    #     input_l = []
    #     # for i in range(N):
    #     #     x_l.append(solver.get(i, "x"))
    #     #     u_l.append(solver.get(i, "u"))
    #     #     input_l.append(np.hstack((x_l[i], u_l[i])))
    #     # params = learned_dyn_model.approx_params(np.stack(input_l, axis=0), flat=True)
    #     # for i in range(N):
    #     #     solver.set(i, "p", params[i])

        
    #     # calc end time for each epoch
    #     elapsed = time.time() - now
    #     opt_times.append(elapsed)
    #     Nsim = Nsim+1
    #     i = Nsim
    #     if i==1000:
    #         break
    #     print(f"Nsim:{Nsim}")


    # print(solver.get_stats('residuals'))
    # # print(f"x_current:{x_current}")
    # # print(f"ref:{x_ref}")
    # # plt.figure()
    # # plt.grid(True)
    # simX = np.array(simX)
    # simU = np.array(simU)
    # # plt.show()
    # Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=xt, target_state=x_ref[len(x_ref)-1], robot_states=simX, )
    # # draw(x_draw,'pose with mlp')
    # # draw(y_ref_draw,'y ref')
    # # draw(data_ref_sim, 'sim y ref')
    # # draw(data_pose_sim, 'pose without mlp')
    # plt.show()
    # # for j in range(len(x)):
    # #     print(f'{x[j]}\n')
     
    # print(f'Mean iteration time with CNN ResNet Model: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')

def draw(data,label):
    x = []
    y = []
    for i in range(len(data)):
        p = data[i]
        x.append(p[0])
        y.append(p[1])
    
    plt.plot(x, y, marker='o', linestyle='-',label=label)
    plt.legend()

    
        
def Set_y_ref(ref_path, idx, Horizon):
    cur_ref_path = []
    min_idx = min(idx + Horizon, len(ref_path)-1)
    if(idx == len(ref_path)-1): 
        return -1
    # print(min_idx)
    # print(f"len(ref_path):{len(ref_path)}")
    # print(f"idx+H:{idx + Horizon}")
    for i in range(idx, min_idx+1):
        cur_ref_path.append(ref_path[i])
    # print(len(cur_ref_path))
    return cur_ref_path


def findnearestIndex(cur_pos,ref_path):
    min_dist = 10000
    index = -1
    for i in range(len(ref_path)):
        ref = ref_path[i]
        dx = cur_pos[0] - ref[0]
        dy = cur_pos[1] - ref[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < min_dist:
            min_dist = dist
            index = i
    return index

if __name__ == '__main__':
    
    run()