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
from turtlesim.msg import Pose  # 导入 Pose 消息类型




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

    def gen_input(self, x, u):
        print(f'new_gen_input!!!!')
        print(f'x:{x.shape}') # x,y,theta,x_dot,y_dot,theta_dot,
        print(f'u:{u.shape}')
    
        # generate input for nn
        x_all = cs.vertcat(cs.transpose(x[3:6,:]))
        u_previous = x[6:,:-1]
        u_all = cs.transpose(cs.horzcat(u_previous,u))
        theta_all = cs.vertcat(cs.transpose(x[2,:]))
        input = cs.horzcat(x_all, u_all, theta_all)

        # generate for RK4
        x_rk = cs.vertcat(cs.transpose(x_all),cs.transpose(theta_all)) # vel,theta 4*5
        u_rk = u_all
        # generate u for integrate into new x
        u_for_integrate = u_all[1:,:] 
        u_for_integrate = cs.transpose(cs.vertcat(u_for_integrate,u_all[0,:])) # add nonsense u_all[0,:]
        
        print(f'x_all:{x_all.shape}')
        print(f'u_all :{u_all.shape}')
        print(f'theta_all:{theta_all.shape}')
        print(f'input gen :{input.shape}') # 5*6
        print(f'x_rk:{x_rk.shape}') # 4*5 vel,theta
        return input,x_rk,u_rk,u_for_integrate

    def gen_rk_input(self, x,u):
        print(f'gen rk input!!!!!!!!')
        x_all_fromrk = cs.transpose(x[:3,:]) # 5*3
        theta_all_fromrk  = cs.transpose(x[3,:])
        u__all_fromrk = u
        rk_input = cs.horzcat(x_all_fromrk,u__all_fromrk,theta_all_fromrk)

        print(f'x_all:{x_all_fromrk .shape}')
        print(f'u_all :{u__all_fromrk .shape}')
        print(f'theta_all:{theta_all_fromrk .shape}')
        print(f'input gen :{rk_input.shape}') # 5*6 [vel,u,theta]
        return rk_input
    
    def rk_permute(self, x):
        # x : [vel,acc] 5*6
        x = cs.transpose(x)
        x_rk_vel = x[-3:,:] # 3*5
        x_rk_theta = x[2,:] # 1*5
        x_rk_frompermute = cs.vertcat(x_rk_vel,x_rk_theta)
        print(f'x_rk_frompermute:{x_rk_frompermute.shape}')
        return x_rk_frompermute

    def create_x4input(self, x):
        sequence_len = 5
        print(f'create x4input!!!!!!!')
        pose = x[:3*sequence_len].reshape((3,sequence_len))
        vel = x[3*sequence_len:3*sequence_len+3*sequence_len].reshape((3,sequence_len))
        u_pst = x[3*sequence_len+3*sequence_len:].reshape((2,sequence_len))
        print(f'pose:{pose.shape}')
        print(f'vel:{vel.shape}')
        print(f'u_pst:{u_pst.shape}')
        new_x = cs.vertcat(pose,vel,u_pst)
        newx_no_u = cs.vertcat(pose,vel)
        return new_x,newx_no_u

    def model(self):
        s_dot_dim = self.learned_dyn.output_size
        x_dim = self.learned_dyn.input_size
        # pose init
        pose = cs.MX.sym('new_pose', 3*self.sequence_len) # 3*5
        # vel init
        pose_dot = cs.MX.sym('new_vel', 3*self.sequence_len) # 3*5
        # past u init
        u_pst = cs.MX.sym('u_pst', 2*self.sequence_len) # 2*5 one nonsense variable
        # state x init
        x = cs.vertcat(pose,pose_dot,u_pst)
        x4_input,newx_no_u = self.create_x4input(x)
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


        # net input generation
        input,x_rk,u_rk,u_for_integrate = self.gen_input(x4_input, u)

        print(f'input:{input.shape}')

        # cur_acc_vel = self.learned_dyn.forward(input)
        # vel_col = cs.vertsplit(previous_vel)

        # set up RK4
        dT = 0.1
        k1 = self.learned_dyn.forward(self.gen_rk_input(x_rk,u_rk))
        print(f'k1:{k1.shape}') # 5*6 vel,acc
        k1_new = self.rk_permute(k1)
        print(f'k1_new:{k1_new.shape}')
        k2 = self.learned_dyn.forward(self.gen_rk_input(x_rk+dT/2*k1_new,u_rk))
        k2_new = self.rk_permute(k2)
        k3 = self.learned_dyn.forward(self.gen_rk_input(x_rk+dT/2*k2_new,u_rk))
        k3_new = self.rk_permute(k3)
        k4 = self.learned_dyn.forward(self.gen_rk_input(x_rk+dT*k3_new,  u_rk))
        print(f'newx_no_u:{newx_no_u.shape}')
        print(f'u_for_integrate:{u_for_integrate.shape}')
        xf = cs.transpose(newx_no_u) + dT/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        xf_u = cs.transpose(cs.vertcat(cs.transpose(xf),u_for_integrate)).reshape((8*self.sequence_len,1))      



        # print(f"state shape : {state.shape}")
        x_start = np.zeros((x_dim))

        # store to struct9
        # model = cs.types.SimpleNamespace()
        model= AcadosModel()
        model.x = x  # contains {x, y, theta, x_dot, y_dot, w_dot}
        # model.xdot = xdot
        # model.u = u_combine # with dt
        model.u = u
        # model.dt = dt
        model.z = cs.vertcat([])
        model.p = cs.vertcat([])
        # model.parameter_values = cs.vertcat([])
        model.disc_dyn_expr = xf_u
        
        model.f_expl = cs.vertcat([])
        model.f_impl = cs.vertcat([])
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
        # model_ac = self.acados_model(model=model)
        # model_ac.p = model.p

        # Dimensions
        nx = 40 
        nu = 2
        ny = nx + nu

        # Create OCP object to formulate the optimization
        sim = AcadosSim()
        sim.model = model
        sim.dims.N = N
        sim.dims.nx = nx
        sim.dims.nu = nu
        sim.dims.ny = ny
        sim.solver_options.tf = t_horizon

        # Solver options
        sim.solver_options.Tsim = 1./ 10.
        sim.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        sim.solver_options.hessian_approx = 'GAUSS_NEWTON'
        sim.solver_options.integrator_type = 'DISCRETE'
        # ocp.solver_options.print_level = 0
        sim.solver_options.nlp_solver_type = 'SQP_RTI'

        return sim

    def ocp(self):
        model = self.model

        t_horizon = 1.
        N = self.N

        # # Get model
        # model_ac = self.acados_model(model=model)

        # Dimensions
        nx = 40
        nu = 2
        ny = nx + nu

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # Initialize cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # for Vxe
        A = np.zeros((nx,nx))
        A[4,4] = 1
        A[9,9] = 1
        A[14,14] = 1
        # for Vx
        B = np.zeros((ny, nx))
        B[4,4] = 1
        B[9,9] = 1
        B[14,14] = 1
        
        # original QR 
        # 已有的 Q 矩阵
        Q_existing = np.zeros((nx,nx))
        Q_existing[4,4] = 10
        Q_existing[9,9] = 10
        Q_existing[14,14] = 2

        # 将 Q_existing 和 Q_additional 合并成一个 6x6 的矩阵
        Q = Q_existing

        R = np.array([[0.5, 0.0], [0.0, 0.5]])

        # Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]])
        # R = np.array([[0.5, 0.0], [0.0, 0.05]])

        ocp.cost.W = scipy.linalg.block_diag(Q, R) # ny*ny
        ocp.cost.W_e = Q # nx*nx 

        ocp.cost.Vx = B
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        # ocp.cost.Vz = np.array([[]])
        ocp.cost.Vx_e = A              
        


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
        ocp.constraints.idxbx = np.array([4, 9])
        # new add
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref
        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'DISCRETE'
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
    solver = MPC(model=model.model(), N=N).solver
    # integrator = MPC(model=model.model(), N=N).simulator
    print('Warming up model...')


    x_l = []
    u_l = []
    input_l = []
    for i in range(N):
        x_l.append(solver.get(i, "x"))
        u_l.append(solver.get(i, "u"))
        input_l = np.hstack((x_l, u_l))

    print('Warmed up!')

    x = []
    y_ref = []
    ts = 1. / N
    xt = np.array([0., 0., 0])# boundary for x 
    opt_times = []
    # solver.print_statistics()

    simX = []
    simU = []
    simX_b4 = []
    x_current = np.array([cur_rec_state_set[0], cur_rec_state_set[1], cur_rec_state_set[2],
                          cur_rec_state_set[4], cur_rec_state_set[5], cur_rec_state_set[6]]) # pose,vel(x,y,theta)
    x_current_withtime = np.hstack([np.zeros(4), cur_rec_state_set[0], np.zeros(4), cur_rec_state_set[1], np.zeros(4), cur_rec_state_set[2],
                          np.zeros(4), cur_rec_state_set[4], np.zeros(4), cur_rec_state_set[5], np.zeros(4), cur_rec_state_set[6], np.zeros(10)]) # pose,vel(x,y,theta)
    simX.append(xt)
    x_draw = []
    y_ref_draw = []
    
    
    ## yzx init yref
    xs = np.array([10.,10.,0])
    t = np.linspace(0, 100, 50)
    # print(len(t))
    ref_len = len(t)
    y_xpos =  (10 * np.sin(0.0156*t)).reshape(-1,1)
    y_ypos = (10 - 10 * np.cos(0.0156*t)).reshape(-1,1)
    y_yaw = (0.0156 * t).reshape(-1,1)
    print(y_xpos.shape)

    yref = np.hstack((np.zeros((ref_len,4)),y_xpos,np.zeros((ref_len,4)),y_ypos,np.zeros((ref_len,4)),y_yaw,np.zeros((ref_len,25))))
    # yref = np.array([y_xpos,y_ypos, y_yaw]).T
    print(yref.shape)
    x_ref = []
    for t, ref in enumerate(yref):
        x_ref.append(ref)
    Nsim = 0
    i = 0
    goal = False
    control = Twist()
    # print(x_ref)
    while(goal is False or rospy.is_shutdown()):
        # solve ocp
        now = time.time()
        # find near ref path
        # x_ref = data_ref_sim
        index = findnearestIndex(x_current, x_ref)
        print(index)

        # set y_ref
        y_cur_ref = Set_y_ref(x_ref, index, N)
        # print(f'y_cur_ref:{y_cur_ref.shape}')
        if(y_cur_ref == -1):
            print("find goal!!!")
            goal = True
            break
        y_ref_draw.append(x_ref[index])
        simX_b4.append(x_current)
        y_cur_ref = np.array(y_cur_ref)
        # new_col = np.zeros((y_cur_ref.shape[0], 3))
        # y_cur_ref = np.hstack((y_cur_ref, new_col))
        print(f"len(y_cur_ref):{len(y_cur_ref)}")
        y_e = y_cur_ref[len(y_cur_ref)-1]
        solver.set(N, 'yref', y_cur_ref[len(y_cur_ref)-1])
        for j in range(len(y_cur_ref)-1):
            xs_between = np.concatenate((y_cur_ref[j], np.zeros(2)))
            solver.set(j, "yref", xs_between)
        if(N-len(y_cur_ref) > 0):
            for k in range(N-len(y_cur_ref)):
                solver.set(len(y_cur_ref)+k, "yref", np.concatenate((y_e, np.zeros(2))))
        
        ##  set inertial (stage 0)
        # x_current = data_pose_sim[i] # let real sim data be the pose
        solver.set(0, 'lbx', x_current_withtime)
        solver.set(0, 'ubx', x_current_withtime)
        status = solver.solve()
        # solver.get_residuals()
        # solver.print_statistics()
        
        if status != 0 :
            continue
            # raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

        cur_u = np.array(solver.get(0, 'u'))
        print(f'Get u from solverS:{cur_u}')
        control.linear.x = cur_u[0]
        control.angular.z = cur_u[1]
        nn_msg.publish(control)
        # rate.sleep()
        simU.append(solver.get(0, 'u'))
        
        
        x_current = np.array([cur_rec_state_set[0], cur_rec_state_set[1], cur_rec_state_set[2],
                          cur_rec_state_set[4], cur_rec_state_set[5], cur_rec_state_set[6]])   
        x_current_withtime = np.hstack([np.zeros(4), cur_rec_state_set[0], np.zeros(4), cur_rec_state_set[1], np.zeros(4), cur_rec_state_set[2],
                          np.zeros(4), cur_rec_state_set[4], np.zeros(4), cur_rec_state_set[5], np.zeros(4), cur_rec_state_set[6], np.zeros(10)]) # pose,vel(x,y,theta)

        # rospy.loginfo(f"x_current:{x_current}\n")
        simX.append(x_current)        
        x_draw.append(x_current)

        # update NN params into ocp solver params
        # update MLP part

        
        # calc end time for each epoch
        elapsed = time.time() - now
        opt_times.append(elapsed)
        Nsim = Nsim+1
        i = Nsim
        if i==2500:
            break
        print(f"Nsim:{Nsim}")


    print(solver.get_stats('residuals'))
    # print(f"x_current:{x_current}")
    # print(f"ref:{x_ref}")
    # plt.figure()
    # plt.grid(True)
    # simX = np.array(simX)
    # simU = np.array(simU)
    # simX_b4 = np.array(simX_b4)
    # y_ref_draw = np.array(y_ref_draw)
    # # plt.show()
    # # Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=xt, target_state=x_ref[len(x_ref)-1], robot_states=simX, )
    # # draw(x_draw,'pose with mlp')
    # # draw(y_ref_draw,'y ref')
    # # draw(data_ref_sim, 'sim y ref')
    # # plt.show()
    
    # save_data2csv(simX[1:],simU,simX_b4,y_ref_draw)  
    print(f'Mean iteration time with CNN ResNet Model: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')

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
    print(f"len(ref_path):{len(ref_path)}")
    print(f"idx+H:{idx + Horizon}")
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

def Callback_base(msg):
    rospy.loginfo("msg got~!!!!!")
    quaternion = msg.pose.pose.orientation
    roll, pitch, yaw = tf.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    # rospy.loginfo(f"x pose{msg.pose.pose.position.x}")
    cur_rec_state_set[0] = msg.pose.pose.position.x 
    cur_rec_state_set[1] = msg.pose.pose.position.y 
    cur_rec_state_set[2] = yaw
    cur_rec_state_set[3] = rospy.Time.now().to_sec()
    cur_rec_state_set[4] = msg.twist.twist.linear.x 
    cur_rec_state_set[5] = msg.twist.twist.linear.y 
    cur_rec_state_set[6] = msg.twist.twist.angular.z

def Callback_base_turtle(msg):
    rospy.loginfo("msg got~!!!!!")
    # quaternion = msg.pose.pose.orientation
    # rospy.loginfo(f"x pose{msg.pose.pose.position.x}")
    cur_rec_state_set[0] = msg.x 
    cur_rec_state_set[1] = msg.y 
    cur_rec_state_set[2] = msg.theta
    cur_rec_state_set[3] = rospy.Time.now().to_sec()
    cur_rec_state_set[4] = msg.linear_velocity * np.cos(msg.theta)
    cur_rec_state_set[5] = msg.linear_velocity * np.sin(msg.theta)
    cur_rec_state_set[6] = msg.angular_velocity
    # rospy.loginfo(f'cur state:{cur_rec_state_set}')

def save_data2csv(next, u, pst, ref):
    # print(f"next:{next}")
    # print(f"u:{u}")
    # print(f"pst:{pst}")
    # print(f"next:{next.shape[0]}")
    # print(f"u:{u.shape[0]}")
    # print(f"pst:{pst.shape[0]}")
    data = pd.DataFrame({'x_position_input': pst[:, 0],
                        'y_position_input': pst[:, 1],
                        'yaw_input': pst[:, 2],
                        'con_x_input': u[:, 0],
                        'con_z_input':u[:, 1],
                        'x_position_output': next[:,0],
                        'y_position_output': next[:,1],
                        'yaw_output': next[:,2],
                        'x_ref': ref[:, 0],
                        'y_ref': ref[:, 1],
                        'yaw_ref': ref[:, 2]})
    data.to_csv("/home/ryan/raigor/train_ws/src/TCN/tcn_with_mlp/tcn_mpc_test/test.csv")

if __name__ == '__main__':
    rospy.init_node("acados", anonymous=True)
    rospy.Subscriber("/turtle1/pose", Pose, Callback_base_turtle)
    nn_msg = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)  
    rate = rospy.Rate(1)   
    run()



