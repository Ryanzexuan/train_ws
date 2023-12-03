import sys
sys.path.append("/home/ryan/raigor/train_ws/src/")
import rospy
import casadi as cs
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import ml_casadi.torch as mc
from tqdm import tqdm
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
from src.model_fitting.mlp_common import NormalizedMLP
import time
import os
import math
import subprocess
import tf.transformations as tf
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy.linalg
from draw import Draw_MPC_point_stabilization_v1
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

    def model(self):
        x = cs.MX.sym('x') # state, can be position 
        y = cs.MX.sym('y') # deriavtive of state, can be velocity
        x_dot = cs.MX.sym('x_dot') # x linear vel
        y_dot = cs.MX.sym('y_dot') # y linear vel
        w_dot = cs.MX.sym('w_dot') # angular vel
        previous_vel = cs.vertcat(x_dot, y_dot, w_dot)
        
        theta = cs.MX.sym('theta')
        v = cs.MX.sym('v')
        omega = cs.MX.sym('omega')
        u = cs.vertcat(v, omega)
        ctr_vel = cs.vertcat(v*cs.cos(theta), v*cs.sin(theta), omega)
        # u_combine = cs.vertcat(v, omega, dt)
        state = cs.vertcat(x, y, theta)
        input = cs.vertcat(ctr_vel, previous_vel)
        res_model = self.learned_dyn(input)
        # print(f"res_model:\n{res_model}")
        # p = self.learned_dyn.sym_approx_params(order=1, flat=True)
        # # print(f"p:\n{p}")
        # parameter_values = self.learned_dyn.approx_params(np.array([0, 0, 0, 0, 0, 0]), flat=True, order=1)
        # print(f"params:\n{parameter_values}")
        ## YZX without dt
        s_dot_dim = self.learned_dyn.output_size
        x_dim = self.learned_dyn.input_size
        
        f_expl = res_model 
        s_dot = cs.MX.sym('x_dot', (s_dot_dim))

        x_start = np.zeros((x_dim))

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = state 
        model.xdot = s_dot
        # model.u = u_combine # with dt
        model.u = u
        # model.dt = dt
        model.z = cs.vertcat([])
        # model.p = cs.vertcat([])
        # model.parameter_values = cs.vertcat([])
        model.f_expl = f_expl
        model.f_impl = x_dot - f_expl
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
        model_ac.p = model.p

        # Dimensions
        nx = 3
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
        model_ac.p = model.p

        # Dimensions
        nx = 3
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
        Q = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 2]])
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
        constraint.v_max = 0.6
        constraint.v_min = -0.6
        constraint.omega_max = np.pi/4.0
        constraint.omega_min = -np.pi/4.0
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

        ocp.parameter_values = model.parameter_values

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
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
    path = os.path.join(ws_path, 'mlp_fit_ros/results', pt_name)
    print(f"path is {path}")
    saved_dict = torch.load(path)
    if MODEL_ARCH == 'MLP':
        learned_dyn_model = MLP(saved_dict['input_size'], saved_dict['hidden_size'], 
                                saved_dict['output_size'], saved_dict['hidden_layers'], 'Tanh')
        learn_model = NormalizedMLP(learned_dyn_model, torch.tensor(np.zeros((saved_dict['input_size'],))).float(),
                                  torch.tensor(np.zeros((saved_dict['input_size'],))).float(),
                                  torch.tensor(np.zeros((saved_dict['output_size'],))).float(),
                                  torch.tensor(np.zeros((saved_dict['output_size'],))).float())
        print("load train model ok")
        learn_model.load_state_dict(saved_dict['state_dict'])
        learn_model.eval()

    elif MODEL_ARCH == 'CNNRESNET':
        learned_dyn_model = mc.TorchMLCasadiModuleWrapper(CNNResNet(), input_size=2, output_size=2)

    if USE_GPU:
        print('Moving Model to GPU')
        learn_model = learn_model.to('cuda:0')
        print('Model is on GPU')

    

    model = DoubleIntegratorWithLearnedDynamics(learn_model)
    print("model init successfully")
    solver = MPC(model=model.model(), N=N).solver
    integrator = MPC(model=model.model(), N=N).simulator

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
    xt = np.array([0., 0., 0]) # boundary for x 
    opt_times = []
    # solver.print_statistics()

    simX = []
    simU = []
    simX_b4 = []
    x_current = np.array([cur_rec_state_set[0], cur_rec_state_set[1], cur_rec_state_set[2]])
    simX.append(xt)
    x_draw = []
    y_ref_draw = []
    
    
    ## yzx init yref
    xs = np.array([7.,5.,0])
    t = np.linspace(0, 100, 50)
        # print(t)
    y_xpos =  7 * np.sin(0.0156*t)
    y_ypos = 7 - 7 * np.cos(0.0156*t)
    y_yaw = 0.0156 * t
    yref = np.array([y_xpos,y_ypos, y_yaw]).T
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
        if(y_cur_ref == -1):
            print("find goal!!!")
            goal = True
            break
        y_ref_draw.append(x_ref[index])
        simX_b4.append(x_current)
        y_cur_ref = np.array(y_cur_ref)
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
        solver.set(0, 'lbx', x_current)
        solver.set(0, 'ubx', x_current)
        status = solver.solve()
        solver.get_residuals()
        solver.print_statistics()
        # print(solver.get_residuals())
        if status != 0 :
            continue
            # raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
        
        # print(solver.get(0, 'u'))
        cur_u = np.array(solver.get(0, 'u'))
        control.linear.x = cur_u[0]
        control.angular.z = cur_u[1]
        nn_msg.publish(control)
        # rate.sleep()
        simU.append(solver.get(0, 'u'))
        # simulate system
        integrator.set('x', x_current)
        integrator.set('u', simU[i])

        status_s = integrator.solve()
        if status_s != 0:
            continue
            # raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        # update
        
        x_current = np.array([cur_rec_state_set[0], cur_rec_state_set[1], cur_rec_state_set[2]])
        rospy.loginfo(f"x_current:{x_current}\n")
        simX.append(x_current)        
        x_draw.append(x_current)

        # update NN params into ocp solver params
        # update MLP part
        x_l = []
        input_l = []
        for i in range(N):
            x_l.append(solver.get(i, "x"))
            u_l.append(solver.get(i, "u"))
            input_l.append(np.hstack((x_l[i], u_l[i])))
        params = learned_dyn_model.approx_params(np.stack(input_l, axis=0), flat=True)
        for i in range(N):
            solver.set(i, "p", params[i])

        
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
    simX = np.array(simX)
    simU = np.array(simU)
    simX_b4 = np.array(simX_b4)
    y_ref_draw = np.array(y_ref_draw)
    # plt.show()
    # Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=xt, target_state=x_ref[len(x_ref)-1], robot_states=simX, )
    # draw(x_draw,'pose with mlp')
    # draw(y_ref_draw,'y ref')
    # draw(data_ref_sim, 'sim y ref')
    # plt.show()
    
    save_data2csv(simX[1:],simU,simX_b4,y_ref_draw)  
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
    data.to_csv("/home/ryan/raigor/train_ws/data/simplified_sim_dataset/train/dataset_gzsim_mlp.csv")

if __name__ == '__main__':
    rospy.init_node("acados", anonymous=True)
    rospy.Subscriber("/base_pose_ground_truth", Odometry, Callback_base)
    nn_msg = rospy.Publisher('/cmd_vel', Twist, queue_size=10)  
    rate = rospy.Rate(1)   
    run()