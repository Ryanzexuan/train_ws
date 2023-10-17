import casadi as cs
import numpy as np
import torch
import torchvision.models as models
import ml_casadi.torch as mc
from tqdm import tqdm
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
from src.model_fitting.mlp_common import NormalizedMLP
import time
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MODEL_ARCH = 'MLP'  # On of 'MLP' or 'CNNRESNET'
# MLP: Dense Network of 32 layers 256 neurons each
# CNNRESNET: ResNet model with 18 Convolutional layers

USE_GPU = False


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
        ## YZX TBD s,s_dot,s_dot_dot dimension need to be changed 

        s = cs.MX.sym('s', 3) # state, can be position 
        s_dot = cs.MX.sym('s_dot', 3) # deriavtive of state, can be velocity
        # s_dot_dot = cs.MX.sym('s_dot_dot', 3) # derivative again, can be acceleration or control input
        u = cs.MX.sym('u', 2)
        dt = cs.MX.sym('dt',1)
        u_combine = cs.vertcat(u, dt)
        state = cs.MX.sym('state',3)
        state_dot = cs.MX.sym('state_dot',3)
        x_input = cs.vertcat(state,u,dt) # need a dt
        # x_input_dot = cs.vertcat(state_dot, u_dot, dt_dot)
        # x = cs.vertcat(s, s_dot)
        # cs.disp(x)
        x_dot = cs.vertcat(s_dot)
        print("1 in")
        res_model = self.learned_dyn.approx(x_input) # (f_a+mac(df_a,(vertcat(s, s_dot)-a),zeros(2x1))) 
        print(res_model)
        p = self.learned_dyn.sym_approx_params(order=1, flat=True) # vertcat(a, f_a, vec(df_a)) 
        print(p)
        parameter_values = self.learned_dyn.approx_params(np.array([0, 0, 0, 0, 0, 0]), flat=True, order=1) # obtain value:(a, f_a, vec(df_a)) ;   np.array([0, 0]) transfers to variable a;
        
        ## YZX with dt
        rhs = [u[0]*cs.cos(state[2])*dt,u[1]*cs.sin(state[2])*dt,u[1]*dt] # u =[v,w] s = [x,y,theta]
        f = cs.Function('f',[state, u, dt],[cs.vcat(rhs)],['input_state','u','dt'],['next_state_nomial'])
        f_expl = f(state,u,dt) + res_model  # f_expl means the next state would be "f(s_dot,u) + f_a+mac(df_a,(vertcat(s, s_dot)-a)"

        
        # ## YZX without dt
        # rhs = [u[0]*cs.cos(state[2]),u[1]*cs.sin(state[2]),u[1]] # u =[v,w] s = [x,y,theta]
        # f = cs.Function('f',[state, u],[cs.vcat(rhs)],['state','u'],['next_state_nomial'])
        # f_expl = f(state,u) + res_model  # f_expl means the next state would be "f(s_dot,u) + f_a+mac(df_a,(vertcat(s, s_dot)-a)"
        # print(f_expl)
        
        x_start = np.zeros((3))

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = state 
        model.xdot = state_dot
        model.u = u_combine # combine time as input
        # model.dt = dt
        model.z = cs.vertcat([])
        model.p = p
        model.parameter_values = parameter_values
        model.f_expl = f_expl
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
        return AcadosSimSolver(self.sim())

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
        nu = 3
        ny = 3

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
        nu = 3
        ny = 3

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

        ocp.cost.W = np.diag([7, 2, 8])

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :ny] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vz = np.array([[]])
        ocp.cost.Vx_e = np.zeros((ny, nx))
        ocp.cost.W_e = np.zeros((3,3)) # not understand ???????
        ocp.cost.yref_e = np.array([0., 0., 0.])

        # Initial reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(3)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start

        # Set constraints
        a_max = 10
        ocp.constraints.lbu = np.array([-a_max, -a_max, -a_max])
        ocp.constraints.ubu = np.array([a_max, a_max, a_max])
        ocp.constraints.idxbu = np.array([0,1,2])

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
    N = 10 
    # YZX add need to change each time after train
    saved_dict = torch.load("/home/ryan/train_ws/results/model_fitting/d264f6b/simple_sim_mlp/drag__motor_noise__noisy__no_payload.pt")
    if MODEL_ARCH == 'MLP':
        learned_dyn_model = MLP(saved_dict['input_size'], saved_dict['hidden_size'], 
                                saved_dict['output_size'], saved_dict['hidden_layers'], 'Tanh')
        learn_model = NormalizedMLP(learned_dyn_model, torch.tensor(np.zeros((saved_dict['input_size'],))).float(),
                                  torch.tensor(np.zeros((saved_dict['input_size'],))).float(),
                                  torch.tensor(np.zeros((saved_dict['output_size'],))).float(),
                                  torch.tensor(np.zeros((saved_dict['output_size'],))).float())
        print("ok")
        learn_model.load_state_dict(saved_dict['state_dict'])
        learn_model.eval()

    elif MODEL_ARCH == 'CNNRESNET':
        learned_dyn_model = mc.TorchMLCasadiModuleWrapper(CNNResNet(), input_size=2, output_size=2)

    if USE_GPU:
        print('Moving Model to GPU')
        learn_model = learn_model.to('cuda:0')
        print('Model is on GPU')
    model = DoubleIntegratorWithLearnedDynamics(learn_model)
    solver = MPC(model=model.model(), N=N).solver

    print('Warming up model...')
    x_l = []
    u_l = []
    input_l = []
    for i in range(N):
        x_l.append(solver.get(i, "x"))
        u_l.append(solver.get(i, "u"))
        input_l = np.hstack((x_l, u_l))
        print(input_l.shape)
    for i in range(20):
        learned_dyn_model.approx_params(np.stack(input_l, axis=0), flat=True)
    print('Warmed up!')

    x = []
    x_ref = []
    ts = 1. / N
    xt = np.array([0., 0., 0]) # boundary for x 
    opt_times = []
    
    # tq = tqdm(range(50))
    for i in range(50):
        now = time.time()
        t = np.linspace(i * ts, i * ts + 1., 10)
        y_xpos = 10 * np.sin(0.0156*t)
        y_ypos = 10 - 10 * np.cos(0.0156*t)
        y_yaw = 0.0156 * t
        yref = np.array([y_xpos,y_ypos, y_yaw]).T
        # yref = np.sin(0.5 * t + np.pi / 2)
        # x_ref.append(yref[0])
        for t, ref in enumerate(yref):
            solver.set(t, "yref", ref)
        # define the first state x must be the current time while lbx & ubx can be used to limit intermediate states status 
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        print("start solving")
        solver.solve()
        print(i)
        xt = solver.get(1, "x")
        x.append(xt)
        # the first deriavtive of MLP learned models as simplifying the NN 
        x_l = []
        u_l = []
        for i in range(N):
            x_l.append(solver.get(i, "x"))
            u_l.append(solver.get(i, "u"))
            input_l = np.hstack((x_l, u_l))
        params = learned_dyn_model.approx_params(np.stack(input_l, axis=0), flat=True)
        for i in range(N):
            solver.set(i, "p", params[i])

        elapsed = time.time() - now
        opt_times.append(elapsed)
        
    draw(x)
    print(f'Mean iteration time with CNN ResNet Model: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')

def draw(data):
    plt.figure(figsize=(5,5))
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    # print(data[50,1])
    for j in range(len(data)):
        x = []
        y = []
        p = data[j]
        # print(p)
        x.append(p[0])
        y.append(p[1])
        #plt.plot(x,y)

    x = np.linspace(-5, 5, 100)
    y1 = np.sin(x)
    plt.plot(x,y1)
    plt.show()




if __name__ == '__main__':
    run()