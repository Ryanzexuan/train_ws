import time
import rospy
import os
import casadi as cs
import math
import subprocess
import tf.transformations as tf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.linalg
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tcn_common import TCNModel,NormalizedTCN
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import ml_casadi.torch as mc
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel



class DoubleIntegratorWithLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn

    def model(self):
        x = cs.MX.sym('x') # state, can be position 
        y = cs.MX.sym('y') # deriavtive of state, can be velocity
        theta = cs.MX.sym('theta')
        v = cs.MX.sym('v')
        omega = cs.MX.sym('omega')
        u = cs.vertcat(v, omega)
        dt = cs.MX.sym('dt',1)
        # u_combine = cs.vertcat(v, omega, dt)
        state = cs.vertcat(x, y, theta)
        input = cs.vertcat(state, u)
        model = self.learned_dyn(input)
        # print(f"res_model:\n{res_model}")
        p = self.learned_dyn.sym_approx_params(order=1, flat=True)
        # print(f"p:\n{p}")
        parameter_values = self.learned_dyn.approx_params(np.array([[[0, 0, 0],[0, 0, 0],[0, 0, 0]]]), flat=True, order=1)
        # print(f"params:\n{parameter_values}")
        ## YZX without dt
        rhs = [v*cs.cos(theta),v*cs.sin(theta),omega] # u =[v,w] s = [x,y,theta]
        
        f_expl = model
        x_dot = cs.MX.sym('x_dot', len(rhs))

        x_start = np.zeros((3))

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = state 
        model.xdot = x_dot
        # model.u = u_combine # with dt
        model.u = u
        # model.dt = dt
        model.z = cs.vertcat([])
        model.p = p
        model.parameter_values = parameter_values
        model.f_expl = f_expl
        model.f_impl = x_dot - f(state, u)
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
