import sys
sys.path.append("/home/ryan/raigor/train_ws/src/")
import pandas as pd
import rospy
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
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_msgs.msg import String

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MODEL_ARCH = 'MLP'  # On of 'MLP' or 'CNNRESNET'
# MLP: Dense Network of 32 layers 256 neurons each
# CNNRESNET: ResNet model with 18 Convolutional layers

USE_GPU = False
cur_rec_state_set = np.zeros(7)
cur_cmd = np.zeros(2)

flag = 0

class MLP(mc.nn.MultiLayerPerceptron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.output_layer.bias.fill_(0.)
            self.output_layer.weight.fill_(0.)


def data_prepare():
    rec_file = "/home/ryan/raigor/train_ws/data/simplified_sim_dataset/train/dataset_gzsim_nominal.csv"
    ds = pd.read_csv(rec_file,index_col=None,header=0)
    return ds

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

df_val_pre = data_prepare()

with torch.no_grad():

