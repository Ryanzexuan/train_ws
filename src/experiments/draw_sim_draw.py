import sys
sys.path.append('/home/ryan/raigor/train_ws/src/')
from model_fit.gp_common import read_dataset
import casadi as cs
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import ml_casadi.torch as mc
from tqdm import tqdm
import time
import os
import math
import subprocess
import tf.transformations as tf
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy.linalg
from draw import Draw_MPC_point_stabilization_v1


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MODEL_ARCH = 'MLP'  # On of 'MLP' or 'CNNRESNET'
# MLP: Dense Network of 32 layers 256 neurons each
# CNNRESNET: ResNet model with 18 Convolutional layers

USE_GPU = False


def run():
    ds_name = "simplified_sim_dataset"
    ds_metadata = {
        "noisy": True,
        "drag": True,
        "payload": False,
        "motor_noise": True
    }
# #### DATASET LOADING #### #
    if isinstance(ds_name, str):
        try:
            rec_file = "/home/ryan/raigor/train_ws/data/simplified_sim_dataset/train/dataset_gzsim_nominal.csv"
            df_val_pre = pd.read_csv(rec_file,index_col=False)
            print(df_val_pre)
        except:
            print('Could not find test dataset.')
    else:
        raise TypeError("dataset_name must be a string.")

    x_input = df_val_pre['x_position_input']
    y_input = df_val_pre['y_position_input']
    yaw_input = df_val_pre['yaw_input']
    print(f"y_in:{y_input.iloc[:1000,]}")
    x_ref = df_val_pre['x_ref']
    y_ref = df_val_pre['y_ref']
    yaw_ref = df_val_pre['yaw_ref']

    x_draw = []
    x_ref_draw = []
    # x_former = np.column_stack([x_input.iloc[:1500,], y_input.iloc[:1500,], yaw_input.iloc[:1500,]])
    # x_cur_ref = np.column_stack([x_ref.iloc[:1500], y_ref.iloc[:1500], yaw_ref.iloc[:1500]])
    # For dataset_gzsim_nominal drawing
    x_former = np.column_stack([x_input, y_input, yaw_input])
    x_cur_ref = np.column_stack([x_ref, y_ref, yaw_ref])
    for i in range(x_former.shape[0]):
        x_draw.append(x_former[i,:])
        x_ref_draw.append(x_cur_ref[i,:])
    # x_later = np.column_stack()
    # print(x_former)
    plt.grid()
    draw(x_draw,'traj nominal')
    draw(x_ref_draw, 'traj ref')
    # Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=np.array([0., 0., 0.]), target_state=np.array([5., 5., 0.]), robot_states=np.array(x_former), )
    plt.show()


def draw(data,label):
    x = []
    y = []
    for i in range(len(data)):
        p = data[i]
        x.append(p[0])
        y.append(p[1])
    
    plt.plot(x, y, marker='o', linestyle='-',label=label)
    plt.legend()


if __name__ == '__main__':
    run()