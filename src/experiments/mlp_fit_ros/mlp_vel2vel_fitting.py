import sys
sys.path.append('/home/ryan/raigor/train_ws/src/experiments/mlp_fit_ros')
import os
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import ml_casadi.torch as mc

from tqdm import tqdm

from gp_common import read_dataset
from mlp_common import RawDataset, NormalizedMlp, MlpDataset, NomialModel
from config.configuration_parameters import ModelFitConfig as Conf
from src.utils.utils import safe_mkdir_recursive
from src.utils.utils import get_model_dir_and_file



def main(x_features, u_features, reg_y_dims, model_ground_effect, quad_sim_options, dataset_name,
         x_cap, hist_bins, hist_thresh,
         model_name="simple_sim_mlp", epochs=100, batch_size=64, hidden_layers=4, hidden_size=64, lr=1e-4,plot=False):
    """
    reg_y_dims: use to determine y dimension
    """
    
    print(f'epho:{epochs},hiddensize:{hidden_size},hidden_layer:{hidden_layers}')
  

    # #### Get data source path #### #
    
    save_file_path = os.path.join(os.getcwd(), 'results/')
    # print(save_file_path)
    safe_mkdir_recursive(save_file_path)
    print(f'{save_file_path}')
    
    # #### DATASET LOADING #### #
    if isinstance(dataset_name, str):
        try:
            df_val_pre = read_dataset(dataset_name, False, quad_sim_options)
        except:
            print('Could not find test dataset.')
    else:
        raise TypeError("dataset_name must be a string.")
    #
    # df_val_pre [x_position_input	y_position_input	yaw_input	input_time	con_x_input	con_z_input	con_time	x_position_output	y_position_output	yaw_output	output_time]
    
    # #### Data processing and Init Data into private Class

    raw_dataset_train = RawDataset(df_val_pre) # Turn raw data into pandas datatype
    dataset_train = MlpDataset(raw_dataset_train) # Init MLP dataset
    x_mean, x_std, y_mean, y_std = dataset_train.stats() # get info to help normalize data in MLP
    print(x_mean, x_std, y_mean, y_std)
    input_dims = 6  # MLP input dimension
    print(input_dims,len(reg_y_dims))
    data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0) # dataloader
    mlp_model = mc.nn.MultiLayerPerceptron(input_dims, hidden_size, len(reg_y_dims), hidden_layers, 'Tanh') # init MLP network structure
    model = NormalizedMlp(mlp_model, torch.tensor(x_mean).float(), torch.tensor(x_std).float(), 
                          torch.tensor(y_mean).float(), torch.tensor(y_std).float()) # normalize MLP
    
    cuda_name = 'cuda:0'
    if torch.cuda.is_available():
        model = model.to(cuda_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # #### Train #### # 
    loss_infos = []
    bar = tqdm(range(epochs))
    for i in bar:
        model.train()
        losses = []
        for x, y in data_loader:
            if torch.cuda.is_available():
                    x = x.to(cuda_name)
                    y = y.to(cuda_name)
            x = x.float()
            y = y.float()
            # print(y)
            optimizer.zero_grad()
            y_pred = model(x)  #  using forward not using model.model(x)
            loss = torch.square(y-y_pred).mean()
            # print(loss)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        # yaw = 0.0312719
        # model_pred = model(torch.tensor([-0.0560548, 0.000513381, 0.0312719, 0.461716*np.cos(yaw), 0.461716*np.sin(yaw), 0.0313876]).to(cuda_name))
        # print(f"real next res vel:{model_pred}")
        train_loss_mean = np.mean(losses)
        loss_info = train_loss_mean
        loss_infos.append(loss_info)
        bar.set_description(f'Train Loss: {train_loss_mean:.6f}')
        bar.refresh()

    save_dict = {
            'state_dict': model.state_dict(),
            'input_size': input_dims,
            'hidden_size': hidden_size,
            'output_size': len(reg_y_dims),
            'hidden_layers': hidden_layers
        }
    print(f'save file in :{save_file_path}')
    torch.save(save_dict, os.path.join(save_file_path, 'ros_mlp.pt'))
    print(save_dict["output_size"])
    
    if 1:
        import matplotlib.pyplot as plt
        plt.plot(loss_infos)
        plt.show()
    # print("ok")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default="100",
                        help="Number of epochs to train the model.")

    parser.add_argument("--batch_size", type=int, default="64",
                        help="Batch Size.")

    parser.add_argument("--hidden_layers", type=int, default="4",
                        help="Number of hidden layers.")

    parser.add_argument("--hidden_size", type=int, default="64",
                        help="Size of hidden layers.")
    
    parser.add_argument("--lr", type=float, default="1e-4",
                        help="Learning rate.")

    parser.add_argument("--model_name", type=str, default="",
                        help="Name assigned to the trained model.")

    parser.add_argument('--x', nargs='+', type=int, default=[2],
                        help='Regression X variables. Must be a list of integers between 0 and 12. Velocities xyz '
                             'correspond to indices 7, 8, 9.')

    parser.add_argument('--u', action="store_true",
                        help='Use the control as input to the model.')

    parser.add_argument("--y", nargs='+', type=int, default=[1,2,3],
                        help="Regression Y variable. Must be an integer between 0 and 12. Velocities xyz correspond to"
                             "indices 7, 8, 9.")

    parser.add_argument('--ge', action="store_true",
                        help='Use the ground map as input to the model.')

    parser.add_argument("--plot", dest="plot", action="store_true",
                        help="Plot the loss after training.")
    parser.set_defaults(plot=False)

    input_arguments = parser.parse_args()

    # Use vx, vy, vz as input features
    x_feats = input_arguments.x
    if input_arguments.u:
        u_feats = [0, 1, 2, 3]
    else:
        u_feats = []

    model_ground_effect = input_arguments.ge

    # Regression dimension
    y_regressed_dims = input_arguments.y
    model_name = input_arguments.model_name

    epochs = input_arguments.epochs
    batch_size = input_arguments.batch_size
    hidden_layers = input_arguments.hidden_layers
    hidden_size = input_arguments.hidden_size
    lr = input_arguments.lr
    plot = input_arguments.plot

    ds_name = Conf.ds_name
    simulation_options = Conf.ds_metadata

    histogram_pruning_bins = Conf.histogram_bins
    histogram_pruning_threshold = Conf.histogram_threshold
    x_value_cap = Conf.velocity_cap

    print(y_regressed_dims)
    main(x_feats, u_feats, y_regressed_dims, model_ground_effect, simulation_options, ds_name,
         x_value_cap, histogram_pruning_bins, histogram_pruning_threshold,
         model_name=model_name, epochs=epochs, batch_size=batch_size, hidden_layers=hidden_layers,
         hidden_size=hidden_size, lr=lr,plot=plot)
