""" Executable script to train a custom Gaussian Process on recorded flight data.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
sys.path.append('/home/ryan/raigor/train_ws/src')
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

from model_fit_test_tum.gp_common import read_dataset
from model_fit_test_tum.mlp_common import RawDataset, NormalizedMlp, MlpDataset, NomialModel
from config.configuration_parameters import ModelFitConfig as Conf
from src.utils.utils import safe_mkdir_recursive
from src.utils.utils import get_model_dir_and_file

def data_pre_process(data):
    """
    It is used for calculate the nomial state given the data
    """
    nomial_model = NomialModel(data,data.shape[0],3)   #  3 is [x, y, yaw]
    nomial_state = nomial_model.calc_nomial()
    df_val_pd = pd.DataFrame(nomial_state, columns=["nomial_x", "nomial_y", "nomial_yaw"])
    df_new = pd.concat([data, df_val_pd], axis=1)
    return df_new

def main(x_features, u_features, reg_y_dims, model_ground_effect, quad_sim_options, dataset_name,
         x_cap, hist_bins, hist_thresh,
         model_name="simple_sim_mlp", epochs=100, batch_size=64, hidden_layers=4, hidden_size=64, lr=1e-4,plot=False):
    """
    reg_y_dims: use to determine y dimension
    """
    
    print(f'epho:{epochs},hiddensize:{hidden_size},hidden_layer:{hidden_layers}')
    # #### Get git commit hash for saving the model #### #
    git_version = ''
    try:
        git_version = subprocess.check_output(['git', 'describe', '--always']).strip().decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(e.returncode, e.output)
    print("The model will be saved using hash: %s" % git_version)

    # #### Get data source path #### #
    gp_name_dict = {"git": git_version, "model_name": model_name, "params": quad_sim_options}
    save_file_path, save_file_name = get_model_dir_and_file(gp_name_dict)
    save_file_path = os.path.join("/home/ryan/train_ws/results/model_fitting/", str(gp_name_dict["git"]), str(gp_name_dict["model_name"]))
    # print(save_file_path)
    safe_mkdir_recursive(save_file_path)
    print(f'{gp_name_dict},{save_file_path},{save_file_name}')
    
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
    # df_nomial = data_pre_process(df_val_pre) # just need to know it is used to add one more property into dataset: nominal_state    in order to calculate residuals between nominal_state and actual output_state 
    raw_dataset_train = RawDataset(df_val_pre) # Turn raw data into pandas datatype
    dataset_train = MlpDataset(raw_dataset_train) # Init MLP dataset
    x_mean, x_std, y_mean, y_std = dataset_train.stats() # get info to help normalize data in MLP
    print(x_mean, x_std, y_mean, y_std)
    input_dims = 6  # MLP input dimension
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
            # ## test forward
            # y_forward_test = model.forward_test(x)
            # y_forward = model.forward(x)
            # print(f'y_forward - y_forward_test: {y_forward - y_forward_test}')
            # print("y_pred - y_forward)",y_pred - y_forward)
            # print(y_pred)
            loss = torch.square(y-y_pred).mean()
            # print(loss)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

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
    torch.save(save_dict, os.path.join(save_file_path, f'{save_file_name}.pt'))
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
