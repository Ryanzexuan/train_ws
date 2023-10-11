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


import os
import time
import argparse
import subprocess
import numpy as np

import torch
from torch.utils.data import DataLoader
import ml_casadi.torch as mc

from tqdm import tqdm
import sys
sys.path.append('/home/ryan/train_ws/src')
from model_fit.gp_common import GPDataset, read_dataset
from model_fit.mlp_common import RawDataset, NormalizedMlp, MlpDataset
from config.configuration_parameters import ModelFitConfig as Conf
from src.utils.utils import safe_mkdir_recursive, load_pickled_models
from src.utils.utils import get_model_dir_and_file


def main(x_features, u_features, reg_y_dims, model_ground_effect, quad_sim_options, dataset_name,
         x_cap, hist_bins, hist_thresh,
         model_name="model", epochs=100, batch_size=64, hidden_layers=1, hidden_size=32, plot=False):

    # Get git commit hash for saving the model
    git_version = ''
    try:
        git_version = subprocess.check_output(['git', 'describe', '--always']).strip().decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(e.returncode, e.output)
    print("The model will be saved using hash: %s" % git_version)

    gp_name_dict = {"git": git_version, "model_name": model_name, "params": quad_sim_options}
    save_file_path, save_file_name = get_model_dir_and_file(gp_name_dict)
    safe_mkdir_recursive(save_file_path)
    print(f'{gp_name_dict},{save_file_path},{save_file_name}')
    
    # #### DATASET LOADING #### #
    if isinstance(dataset_name, str):
        df_train = read_dataset(dataset_name, True, quad_sim_options)
        # gp_dataset_train = GPDataset(df_train, x_features, u_features, reg_y_dims,
        #                              cap=x_cap, n_bins=hist_bins, thresh=hist_thresh)
        gp_dataset_val = None
        try:
            df_val = read_dataset(dataset_name, False, quad_sim_options)
            # gp_dataset_val = GPDataset(df_val, x_features, u_features, reg_y_dims,
            #                            cap=x_cap, n_bins=hist_bins, thresh=hist_thresh)
        except:
            print('Could not find test dataset.')
    else:
        raise TypeError("dataset_name must be a string.")
    # invalid = np.where( == 0)
    # print(df_val['vel_x'])
    raw_dataset_train = RawDataset(df_val)
    dataset_train = MlpDataset(raw_dataset_train)
    x_mean, x_std, y_mean, y_std = dataset_train.stats() # SMU seems useless
    
    input_dims = 2   # inpute dimension +(~ if groud_effect else 0) or +len(x_features)
    data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    mlp_model = mc.nn.MultiLayerPerceptron(input_dims, hidden_size, len(reg_y_dims), hidden_layers, 'Tanh')
    model = NormalizedMlp(mlp_model,torch.tensor(x_mean).float(), torch.tensor(x_std).float(), torch.tensor(y_mean).float(), torch.tensor(y_std).float())

    cuda_name = 'cuda:0'
    if torch.cuda.is_available():
        model = model.to(cuda_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
            optimizer.zero_grad()
            y_pred = model(x)
            loss = torch.square(y-y_pred).mean()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss_mean = np.mean(losses)
        train_loss_std = np.std(losses)
        loss_info = train_loss_mean
        loss_infos.append(loss_info)
        bar.set_description(f'Train Loss: {train_loss_mean:.6f}, Val Loss {loss_info:.6f}')
        bar.refresh()

    save_dict = {
            'state_dict': model.state_dict(),
            'input_size': input_dims,
            'hidden_size': hidden_size,
            'output_size': len(reg_y_dims),
            'hidden_layers': hidden_layers
        }
    torch.save(save_dict, os.path.join(save_file_path, f'{save_file_name}.pt'))


    if 1:
        import matplotlib.pyplot as plt
        # plt.plot(loss_infos)
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default="1000",
                        help="Number of epochs to train the model.")

    parser.add_argument("--batch_size", type=int, default="64",
                        help="Batch Size.")

    parser.add_argument("--hidden_layers", type=int, default="1",
                        help="Number of hidden layers.")

    parser.add_argument("--hidden_size", type=int, default="32",
                        help="Size of hidden layers.")

    parser.add_argument("--model_name", type=str, default="",
                        help="Name assigned to the trained model.")

    parser.add_argument('--x', nargs='+', type=int, default=[2],
                        help='Regression X variables. Must be a list of integers between 0 and 12. Velocities xyz '
                             'correspond to indices 7, 8, 9.')

    parser.add_argument('--u', action="store_true",
                        help='Use the control as input to the model.')

    parser.add_argument("--y", nargs='+', type=int, default=[2],
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
    plot = input_arguments.plot

    ds_name = Conf.ds_name
    simulation_options = Conf.ds_metadata

    histogram_pruning_bins = Conf.histogram_bins
    histogram_pruning_threshold = Conf.histogram_threshold
    x_value_cap = Conf.velocity_cap

    main(x_feats, u_feats, y_regressed_dims, model_ground_effect, simulation_options, ds_name,
         x_value_cap, histogram_pruning_bins, histogram_pruning_threshold,
         model_name=model_name, epochs=epochs, batch_size=batch_size, hidden_layers=hidden_layers,
         hidden_size=hidden_size, plot=plot)
