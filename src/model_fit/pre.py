import sys
import numpy as np
sys.path.append('/home/ryan/train_ws/src')
from model_fit.mlp_common import RawDataset, NormalizedMlp, MlpDataset, NomialModel
import pandas as pd



def data_pre_process(data):
    """
    It is used for calculate the nomial state given the data
    """
    nomial_model = NomialModel(data,data.shape[0],3)
    nomial_state = nomial_model.calc_nomial()
    # print(nomial_state.shape[1])
    df_val_pd = pd.DataFrame(nomial_state, columns=["nomial_x", "nomial_y", "nomial_yaw"])
    df_val111 = pd.concat([data, df_val_pd], axis=1)
    return df_val111


def main():
    rec_file = "/home/ryan/train_ws/data/simplified_sim_dataset/train/dataset_002.csv"
    ds = pd.read_csv(rec_file)
    df_val = data_pre_process(ds)
    print(df_val['x_position_output'] - df_val['nomial_x'])
    # df_val_pd = pd.DataFrame(df_val, columns=["nomial_x", "nomial_y", "nomial_yaw"])
    # df_val111 = pd.concat([ds, df_val_pd], axis=1)
    # print(df_val111.shape[0])
    # print(df_val111.shape[1])
    # print(df_val[5].shape[1])

if __name__ == '__main__':
    main()