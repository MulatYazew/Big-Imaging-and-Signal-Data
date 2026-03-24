import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoilDataset(torch.utils.data.Dataset):

    def __init__(self, csv, src_prefix='spc.'):
        # read the csv file
        self.df = pd.read_csv(csv, sep=',')
        # self.df = self.df.dropna(axis=0)
        # save cols
        self.output_cols = ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'] # 'GPS_LAT','GPS_LONG'
        # get columns of dataframe
        self.input_cols = [col for col in self.df.columns if col.startswith(src_prefix)]
        # get x values
        x = np.array([float(col[len(src_prefix):]) for col in self.input_cols])
        # sort x points in increasing order
        pos = np.argsort(x)
        self.input_cols = [self.input_cols[cur_pos] for cur_pos in pos]


    def __len__(self):
        # here i will return the number of samples in the dataset
        return len(self.df)


    def __getitem__(self, idx):
        # here i will load the file in position idx
        cur_sample = self.df.iloc[idx]
        # split in input / ground-truth
        cur_sample_x = cur_sample[self.input_cols]
        cur_sample_y = cur_sample[self.output_cols]
        # convert to torch format
        cur_sample_x = torch.tensor(cur_sample_x.tolist())
        cur_sample_y = torch.tensor(cur_sample_y.tolist())
        # return values
        return cur_sample_x, cur_sample_y
    


