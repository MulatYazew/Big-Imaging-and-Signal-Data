import pandas as pd
import torch

class Arrhythmia_Dataset(torch.utils.data.Dataset):

    def __init__(self, csv):
        # read the csv file
        self.df = pd.read_csv(csv, sep=',')
        # save cols
        self.output_cols = ['class']
        # get columns of dataframe
        self.input_cols = list(set(self.df.columns) - set(self.output_cols))


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
        cur_sample_x = torch.tensor(cur_sample_x.tolist()).unsqueeze(0)
        cur_sample_y = torch.tensor(cur_sample_y.tolist()).squeeze()
        # return values
        return cur_sample_x, cur_sample_y
	