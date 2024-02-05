"""
This works in tandem with the mlp-mnistDataset-torch notebook based on this (https://www.kaggle.com/code/mishra1993/pytorch-multi-layer-perceptron-mnist/notebook) tutorial
"""

import torch

import numpy as np
import pandas as pd

from typing import Iterator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler, SubsetRandomSampler


class Dataset(object):
    """An abstract class representing a dataset. All other datasets should be a subclass of this class. All subclasses should override 
    ``__len__``, that provides the size of the dataset, and 
    ``__getitem__``, supporting integer indexing in range from 0 to len(self) exlusive.
    """

    # a method that allows its instances (anyting we pass in) to use the [ ] (indexer) operations. thus as as a list of that type passed in
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
    
    def select_sample_method(self):
        raise NotImplementedError("Subclasses must implement my_abstract_method")

    def prepare_data_loader(self):
        raise NotImplementedError("Subclasses must implement my_abstract_method")


    
class TrainMNIST(Dataset):
    
    # initial processes like reading a csv file, assigning transforms
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    # return the of input data
    def __len__(self):
        return len(self.data)
    
    # return the data and label at orbitary index (orbitary index?)
    def __getitem__(self, index):
        # load image as ndarray type (Height, Width, Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def select_sample_method(self, sample_method: str, sampler_idx: int):

        if sample_method == 'sequential':
            print("Sampling Method -- ",sample_method)
            sampler = SequentialSampler(sampler_idx)
            return sampler
        elif sample_method == 'subset':
            print("Sampling Method -- ", sample_method)
            sampler = SubsetRandomSampler(sampler_idx)
            return sampler
        elif sample_method == 'random':
            print("Sampling Method -- ", sample_method)
            sampler = RandomSampler(sampler_idx)
            return sampler
        
    
    def prepare_data_loader(self, batch_size: int, sample_method: str, sampler_idx: int, num_workers: int):
        sample_order = self.select_sample_method(sample_method, sampler_idx)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sample_order, num_workers=num_workers)
        return loader
    
   


class TestMNIST(Dataset):
    """Same as TrainMIST, except we don't retrun the label here"""
    

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height, Width, Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
        image = self.data.iloc[index, 0:].values.astype(np.uint8).reshape((28, 28, 1))

        if self.transform is not None:
            image = self.transform(image)

        return image
    
    def prepare_data_loader(self, batch_size: int, num_workers: int):
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=num_workers)
        return loader
