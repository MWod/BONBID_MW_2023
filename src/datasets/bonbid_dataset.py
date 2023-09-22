### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Callable
import time
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import torchio as tio

### Internal Imports ###

from augmentation import aug
from input_output import volumetric as v
from helpers import utils as u

########################


class BonbidDataset(tc.utils.data.Dataset):
    """
    TODO
    """
    def __init__(
        self,
        data_path : Union[str, pathlib.Path],
        csv_path : Union[str, pathlib.Path],
        iteration_size : int = -1,
        loading_params : dict = {},
        return_load_time : bool=False,
        return_paths : bool=False, 
        geometric_torchio_transforms = None,
        adc_intensity_torchio_transforms = None,
        zadc_intensity_torchio_transforms = None):
        """
        TODO
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(self.csv_path)
        self.iteration_size = iteration_size
        self.geometric_torchio_transforms = geometric_torchio_transforms
        self.adc_torchio_transforms = adc_intensity_torchio_transforms 
        self.zadc_torchio_transforms = zadc_intensity_torchio_transforms
        self.loading_params = loading_params
        self.return_load_time = return_load_time
        self.return_paths = return_paths
        if self.iteration_size > len(self.dataframe):
            self.dataframe = self.dataframe.sample(n=self.iteration_size, replace=True).reset_index(drop=True)

    def __len__(self):
        if self.iteration_size < 0:
            return len(self.dataframe)
        else:
            return self.iteration_size
        
    def shuffle(self):
        if self.iteration_size > 0:
            self.dataframe = self.dataframe.sample(n=len(self.dataframe), replace=False).reset_index(drop=True)

    def __getitem__(self, idx):
        current_case = self.dataframe.loc[idx]
        adc_path = self.data_path / current_case['ADC Path']
        zadc_path = self.data_path / current_case['ZADC Path']
        ground_truth_path = self.data_path / current_case['GT Path']

        b_t = time.time()
        adc_loader = v.VolumetricLoader(**self.loading_params).load(adc_path)
        adc, spacing, input_metadata = adc_loader.volume, adc_loader.spacing, adc_loader.metadata
        adc = (adc - tc.min(adc)) / (tc.max(adc) - tc.min(adc))

        zadc_loader = v.VolumetricLoader(**self.loading_params).load(zadc_path)
        zadc, _, _= zadc_loader.volume, zadc_loader.spacing, zadc_loader.metadata
        zadc = (zadc + 10) / 20 #[-10, 10 -> [0, 1]]

        gt_loader = v.VolumetricLoader(**self.loading_params).load(ground_truth_path)
        ground_truth = gt_loader.volume

        output = (adc, zadc, ground_truth)

        e_t = time.time()
        loading_time = e_t - b_t

        cons = tc.cat((output[0], output[1]), dim=0)
        output = (cons, output[2])
            
        if self.geometric_torchio_transforms is not None:
            subject = tio.Subject(
            input = tio.ScalarImage(tensor=output[0]),
            label = tio.LabelMap(tensor=output[1]))
            result = self.geometric_torchio_transforms(subject)
            transformed_input = result['input'].data
            transformed_gt = result['label'].data
            output = (transformed_input, transformed_gt)

        if self.adc_torchio_transforms is not None:
            subject = tio.Subject(
            input = tio.ScalarImage(tensor=output[0][0:1]))
            result = self.adc_torchio_transforms(subject)
            transformed_input = result['input'].data
            output[0][0:1] = transformed_input

        if self.zadc_torchio_transforms is not None:
            subject = tio.Subject(
            input = tio.ScalarImage(tensor=output[0][1:2]))
            result = self.zadc_torchio_transforms(subject)
            transformed_input = result['input'].data
            output[0][1:2] = transformed_input

        e_t = time.time()    
        augmentation_time = e_t - b_t
        total_time = (loading_time, augmentation_time)

        if self.return_load_time:
            return *output, spacing, total_time
        if self.return_paths:
            return *output, spacing, dict(**current_case, **input_metadata)
        else:
            return *output, spacing