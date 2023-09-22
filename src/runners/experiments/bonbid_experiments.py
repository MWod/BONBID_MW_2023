### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib

### External Imports ###
import numpy as np
import torch as tc
from torch.utils.tensorboard import SummaryWriter

### Internal Imports ###
from paths import hpc_paths as p
from input_output import volumetric as io_vol
from helpers import utils as u
from helpers import cost_functions as cf
from datasets import bonbid_dataset as ds
from augmentation import torchio as aug_tio
from networks import runet

########################

# Only final experiment left for clarity

def get_experiment_fs_7():
    """
    Check RUNet with elastic augmentation (default config)
    """
    dataset_path = p.parsed_bonbid_path / "ElasticShape_256_256_128"
    validation_dataset_path = p.parsed_bonbid_path / "Shape_256_256_128"
    training_csv_path = p.parsed_bonbid_path / "elasticaug_training_dataset.csv"
    validation_csv_path = p.parsed_bonbid_path / "validation_dataset.csv"

    loading_params = io_vol.default_volumetric_pytorch_load_params
    
    geometric_torchio_transforms = aug_tio.bonbid_geometric_transforms()
    adc_intensity_torchio_transforms = aug_tio.bonbid_adc_transforms()
    zadc_intensity_torchio_transforms = aug_tio.bonbid_zadc_transforms()
    
    return_load_time = True
    iteration_size = 100
    training_dataset = ds.BonbidDataset(dataset_path, training_csv_path, iteration_size=iteration_size, loading_params=loading_params, return_load_time=return_load_time,
                                        geometric_torchio_transforms=geometric_torchio_transforms,
                                        adc_intensity_torchio_transforms=adc_intensity_torchio_transforms,
                                        zadc_intensity_torchio_transforms=zadc_intensity_torchio_transforms)
    validation_dataset = ds.BonbidDataset(validation_dataset_path, validation_csv_path, iteration_size=-1, loading_params=loading_params, return_load_time=return_load_time)
    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    num_workers = 16
    outer_batch_size = 1
    training_dataloader = tc.utils.data.DataLoader(training_dataset, batch_size=outer_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=outer_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = "cuda:0"
    config = runet.default_config()
    config['input_channels'] = [2, 16, 64, 128, 256]
    config['use_sigmoid'] = False
    model = runet.RUNet(**config).to(device)
    
    ### Parameters ###
    experiment_name = "BONBID_HPC_RUNET_FS7"
    inner_batch_size = 1
    effective_batch_multiplier = 16
    num_iterations = 3001
    learning_rate = 0.001
    save_step = 50
    to_load_checkpoint_path = p.checkpoints_path / "BONBID_HPC_RUNET_FS7" / "Iteration_1500"
    number_of_images_to_log = 3
    lr_decay = 0.999
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True}
    max_gradient_value = 5
    max_gradient_norm = 30
    optimizer_weight_decay = 0.005
    dtype = tc.float32
    log_time = True
    use_amp = False
    non_blocking = True

    ### Parse Parameters ###
    training_params = dict()
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['model'] = model
    training_params['device'] = device
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['num_iterations'] = num_iterations
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['inner_batch_size'] = inner_batch_size
    training_params['effective_batch_multiplier'] = effective_batch_multiplier
    training_params['log_time'] = log_time
    training_params['non_blocking'] = non_blocking

    ### Cost functions and params
    training_params['objective_function'] = objective_function
    training_params['objective_function_params'] = objective_function_params
    training_params['max_gradient_value'] = max_gradient_value
    training_params['max_gradient_norm'] = max_gradient_norm
    training_params['dtype'] = dtype
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['use_amp'] = use_amp
    training_params['use_sigmoid'] = False

    ########################################
    return training_params



