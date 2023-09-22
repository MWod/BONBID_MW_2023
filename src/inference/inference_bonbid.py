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
import SimpleITK as sitk

### Internal Imports ###
from paths import paths as p
from preprocessing import preprocessing_volumetric as pre_vol
from networks import runet
from helpers import utils as u

########################


def default_single_step_inference_params(checkpoint_path):
    config = {}
    device = "cuda:0"
    network_config = runet.default_config()
    network_config['input_channels'] = [2, 16, 64, 128, 256]
    model = runet.RUNet(**network_config).to(device)
    checkpoint = tc.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)
    echo = True
    config['device'] = device
    config['output_size'] =  (256, 256, 128)
    config['model'] = model
    config['echo'] = echo
    return config

def single_step_inference(adc : np.ndarray, zadc : np.ndarray, **params) -> np.ndarray:
    device = params['device']
    model = params['model']
    output_size = params['output_size']
    echo = params['echo']
    with tc.set_grad_enabled(False):
        adc_tc = tc.from_numpy(adc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        adc_tc = u.normalize(adc_tc)
        zadc_tc = tc.from_numpy(zadc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        zadc_tc = (zadc_tc + 10) / 20 
        volume_tc = tc.cat((adc_tc, zadc_tc), dim=1)
        print(f"Input shape: {volume_tc.shape}") if echo else None
        original_shape = volume_tc.shape
        volume_tc = pre_vol.resample_tensor(volume_tc, (1, 1, *output_size), mode='bilinear')
        print(f"Resampled input shape: {volume_tc.shape}") if echo else None
        output_tc = model(volume_tc)
        print(f"Output shape: {output_tc.shape}") if echo else None
        output_tc = pre_vol.resample_tensor(output_tc, original_shape, mode='bilinear')
        print(f"Resampled output shape: {output_tc.shape}") if echo else None
        output = (output_tc[0, 0, :, :, :] > 0.5).detach().cpu().numpy()
    return output

def run_inference(adc_path, zadc_path, inference_method, inference_method_params, ground_truth_path=None, output_path=None):
    echo = inference_method_params['echo']
    adc = sitk.ReadImage(adc_path)
    zadc = sitk.ReadImage(zadc_path)
    spacing = adc.GetSpacing()
    direction = adc.GetDirection()
    origin = adc.GetOrigin()
    adc = sitk.GetArrayFromImage(adc).swapaxes(0, 1).swapaxes(1, 2)
    zadc = sitk.GetArrayFromImage(zadc).swapaxes(0, 1).swapaxes(1, 2)

    if ground_truth_path is not None:
        ground_truth = sitk.ReadImage(ground_truth_path)
        ground_truth = sitk.GetArrayFromImage(ground_truth).swapaxes(0, 1).swapaxes(1, 2)
    output = inference_method(adc, zadc, **inference_method_params)
    if output_path is not None:
        to_save = sitk.GetImageFromArray(output.swapaxes(2, 1).swapaxes(1, 0))
        to_save.SetSpacing(spacing)
        to_save.SetDirection(direction)
        to_save.SetOrigin(origin)
        sitk.WriteImage(to_save, str(output_path), useCompression=True)
    return output