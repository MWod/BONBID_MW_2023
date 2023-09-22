### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pathlib
import shutil
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk
import torchio

### Internal Imports ###
from paths import hpc_paths as p

from input_output import volumetric as io
from visualization import volumetric as vis
from augmentation import volumetric as aug_vol
from input_output import utils_io as uio
from helpers import utils as u
from preprocessing import preprocessing_volumetric as pre_vol


def augment_by_random_elastic(input_data_path, output_data_path, input_csv_path, output_csv_path):
    ### Params ###
    loading_params = io.default_volumetric_pytorch_load_params
    saving_params = io.default_volumetric_save_params
    loader = io.VolumetricLoader(**loading_params)
    image_saver = io.VolumetricSaver(**saving_params | {'use_compression' : False})
    gt_saver = io.VolumetricSaver(**saving_params)

    ###################

    ### Params ###
    cases_to_generate = 10000
    min_control_points = 5
    max_control_points = 20 #15

    ### Augmentation ###
    input_dataframe = pd.read_csv(input_csv_path)
    num_cases = len(input_dataframe)
    output_data = []
    for idx in range(cases_to_generate):
        case_id = random.randint(0, num_cases - 1)
        row = input_dataframe.loc[case_id]
        input_adc_path = row['ADC Path']
        input_zadc_path = row['ZADC Path']
        input_gt_path = row['GT Path']

        loader.load(input_data_path / input_adc_path)
        input_adc = loader.volume
        spacing = loader.spacing
        loader.load(input_data_path / input_zadc_path)
        input_zadc = loader.volume
        loader.load(input_data_path / input_gt_path)
        input_gt = loader.volume

        control_points = random.randint(min_control_points, max_control_points)
        displacement = 45 * 5 / control_points
        transform = torchio.RandomElasticDeformation(num_control_points=control_points, max_displacement=displacement)
        subject = torchio.Subject(
            adc=torchio.ScalarImage(tensor=input_adc),
            zadc=torchio.ScalarImage(tensor=input_zadc),
            label=torchio.LabelMap(tensor=input_gt))
        result = transform(subject)
        warped_adc = result['adc'].data
        warped_zadc = result['zadc'].data
        warped_gt = result['label'].data

        save_adc_path = pathlib.Path(f"E{idx}") / "ADC.mha"
        save_zadc_path = pathlib.Path(f"E{idx}") / "ZADC.mha"
        save_gt_path = pathlib.Path(f"E{idx}") / "GT.nrrd"

        create_folder(output_data_path / save_adc_path)
        create_folder(output_data_path / save_zadc_path)
        create_folder(output_data_path / save_gt_path)
        save_image(warped_adc[0].detach().cpu().numpy(), list(spacing.numpy().astype(np.float64)), output_data_path / save_adc_path)
        save_image(warped_zadc[0].detach().cpu().numpy(), list(spacing.numpy().astype(np.float64)), output_data_path / save_zadc_path)
        gt_saver.save(warped_gt, spacing, output_data_path / save_gt_path)

        to_append = (save_adc_path, save_zadc_path, save_gt_path)
        output_data.append(to_append)

    ### Copy Original ###
    for idx, row in input_dataframe.iterrows():
        adc_path = row['ADC Path']
        zadc_path = row['ZADC Path']
        gt_path = row['GT Path']
        copy_file(input_data_path / adc_path, output_data_path / adc_path)
        copy_file(input_data_path / zadc_path, output_data_path / zadc_path)
        copy_file(input_data_path / gt_path, output_data_path / gt_path)
        to_append = (adc_path, zadc_path, gt_path)
        output_data.append(to_append)

    ### Create Dataframe ###
    output_dataframe = pd.DataFrame(output_data, columns=['ADC Path', 'ZADC Path', 'GT Path'])
    output_dataframe.to_csv(output_csv_path, index=False)


def save_image(volume, spacing, save_path, origin=None, direction=None):
    image  = sitk.GetImageFromArray(volume.swapaxes(2, 1).swapaxes(1, 0))
    image.SetSpacing(spacing)
    if origin is not None:
        image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    sitk.WriteImage(image, str(save_path))

def copy_file(input_path, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    shutil.copy(input_path, output_path)

def create_folder(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))    

def run():
    input_data_path = p.parsed_bonbid_path / "Shape_256_256_128"
    output_data_path = p.parsed_bonbid_path / "ElasticShape_V2_256_256_128"
    input_csv_path = p.parsed_bonbid_path / "training_dataset.csv"
    output_csv_path = p.parsed_bonbid_path / "elasticaug_v2_training_dataset.csv"
    augment_by_random_elastic(input_data_path, output_data_path, input_csv_path, output_csv_path)

if __name__ == "__main__":
    run()

