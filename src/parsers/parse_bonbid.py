### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pathlib
import shutil
from string import Template

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk

### Internal Imports ###
from paths import paths as p

from input_output import volumetric as io
from visualization import volumetric as vis
from augmentation import volumetric as aug_vol
from input_output import utils_io as uio
from helpers import utils as u
from preprocessing import preprocessing_volumetric as pre_vol

from registration import iterative_registration as reg
from registration import reg_utils
from registration import cost_functions as cf


def parse_bonbid():
    data_path = p.raw_bonbid_path
    output_path = p.parsed_bonbid_path / "Shape_256_256_128"
    output_csv_path = p.parsed_bonbid_path / "dataset.csv"

    all_cases = os.listdir(data_path / "1ADC_ss")
    available_ids = [(str(item.split("_")[1]).split("-")[0]) for item in all_cases]

    output_size = (256, 256, 128)
    device = "cuda:0"
    dataframe = []
    for idx, current_id in enumerate(available_ids):
        print()
        print(f"Current case: {idx+1}/{len(available_ids)}")
        adc_template = Template('MGHNICU_$id-VISIT_01-ADC_ss.mha')
        zadc_template = Template('Zmap_MGHNICU_$id-VISIT_01-ADC_smooth2mm_clipped10.mha')
        gt_template = Template('MGHNICU_$id-VISIT_01_lesion.mha')

        adc_path = data_path / "1ADC_ss" / adc_template.substitute(id=current_id)
        zadc_path = data_path / "2Z_ADC" / zadc_template.substitute(id=current_id)
        gt_path = data_path / "3LABEL" / gt_template.substitute(id=current_id)

        temp_volume = sitk.ReadImage(adc_path)
        temp_volume = sitk.GetArrayFromImage(temp_volume).swapaxes(0, 1).swapaxes(1, 2)

        adc, zadc, gt, spacing = parse_case(adc_path, zadc_path, gt_path, output_size, device)
        shape = temp_volume.shape
        new_spacing = tuple(np.array(spacing) * np.array(shape) / np.array(output_size))
        print(f"Spacing: {spacing}")
        print(f"New Spacing: {new_spacing}")

        output_folder = output_path / current_id
        adc_output_path = output_folder / "ADC.mha"
        zadc_output_path = output_folder / 'ZADC.mha'
        gt_output_path = output_folder / 'GT.nrrd'

        adc_temp_path = pathlib.Path(current_id) / "ADC.mha"
        zadc_temp_path = pathlib.Path(current_id) / 'ZADC.mha'
        gt_temp_path = pathlib.Path(current_id) / 'GT.nrrd'
        dataframe.append((adc_temp_path, zadc_temp_path, gt_temp_path))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        to_save = sitk.GetImageFromArray(adc.swapaxes(2, 1).swapaxes(1, 0))
        to_save.SetSpacing(new_spacing)
        sitk.WriteImage(to_save, str(adc_output_path))

        to_save = sitk.GetImageFromArray(zadc.swapaxes(2, 1).swapaxes(1, 0))
        to_save.SetSpacing(new_spacing)
        sitk.WriteImage(to_save, str(zadc_output_path))

        to_save = sitk.GetImageFromArray(gt.swapaxes(2, 1).swapaxes(1, 0))
        to_save.SetSpacing(new_spacing)
        sitk.WriteImage(to_save, str(gt_output_path), useCompression=True)

    dataframe = pd.DataFrame(dataframe, columns=['ADC Path', 'ZADC Path', 'GT Path'])
    dataframe.to_csv(output_csv_path, index=False)



def parse_case(adc_path, zadc_path, gt_path, output_size, device):
    adc_volume = sitk.ReadImage(adc_path)
    zadc_volume = sitk.ReadImage(zadc_path)
    gt_volume = sitk.ReadImage(gt_path)
    spacing = adc_volume.GetSpacing()

    adc_volume = sitk.GetArrayFromImage(adc_volume).swapaxes(0, 1).swapaxes(1, 2)
    zadc_volume = sitk.GetArrayFromImage(zadc_volume).swapaxes(0, 1).swapaxes(1, 2)
    gt_volume = sitk.GetArrayFromImage(gt_volume).swapaxes(0, 1).swapaxes(1, 2)

    print(f"ADC shape: {adc_volume.shape}")
    print(f"ZADC shape: {zadc_volume.shape}")
    print(f"GT shape: {gt_volume.shape}")
    print(f"Spacing: {spacing}")

    adc_tc = tc.from_numpy(adc_volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    zadc_tc = tc.from_numpy(zadc_volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    gt_tc = tc.from_numpy(gt_volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    print(f"ADC TC shape: {adc_tc.shape}")
    print(f"ZADC TC shape: {zadc_tc.shape}")
    print(f"GT TC shape: {gt_tc.shape}")

    resampled_adc_tc = pre_vol.resample_tensor(adc_tc, (1, 1, *output_size), mode='bilinear')
    resampled_zadc_tc = pre_vol.resample_tensor(zadc_tc, (1, 1, *output_size), mode='bilinear')
    resampled_gt_tc = pre_vol.resample_tensor(gt_tc, (1, 1, *output_size), mode='bilinear')

    print(f"Resampled ADC TC shape: {resampled_adc_tc.shape}")
    print(f"Resampled ZADC TC shape: {resampled_zadc_tc.shape}")
    print(f"Resampled GT TC shape: {resampled_gt_tc.shape}")

    adc_tc = resampled_adc_tc[0, 0, :, :, :].detach().cpu().numpy()
    zadc_tc = resampled_zadc_tc[0, 0, :, :, :].detach().cpu().numpy()
    gt_tc = (resampled_gt_tc[0, 0, :, :, :].detach().cpu().numpy() > 0.5).astype(np.uint8)

    return adc_tc, zadc_tc, gt_tc, spacing


def split_dataframe(input_csv_path, training_csv_path, validation_csv_path, split_ratio = 0.8, seed=1234):
    dataframe = pd.read_csv(input_csv_path)
    dataframe = dataframe.sample(frac=1, random_state=seed)
    training_dataframe = dataframe[:int(split_ratio*len(dataframe))]
    validation_dataframe = dataframe[int(split_ratio*len(dataframe)):]
    print(f"Dataset size: {len(dataframe)}")
    print(f"Training dataset size: {len(training_dataframe)}")
    print(f"Validation dataset size: {len(validation_dataframe)}")
    if not os.path.isdir(os.path.dirname(training_csv_path)):
        os.makedirs(os.path.dirname(training_csv_path))
    if not os.path.isdir(os.path.dirname(validation_csv_path)):
        os.makedirs(os.path.dirname(validation_csv_path))
    training_dataframe.to_csv(training_csv_path)
    validation_dataframe.to_csv(validation_csv_path)


def run():
    parse_bonbid()
    pass


if __name__ == "__main__":
    run()