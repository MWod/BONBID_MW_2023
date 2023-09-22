# BONBID_MW_2023
Contribution to the BONBID Challenge (MICCAI 2023) by Marek Wodzinski.

The challenge website: [Link](https://bonbid-hie2023.grand-challenge.org/)

Here you can see the full source code used to train / test the proposed solution.

Only the final experiment is left (the one used for the final Docker submission).

* In order to reproduce the experiment you should:
    * Download the BONBID dataset [Link](https://bonbid-hie2023.grand-challenge.org/data/)
    * Update the [hpc_paths.py](./src/paths/hpc_paths.py) and [paths.py](./src/paths/paths.py) files.
    * Run the [parse_bonbid.py](./src/parsers/parse_bonbid.py)
    * Run the [run_aug_bonbid.py](./src/parsers/run_aug_bonbid.py)
    * Run the training using [run_segmentation_trainer.py](./src/runners/run_segmentation_trainer.py)
    * And finally use the trained model for inference using [inference.py](./src/inference/inference_bonbid.py)

The network was trained using HPC infrastructure (PLGRID). Therefore the .slurm scripts are omitted for clarity.

Please cite the BONBID publication (TODO) if you found the source code useful.
