from nnunetv2_mod.nnunetv2.paths import nnUNet_preprocessed
from pathlib import Path

import numpy as np
import yaml


def split(nnunet_dataset_name, png_dataset_path: Path):
    '''
    Split nnunet dataset into 2 parts, return paths. Used to get the rejection threshold for synthetic data.
    '''
    case_name_to_files = yaml.safe_load(open(Path(nnUNet_preprocessed) / nnunet_dataset_name / 'case_name_to_files.yaml', 'r'))
    case_names = list(case_name_to_files.keys())

    rng = np.random.default_rng(42)
    rng.shuffle(case_names)

    part0, part1 = np.array_split(case_names, 2)
    part0_filenames = [f for case_name in part0
                         for f in case_name_to_files[case_name]]
    part1_filenames = [f for case_name in part1
                         for f in case_name_to_files[case_name]]

    part0_paths = [png_dataset_path / (f + '_0000.png') for f in part0_filenames]
    part1_paths = [png_dataset_path / (f + '_0000.png') for f in part1_filenames]

    return part0_paths, part1_paths