from pathlib import Path

import numpy as np


def split(png_dataset_path: Path):
    '''
    Split nnunet dataset into 2 parts, return paths. Used to get the rejection threshold for synthetic data.
    '''

    # png have names like cervix_0_0000.png, case name is 'cervix_0'
    case_name_to_files = {}
    for png_path in png_dataset_path.glob('*.png'):
        case_name = png_path.stem[:png_path.stem.rfind('_')]
        if case_name not in case_name_to_files:
            case_name_to_files[case_name] = []
        case_name_to_files[case_name].append(png_path)

    case_names = list(case_name_to_files.keys())

    rng = np.random.default_rng(42)
    rng.shuffle(case_names)

    part0, part1 = np.array_split(case_names, 2)

    part0_paths = []
    for case_name in part0:
        part0_paths.extend(case_name_to_files[case_name])
    part1_paths = []
    for case_name in part1:
        part1_paths.extend(case_name_to_files[case_name])

    return part0_paths, part1_paths