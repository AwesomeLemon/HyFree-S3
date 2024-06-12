'''
Used to convert the cervix dataset to nnUnet format
'''
import copy
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from typing import Tuple

import pandas as pd
import yaml
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles, join, isdir, isfile, \
    maybe_mkdir_p, load_json
from nnunetv2.utilities.dataset_name_id_conversion import find_candidate_datasets

from dataset.preprocess_new.cervix.cervix_dicom_utils import _create_dataset_niigz_from_dicoms
from nnunetv2.dataset_conversion.convert_MSD_dataset import split_4d_nifti
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
from utils.general import setup_logging
from find_scanners import extract_and_store_scanner_info
from collections import defaultdict
def create_base_dataset_from_dicoms(data_path, save_path, patient_csv, target_labels, img_mode, processes, dataset_descriptor):
    """
    Does most of the preprocessing but no splitting into folds and therefore doesn't create dataset.json

    Args:
        data_path: folder path with all the patient folders
        save_path: folder path to save everything
        patient_csv: csv with patient data information
        target_labels: dictionary with class_index -> class_label as keys, values
        img_mode: see convert_patient_images_from_dcm_to_nii
        processes: number of processors to use

    Returns:

    """
    assert not os.path.exists(save_path), f"Path '{save_path}' already exists."
    Path(save_path).mkdir(parents=True)

    dataset_name = dataset_descriptor #  save_path.split('/')[-2].split('_')[-1]
    _, _, df_patient_data = _create_dataset_niigz_from_dicoms(data_path, dataset_name, img_mode, patient_csv,
                                          processes, save_path, target_labels)
    return df_patient_data


def generate_dataset_json(output_file: str, train_identifiers, test_identifiers, modalities: Tuple,
                          labels: dict, dataset_name: str):
    """
    Function needed to run nnUnet preprocessing.

    Args:
        output_file: This needs to be the full path to the dataset.json you intend to write, so
            output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
            imagesTr and labelsTr subfolders
        imagesTr_dir: path to the imagesTr folder of that dataset
        imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
        modalities: tuple of strings with modality names. must be in the same order as the images (first entry
            corresponds to _0000.nii.gz, etc.). Example: ('T1', 'T2', 'FLAIR')
        labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
            supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
        dataset_name: The name of the dataset. Can be anything you want

    Returns:

    """
    json_dict = {'name': dataset_name, 'description': '', 'tensorImageSize': "4D", 'reference': '',
                 'licence': '', 'release': '0.0',
                 'modality': {str(i): modalities[i] for i in range(len(modalities))},
                 'labels': {str(i): labels[i] for i in labels.keys()}, 'numTraining': len(train_identifiers),
                 'numTest': len(test_identifiers),
                 'training': [
                     {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                     for i in train_identifiers
                 ],
                 'test': ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]}

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def convert_msd_dataset_including_test_labels(path, dataset_name, id, num_processes):
    _convert_msd_dataset_my(path, dataset_name, id, num_processes=num_processes)
    labelsTs = os.path.join(path, 'labelsTs')

    target_dataset_name = f"Dataset{id:03d}_{dataset_name}"
    target_folder = os.path.join(nnUNet_raw, target_dataset_name)
    target_labelsTs = os.path.join(target_folder, 'labelsTs')
    os.mkdir(target_labelsTs)

    # copy segmentations
    source_images = [i for i in subfiles(labelsTs, suffix='.nii.gz', join=False) if
                     not i.startswith('.') and not i.startswith('_')]
    for s in source_images:
        shutil.copy(os.path.join(labelsTs, s), os.path.join(target_labelsTs, s))
    return target_dataset_name

def _convert_msd_dataset_my(source_folder: str, dataset_name, overwrite_target_id: int,
                        num_processes: int = 8) -> None:
    '''
    instead of fruitlessly trying to parse dataset name and task id from the folder name,
    pass them.
    '''
    if source_folder.endswith('/') or source_folder.endswith('\\'):
        source_folder = source_folder[:-1]

    labelsTr = join(source_folder, 'labelsTr')
    imagesTs = join(source_folder, 'imagesTs')
    imagesTr = join(source_folder, 'imagesTr')
    assert isdir(labelsTr), f"labelsTr subfolder missing in source folder"
    assert isdir(imagesTs), f"imagesTs subfolder missing in source folder"
    assert isdir(imagesTr), f"imagesTr subfolder missing in source folder"
    dataset_json = join(source_folder, 'dataset.json')
    assert isfile(dataset_json), f"dataset.json missing in source_folder"

    # infer source dataset id and name

    # check if target dataset id is taken
    target_id = overwrite_target_id
    existing_datasets = find_candidate_datasets(target_id)
    assert len(existing_datasets) == 0, f"Target dataset id {target_id} is already taken, please consider changing " \
                                        f"it using overwrite_target_id. Conflicting dataset: {existing_datasets} (check nnUNet_results, nnUNet_preprocessed and nnUNet_raw!)"

    target_dataset_name = f"Dataset{target_id:03d}_{dataset_name}"
    target_folder = join(nnUNet_raw, target_dataset_name)
    target_imagesTr = join(target_folder, 'imagesTr')
    target_imagesTs = join(target_folder, 'imagesTs')
    target_labelsTr = join(target_folder, 'labelsTr')
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        results = []

        # convert 4d train images
        source_images = [i for i in subfiles(imagesTr, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images = [join(imagesTr, i) for i in source_images]

        results.append(
            p.starmap_async(
                split_4d_nifti, zip(source_images, [target_imagesTr] * len(source_images))
            )
        )

        # convert 4d test images
        source_images = [i for i in subfiles(imagesTs, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images = [join(imagesTs, i) for i in source_images]

        results.append(
            p.starmap_async(
                split_4d_nifti, zip(source_images, [target_imagesTs] * len(source_images))
            )
        )

        # copy segmentations
        source_images = [i for i in subfiles(labelsTr, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        for s in source_images:
            shutil.copy(join(labelsTr, s), join(target_labelsTr, s))

        [i.get() for i in results]

    dataset_json = load_json(dataset_json)
    dataset_json['labels'] = {j: int(i) for i, j in dataset_json['labels'].items()}
    dataset_json['file_ending'] = ".nii.gz"
    dataset_json["channel_names"] = dataset_json["modality"]
    del dataset_json["modality"]
    del dataset_json["training"]
    del dataset_json["test"]
    save_json(dataset_json, join(nnUNet_raw, target_dataset_name, 'dataset.json'), sort_keys=False)


def _create_hosp_fold(fold_path, train_cases, val_cases, test_cases, base_dataset_path, dataset_id,
                      hosp, i_fold, img_mode, target_labels):
    train_path = fold_path / 'imagesTr'
    train_path_labels = fold_path / 'labelsTr'
    train_path.mkdir()
    train_path_labels.mkdir()
    test_path = fold_path / 'imagesTs'
    test_path_labels = fold_path / 'labelsTs'
    test_path.mkdir()
    test_path_labels.mkdir()
    for case in train_cases + val_cases:
        os.symlink(os.path.join(base_dataset_path, 'imagesTr', case + '.nii.gz'),
                   os.path.join(train_path, case + '.nii.gz'))
        os.symlink(os.path.join(base_dataset_path, 'labelsTr', case + '.nii.gz'),
                   os.path.join(train_path_labels, case + '.nii.gz'))
    for case in test_cases:
        # since all images in the base dataset are in train, symlink there.
        os.symlink(os.path.join(base_dataset_path, 'imagesTr', case + '.nii.gz'),
                   os.path.join(test_path, case + '.nii.gz'))
        os.symlink(os.path.join(base_dataset_path, 'labelsTr', case + '.nii.gz'),
                   os.path.join(test_path_labels, case + '.nii.gz'))
    # create dataset.json
    generate_dataset_json(os.path.join(fold_path, 'dataset.json'), train_cases + val_cases, test_cases,
                          modalities=tuple(img_mode.split('_')), labels=target_labels,
                          dataset_name=f'{dataset_id:03d}_Cervix: hospital {hosp} fold {i_fold}')
    # create nnunet/raw
    target_dataset_name = convert_msd_dataset_including_test_labels(str(fold_path.absolute()),
                                                                    'cervix',
                                                                    dataset_id, 8)
    # create nnunet/preprocessed
    original_argv = copy.deepcopy(sys.argv)
    sys.argv = ['PLACEHOLDER', '-d', str(dataset_id), '--verify_dataset_integrity', '-c', '2d',
                '-preprocessor_name', 'DefaultPreprocessorStoreStats']
    plan_and_preprocess_entry()
    sys.argv = original_argv
    # create splits_final.json
    nnunet_splits_final_cur = [{'train': train_cases, 'val': val_cases}]
    json_path = os.path.join(nnUNet_preprocessed, target_dataset_name, 'splits_final.json')
    save_json(nnunet_splits_final_cur, json_path)


if __name__ == '__main__':
    setup_logging('/tmp/log.txt')
    data_superdir = '/export/scratch2/data/aleksand/data/'

    input_dataset_path = os.path.join(data_superdir, 'daedalus_full/')
    csv_path = os.path.join(input_dataset_path, 'dataset_v1.csv')
    out_dataset_path = os.path.join(data_superdir, 'Dataset503_cervix')
    if os.path.exists(out_dataset_path):
        shutil.rmtree(out_dataset_path)
    target_labels = {"0": "background", "1": "bladder", "2": "bowel", "3": "rectum", "4": "sigmoid"}

    # ==============================================
    # 1. base dataset: everything is in the train set
    # ==============================================

    img_mode = 't2' #  't2_bffe'
    base_dataset_path = os.path.join(out_dataset_path, 'base/')
    dataset_descriptor = 'cervix'
    df_patient_data = create_base_dataset_from_dicoms(input_dataset_path, base_dataset_path, csv_path, target_labels,
                                    img_mode=img_mode, processes=8, dataset_descriptor=dataset_descriptor)
    # sys.exit(0) # uncomment and go create scanner_info_from_dicoms.yaml via find_scanners.py

    # get scan info: need it to split into hospitals
    # but that data refers to old case names, we need the new names, hence:
    old_to_new_patient_ids = {}
    for index, patient_row in df_patient_data.iterrows():
        old_to_new_patient_ids[patient_row['patient id']] = f'{dataset_descriptor}_{index + 1:03d}' # see _create_dataset_niigz_from_dicoms

    dir_with_csvs = os.path.join(out_dataset_path, 'base', 'imagesTr')
    scanner_info_path = os.path.join(out_dataset_path, 'scanner_info_from_dicoms.yaml')
    extract_and_store_scanner_info(dir_with_csvs, scanner_info_path)

    scanner_info = yaml.safe_load(open(scanner_info_path, 'r'))
    scanner_to_patients = defaultdict(list)
    for patient, scanner in scanner_info.items():
        scanner_to_patients[(scanner[1], scanner[2])].append(old_to_new_patient_ids[patient])

    # ==============================================
    # 2. split into hospitals
    # ==============================================
    n_hospitals = 2
    base_seed = 42
    n_folds = 5
    val_fraction = 0.2
    test_fraction = 0.2
    dataset_id = 300
    if_hospital_based_on_scanner = True
    if if_hospital_based_on_scanner:
        assert n_hospitals == 2
        hosp_to_case_names_based_on_scanner = {}

        hosp = chr(ord('A'))
        hosp_to_case_names_based_on_scanner[hosp] = scanner_to_patients[('Ingenia', '1.5')]

        hosp = chr(ord('B'))
        hosp_to_case_names_based_on_scanner[hosp] = []
        for scanner, patients in scanner_to_patients.items():
            if scanner != ('Ingenia', '1.5'):
                hosp_to_case_names_based_on_scanner[hosp] += patients
    print(f'{hosp_to_case_names_based_on_scanner=}')

    case_names = get_identifiers_from_splitted_files(os.path.join(base_dataset_path, 'imagesTr')).tolist()
    cur_seed = base_seed
    rng = np.random.default_rng(cur_seed)
    rng.shuffle(case_names)

    hosp_to_case_names = {}
    # hosp_to_all_folds = {}
    dataset_id_cases_info = []
    for i_hosp in range(n_hospitals):
        hosp = chr(ord('A') + i_hosp)
        if not if_hospital_based_on_scanner:
            hosp_to_case_names[hosp] = case_names[i_hosp::n_hospitals]
        else:
            hosp_to_case_names[hosp] = hosp_to_case_names_based_on_scanner[hosp]

        folds = np.array_split(hosp_to_case_names[hosp], n_folds)
        # hosp_to_all_folds[hosp] = [f.tolist() for f in folds]

        for i_fold in range(n_folds):
            fold_path = Path(out_dataset_path) / f'hosp_{hosp}' / f'fold_{i_fold}'
            if os.path.exists(fold_path):
                shutil.rmtree(fold_path)
            fold_path.mkdir(parents=True)

            train_cases, val_cases, test_cases = [], [], []
            for j_fold in range(n_folds):
                if j_fold == i_fold:
                    test_cases = folds[j_fold].tolist()
                elif j_fold == (i_fold + 1) % n_folds:
                    val_cases = folds[j_fold].tolist()
                else:
                    train_cases += folds[j_fold].tolist()
            print(f'{train_cases=} {val_cases=} {test_cases=}')
            _create_hosp_fold(fold_path, train_cases, val_cases, test_cases, base_dataset_path, dataset_id,
                              hosp, i_fold, img_mode, target_labels)

            # add to df_dataset_id
            dataset_id_cases_info.append({'dataset_id': dataset_id,
                                          'dataset_name': f'Dataset{dataset_id}_{dataset_descriptor}',
                                          'hosp': hosp, 'fold': i_fold,
                                          'train': train_cases, 'val': val_cases, 'test': test_cases, 'real': 'real'})

            dataset_id += 1

    # 'all'
    hosp = 'all'
    hosp_to_case_names[hosp] = case_names

    df_dataset_id_sep_hosp = pd.DataFrame(dataset_id_cases_info, columns=['dataset_id', 'hosp', 'fold', 'train', 'val', 'test'])

    for i_fold in range(n_folds):
        fold_path = Path(out_dataset_path) / f'hosp_{hosp}' / f'fold_{i_fold}'
        fold_path.mkdir(parents=True, exist_ok=True)

        train_cases, val_cases, test_cases = [], [], []
        for i_hosp in range(n_hospitals):
            hosp_individ = chr(ord('A') + i_hosp)
            row = df_dataset_id_sep_hosp[(df_dataset_id_sep_hosp['hosp'] == hosp_individ) & (df_dataset_id_sep_hosp['fold'] == i_fold)]
            train_cases += row['train'].tolist()[0]
            val_cases += row['val'].tolist()[0]
            test_cases += row['test'].tolist()[0]

        _create_hosp_fold(fold_path, train_cases, val_cases, test_cases, base_dataset_path, dataset_id,
                            hosp, i_fold, img_mode, target_labels)

        # add to df_dataset_id
        dataset_id_cases_info.append({'dataset_id': dataset_id,
                                      'dataset_name': f'Dataset{dataset_id}_{dataset_descriptor}',
                                      'hosp': hosp, 'fold': i_fold,
                                        'train': train_cases, 'val': val_cases, 'test': test_cases, 'real': 'real'})
        dataset_id += 1

    # create entries for syn data to avoid problems with creating nnunet dataset ids on the fly when generating data
    for i_hosp in range(n_hospitals):
        hosp = chr(ord('A') + i_hosp)
        for i_fold in range(n_folds):
            dataset_id_cases_info.append({'dataset_id': dataset_id,
                                          'dataset_name': f'Dataset{dataset_id}_{dataset_descriptor}',
                                          'hosp': hosp, 'fold': i_fold,
                                          'train': [], 'val': [], 'test': [], 'real': 'syn'})
            dataset_id += 1

    for i_fold in range(n_folds):
        dataset_id_cases_info.append({'dataset_id': dataset_id,
                                      'dataset_name': f'Dataset{dataset_id}_{dataset_descriptor}',
                                      'hosp': 'all', 'fold': i_fold,
                                      'train': [], 'val': [], 'test': [], 'real': 'syn'})
        dataset_id += 1

    # ditto for syn-real: for each hospital (but not 'all')
    for i_hosp in range(n_hospitals):
        hosp = chr(ord('A') + i_hosp)
        for i_fold in range(n_folds):
            dataset_id_cases_info.append({'dataset_id': dataset_id,
                                          'dataset_name': f'Dataset{dataset_id}_{dataset_descriptor}',
                                          'hosp': hosp, 'fold': i_fold,
                                          'train': [], 'val': [], 'test': [], 'real': 'syn-real'})
            dataset_id += 1

    df_dataset_id = pd.DataFrame(dataset_id_cases_info, columns=['dataset_id', 'dataset_name', 'hosp', 'fold', 'real', 'train', 'val', 'test'])
    yaml.safe_dump(df_dataset_id.to_dict(), open(os.path.join(out_dataset_path, 'df_dataset_id.yaml'), 'w'))
    # yaml.safe_dump(hosp_to_case_names, open(os.path.join(out_dataset_path, 'hosp_to_case_names.yaml'), 'w'))
    # yaml.safe_dump(hosp_to_all_folds, open(os.path.join(out_dataset_path, 'hosp_to_all_folds.yaml'), 'w'))