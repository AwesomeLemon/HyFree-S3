import sys

import numpy as np
import pandas as pd
import yaml
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunetv2_mod.nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
from pathlib import Path

import shutil
import copy

import os
from nnunetv2_mod.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2_mod.nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


def create_base_dataset_from_pngs(in_path, out_path):
    images_tr_path = out_path / 'imagesTr'
    labels_tr_path = out_path / 'labelsTr'
    if images_tr_path.exists():
        shutil.rmtree(images_tr_path)
    if labels_tr_path.exists():
        shutil.rmtree(labels_tr_path)
    images_tr_path.mkdir(parents=True, exist_ok=True)
    labels_tr_path.mkdir(parents=True, exist_ok=True)
    (out_path / 'imagesTs').mkdir(exist_ok=True)

    for img_path in in_path.glob('images/*.png'):
        new_name_added_0000 = img_path.name.replace('.png', '_0000.png')
        shutil.copy(img_path, images_tr_path / new_name_added_0000)

        mask_path = in_path / 'masks' / f"{img_path.name}"
        shutil.copy(mask_path, labels_tr_path / img_path.name)


def _create_hosp_fold(fold_path, train_cases_files, val_cases_files, test_cases_files,
                      base_dataset_path, dataset_id, dataset_descriptor,
                      hosp, i_fold, target_labels, channel_names):
    train_path = fold_path / 'imagesTr'
    train_path_labels = fold_path / 'labelsTr'
    train_path.mkdir()
    train_path_labels.mkdir()
    test_path = fold_path / 'imagesTs'
    test_path_labels = fold_path / 'labelsTs'
    test_path.mkdir()
    test_path_labels.mkdir()
    for case in train_cases_files + val_cases_files:
        os.symlink(os.path.join(base_dataset_path, 'imagesTr', case + '_0000.png'),
                   os.path.join(train_path, case + '_0000.png'))
        os.symlink(os.path.join(base_dataset_path, 'labelsTr', case + '.png'),
                   os.path.join(train_path_labels, case + '.png'))
    for case in test_cases_files:
        # since all images in the base dataset are in train, symlink there.
        os.symlink(os.path.join(base_dataset_path, 'imagesTr', case + '_0000.png'),
                   os.path.join(test_path, case + '_0000.png'))
        os.symlink(os.path.join(base_dataset_path, 'labelsTr', case + '.png'),
                   os.path.join(test_path_labels, case + '.png'))
    # create dataset.json
    dataset_name = f'Dataset{dataset_id}_{dataset_descriptor}'
    generate_dataset_json(str((fold_path).absolute()), channel_names,
                          target_labels,
                          len(train_cases_files + val_cases_files), '.png', dataset_name=dataset_name,
                          description=f'{dataset_name}: hospital {hosp} fold {i_fold}')
    # create nnunet/raw
    shutil.copytree(str(fold_path.absolute()), os.path.join(nnUNet_raw, dataset_name))
    # create nnunet/preprocessed
    original_argv = copy.deepcopy(sys.argv)
    sys.argv = ['PLACEHOLDER', '-d', str(dataset_id), '--verify_dataset_integrity', '-c', '2d',
                '-preprocessor_name', 'DefaultPreprocessorStoreStats']
    plan_and_preprocess_entry()
    sys.argv = original_argv
    # create splits_final.json
    nnunet_splits_final_cur = [{'train': train_cases_files, 'val': val_cases_files}]
    json_path = os.path.join(nnUNet_preprocessed, dataset_name, 'splits_final.json')
    save_json(nnunet_splits_final_cur, json_path)
    return dataset_name


if __name__ == '__main__':
    data_superdir = '/export/scratch2/data/aleksand/data/'

    dataset_descriptor = 'polyp'
    dataset_id = 220
    out_dataset_path = os.path.join(data_superdir, f'{dataset_descriptor}_base_{dataset_id}/')
    if os.path.exists(out_dataset_path):
        shutil.rmtree(out_dataset_path)
    target_labels = {'background': 0, 'foreground': 1} # yes, nnunet wants the reverse order
    channel_names = {0: 'R', 1: 'G', 2: 'B'}  #{0: 'L'}

    # ==============================================
    # 1. base dataset: everything is in the train set
    # ==============================================

    base_dataset_path = os.path.join(out_dataset_path, 'base/')
    png_dataset_path = os.path.join(data_superdir, 'polyp_v0')
    create_base_dataset_from_pngs(Path(png_dataset_path), Path(base_dataset_path))
    # sys.exit(0)

    # ==============================================
    # 2. split into hospitals
    # ==============================================
    n_hospitals = 2
    base_seed = 42
    n_folds = 5
    val_fraction = 0.2
    test_fraction = 0.2

    subject_id_map = yaml.safe_load(open(os.path.join(png_dataset_path, 'subject_id_map.yaml'), 'r'))
    case_names = list(subject_id_map.keys())
    def _case_name_list_to_all_files(case_list):
        return [f'{dataset_descriptor}_{str(tt).zfill(6)}' for t in case_list for tt in subject_id_map[t]]
    def _case_name_to_files_dict(case_list):
        res = {}
        for t in case_list:
            res[t] = [f'{dataset_descriptor}_{str(tt).zfill(6)}' for tt in subject_id_map[t]]
        return res
    cur_seed = base_seed
    rng = np.random.default_rng(cur_seed)
    rng.shuffle(case_names)

    hosp_to_case_names = yaml.safe_load(open(os.path.join(png_dataset_path, 'hospital_to_cases.yaml'), 'r'))
    # hosp_to_all_folds = {}
    dataset_id_cases_info = []
    for i_hosp in range(n_hospitals):
        hosp = chr(ord('A') + i_hosp)

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
            test_cases_files = _case_name_list_to_all_files(test_cases)
            val_cases_files = _case_name_list_to_all_files(val_cases)
            train_cases_files = _case_name_list_to_all_files(train_cases)

            print(f'{train_cases=} {val_cases=} {test_cases=}')
            dataset_name = _create_hosp_fold(fold_path, train_cases_files, val_cases_files, test_cases_files, base_dataset_path, dataset_id, dataset_descriptor,
                              hosp, i_fold, target_labels, channel_names)
            case_name_to_files_dict = _case_name_to_files_dict(train_cases + val_cases)
            yaml.safe_dump(case_name_to_files_dict,
                           open(os.path.join(nnUNet_preprocessed, dataset_name, 'case_name_to_files.yaml'), 'w'))

            # add to df_dataset_id
            dataset_id_cases_info.append({'dataset_id': dataset_id,
                                          'dataset_name': f'Dataset{dataset_id}_{dataset_descriptor}',
                                          'hosp': hosp, 'fold': i_fold,
                                          'train': train_cases_files, 'val': val_cases_files, 'test': test_cases_files, 'real': 'real'})

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
                          dataset_descriptor, hosp, i_fold, target_labels, channel_names)

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