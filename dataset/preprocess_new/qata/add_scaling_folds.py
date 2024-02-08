import pandas as pd
import yaml
from dataset.preprocess_new.qata.qata_to_nnunet_folds import _create_hosp_fold
from pathlib import Path

import os

if __name__ == '__main__':
    data_superdir = '/export/scratch2/data/aleksand/data/'

    dataset_descriptor = 'qata'
    dataset_id = 500
    out_dataset_path = os.path.join(data_superdir, f'{dataset_descriptor}_base_{dataset_id}/')
    target_labels = {'background': 0, 'foreground': 1}

    # ==============================================
    # 1. skip base dataset
    # ==============================================

    base_dataset_path = os.path.join(out_dataset_path, 'base/')
    qata_named_path = os.path.join(data_superdir, 'qata_named_v1')

    # ==============================================
    # 2. split into hospitals
    # ==============================================
    n_hospitals = 8
    base_seed = 42
    n_folds = 5
    val_fraction = 0.2
    test_fraction = 0.2

    subject_id_map = yaml.safe_load(open(os.path.join(qata_named_path, 'subject_id_map.yaml'), 'r'))
    case_names = list(subject_id_map.keys())


    def _case_name_list_to_all_files(case_list):
        return [f'qata_{str(tt).zfill(4)}' for t in case_list for tt in subject_id_map[t]]


    def _case_name_to_files_dict(case_list):
        res = {}
        for t in case_list:
            res[t] = [f'qata_{str(tt).zfill(4)}' for tt in subject_id_map[t]]
        return res


    dataset_id_cases_info = yaml.safe_load(open(os.path.join(out_dataset_path, 'df_dataset_id.yaml'), 'r'))
    dataset_id = 100
    # max_existing_dataset_id = max(dataset_id_cases_info['dataset_id'].values())
    # dataset_id = max_existing_dataset_id + 1

    df_dataset_id = pd.DataFrame(dataset_id_cases_info)
    dataset_id_cases_info_new = []

    # real2, real4
    for merged_hosps in [2, 4]:
        cur_hosps = list(range(n_hospitals))[:merged_hosps]
        cur_hosps_names = [chr(ord('A') + i_hosp) for i_hosp in cur_hosps]
        hosp = ''.join(cur_hosps_names)
        for i_fold in range(n_folds):
            fold_path = Path(out_dataset_path) / f'hosp_{hosp}' / f'fold_{i_fold}'
            fold_path.mkdir(parents=True, exist_ok=True)

            train_cases, val_cases, test_cases = [], [], []

            for hosp_individ in cur_hosps_names:
                row = df_dataset_id[(df_dataset_id['hosp'] == hosp_individ) & (df_dataset_id['fold'] == i_fold) & (df_dataset_id['real'] == 'real')]
                train_cases += row['train'].tolist()[0]
                val_cases += row['val'].tolist()[0]
                test_cases += row['test'].tolist()[0]

            _create_hosp_fold(fold_path, train_cases, val_cases, test_cases, base_dataset_path, dataset_id,
                              dataset_descriptor, hosp, i_fold, target_labels)

            # add to df_dataset_id
            dataset_id_cases_info_new.append({'dataset_id': dataset_id,
                                          'dataset_name': f'Dataset{dataset_id}_{dataset_descriptor}',
                                          'hosp': hosp, 'fold': i_fold,
                                            'train': train_cases, 'val': val_cases, 'test': test_cases, 'real': 'real'})
            dataset_id += 1

    # syn2, syn4
    for merged_hosps in [2, 4]:
        cur_hosps = list(range(n_hospitals))[:merged_hosps]
        cur_hosps_names = [chr(ord('A') + i_hosp) for i_hosp in cur_hosps]
        hosp = ''.join(cur_hosps_names)
        for i_fold in range(n_folds):
            dataset_id_cases_info_new.append({'dataset_id': dataset_id,
                                          'dataset_name': f'Dataset{dataset_id}_{dataset_descriptor}',
                                          'hosp': hosp, 'fold': i_fold,
                                          'train': [], 'val': [], 'test': [], 'real': 'syn'})
            dataset_id += 1

    # syn2-real, syn4-real for all hospitals
    for merged_hosps in [2, 4]:
        for i_hosp in range(n_hospitals):
            hosp = chr(ord('A') + i_hosp)
            for i_fold in range(n_folds):
                dataset_id_cases_info_new.append({'dataset_id': dataset_id,
                                              'dataset_name': f'Dataset{dataset_id}_{dataset_descriptor}',
                                              'hosp': hosp, 'fold': i_fold,
                                              'train': [], 'val': [], 'test': [], 'real': f'syn{merged_hosps}-real'})
                dataset_id += 1

    new_entries = pd.DataFrame.from_records(dataset_id_cases_info_new)
    df_dataset_id = pd.concat([df_dataset_id, new_entries], ignore_index=True)
    yaml.safe_dump(df_dataset_id.to_dict(), open(os.path.join(out_dataset_path, 'df_dataset_id_scaling.yaml'), 'w'))