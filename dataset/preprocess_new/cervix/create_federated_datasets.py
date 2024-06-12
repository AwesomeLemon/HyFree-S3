import json

import shutil

from pathlib import Path

import pandas as pd
import yaml
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from tqdm import tqdm


def _get_dataset_row_from_df(df_dataset_id, hospital, real, fold):
    return df_dataset_id[(df_dataset_id['hosp'] == hospital)
                         & (df_dataset_id['fold'] == fold)
                         & (df_dataset_id['real'] == real)
                         ].iloc[0]

if __name__ == '__main__':
    path_data = Path('/export/scratch2/data/aleksand/data/')
    base_dataset = 'Dataset503_cervix'
    df_dataset_id = pd.DataFrame(yaml.safe_load(
        open(path_data / base_dataset / 'df_dataset_id.yaml', 'r')))
    dataset_id_cases_info_new = []

    dataset_id_new = 340
    # for each hospital/fold:
    # copy dataset.json, splits_final.json from the hospital itself
    # copy dataset_fingerprint.json, nnUNetPlans.json from real-all
    # copy only the cases of the specific hospital (based on df_dataset_id) in gt_segmentations, nnUNetPlans_2d from real-all
    for hosp in ['A', 'B']:
        for i_fold in tqdm(range(5)):
            row = _get_dataset_row_from_df(df_dataset_id, hosp, 'real', i_fold)
            dataset_name_new = row['dataset_name'].replace(str(row['dataset_id']), str(dataset_id_new))

            # copy
            dataset_name = row['dataset_name']
            dataset_path = Path(nnUNet_preprocessed) / dataset_name
            dataset_path_new = Path(nnUNet_preprocessed) / dataset_name_new
            if dataset_path_new.exists():
                shutil.rmtree(dataset_path_new)
            dataset_path_new.mkdir(parents=True)
            (dataset_path_new / 'gt_segmentations').mkdir()
            (dataset_path_new / 'nnUNetPlans_2d').mkdir()

            dataset_json = json.load(open(dataset_path / 'dataset.json', 'r'))
            dataset_json['name'] = dataset_name_new
            json.dump(dataset_json, open(dataset_path_new / 'dataset.json', 'w'))
            shutil.copy(dataset_path / 'splits_final.json', dataset_path_new)

            row_real_all = _get_dataset_row_from_df(df_dataset_id, 'all', 'real', i_fold)
            dataset_name_real_all = row_real_all['dataset_name']
            dataset_path_real_all = Path(nnUNet_preprocessed) / dataset_name_real_all
            shutil.copy(dataset_path_real_all / 'dataset_fingerprint.json',
                        dataset_path_new)
            nnunet_plans_json = json.load(open(dataset_path_real_all / 'nnUNetPlans.json', 'r'))
            nnunet_plans_json['dataset_name'] = dataset_name_new
            json.dump(nnunet_plans_json, open(dataset_path_new / 'nnUNetPlans.json', 'w'))

            # copy only the relevant cases
            #      test:
            test_cases = row['test']
            dataset_path_real_all_raw = Path(nnUNet_raw) / dataset_name_real_all
            dataset_path_new_raw = Path(nnUNet_raw) / dataset_name_new
            if dataset_path_new_raw.exists():
                shutil.rmtree(dataset_path_new_raw)
            dataset_path_new_raw.mkdir(parents=True)
            (dataset_path_new_raw / 'imagesTs').mkdir()
            (dataset_path_new_raw / 'labelsTs').mkdir()

            for c in test_cases:
                matching_cases = list((dataset_path_real_all_raw / 'imagesTs').glob(c + '*'))
                for case_file in matching_cases: # may have >1 modality => >1 file
                    shutil.copy(case_file, dataset_path_new_raw / 'imagesTs')

                matching_cases = list((dataset_path_real_all_raw / 'labelsTs').glob(c + '*'))
                assert len(matching_cases) == 1
                case_file = matching_cases[0]
                shutil.copy(case_file, dataset_path_new_raw / 'labelsTs')

            #      train & val:
            train_cases = row['train']
            val_cases = row['val']
            for c in train_cases + val_cases:
                matching_cases = list((dataset_path_real_all / 'nnUNetPlans_2d').glob(c + '*'))
                for case_file in matching_cases:
                    shutil.copy(case_file, dataset_path_new / 'nnUNetPlans_2d')

                matching_cases = list((dataset_path_real_all / 'gt_segmentations').glob(c + '*'))
                assert len(matching_cases) == 1
                case_file = matching_cases[0]
                shutil.copy(case_file, dataset_path_new / 'gt_segmentations')

            # overwrite & save new
            row['dataset_id'] = dataset_id_new
            row['dataset_name'] = dataset_name_new
            dataset_id_cases_info_new.append(row.to_dict())

            dataset_id_new += 1

    new_entries = pd.DataFrame.from_records(dataset_id_cases_info_new)
    yaml.safe_dump(new_entries.to_dict(),
                   open(path_data / base_dataset / 'df_dataset_id_federated.yaml', 'w'))