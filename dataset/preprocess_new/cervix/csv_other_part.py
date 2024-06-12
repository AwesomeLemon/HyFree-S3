import glob
import sys
from os.path import join
from pathlib import Path
import os
import tqdm
from collections import defaultdict
import traceback

import pandas as pd
import pydicom
from functools import cmp_to_key
import re


def create_patient_dict(data_path):
    '''
    Function from Vangelis
    '''
    # data_path contains folders and some of them contain the desired MRI folders with images and RTSTRUCT folders with
    # ground truths
    patient_folder_paths = glob.glob(data_path + '*/')
    # patient dict will contain a list for every key (patient id), which will contain dicts with pairs of
    # image-ground truths for every patient image-ground truth
    patient_dict = {}
    for folder_path in patient_folder_paths:
        patient_id = folder_path.split('/')[-2]
        subfolder_paths = glob.glob(folder_path + '*/')
        # search every subfolder if it has an MRI folder and an RTSTRUCT folder
        for subfolder_path in subfolder_paths:
            final_folder_paths = glob.glob(subfolder_path + '*/')
            mr_path = [path for path in final_folder_paths if 'MR' in path]
            gr_path = [path for path in final_folder_paths if 'RTSTRUCT' in path]
            if mr_path and gr_path:
                # keep this MR folder path together with its ground truth folder path
                if patient_id in patient_dict.keys():
                    patient_dict[patient_id].append({'mri_path': mr_path[0], 'gr_path': gr_path[0]})
                else:
                    patient_dict[patient_id] = [{'mri_path': mr_path[0], 'gr_path': gr_path[0]}]
            elif mr_path and not gr_path:
                # some patient folders do not contain the MRI and the RTSTRUCT in the same folder
                if patient_id in patient_dict.keys():
                    patient_dict[patient_id].append({'mri_path': mr_path[0], 'gr_path': None})
                else:
                    patient_dict[patient_id] = [{'mri_path': mr_path[0], 'gr_path': None}]
    return patient_dict

def get_all_ok_patients(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Split patient_id to separate the name and index
    df['patient_name'] = df['patient_id'].str.rsplit('_', n=1).str[0]

    # Group by patient name and check if all statuses are 'ok'
    all_ok = df.groupby('patient_name')['status'].apply(lambda x: (x == 'ok').all())

    # Get the names of patients where all statuses are 'ok'
    ok_patients = all_ok[all_ok].index.tolist()

    return ok_patients


def _select_mri_by_scanner(patient_mris):
    scanners = []
    for potential_path in patient_mris:
        potential_mri_path = potential_path['mri_path']
        scanner_info_cur = []
        # read scanner info from all .dcm slices
        for root, dirs, files in os.walk(potential_mri_path):
            for file in tqdm.tqdm(files):
                if file.endswith('.dcm'):
                    try:
                        file_path = os.path.join(root, file)
                        dcm = pydicom.dcmread(file_path)
                        model = dcm.get("ManufacturerModelName", "Unknown")
                        magnetic_field_strength = dcm.get("MagneticFieldStrength", "Unknown")
                        scanner_info_cur.append((str(model), str(magnetic_field_strength)))
                    except:
                        print(traceback.format_exc())
                        print(f"Error reading {file}")
        # majority vote (because some slices have incorrect info on the scanner)
        scanner_to_count = defaultdict(int)
        for scanner in scanner_info_cur:
            scanner_to_count[scanner] += 1
        scanners.append(max(scanner_to_count, key=scanner_to_count.get))

    paths_and_scanners = zip(patient_mris, scanners)
    paths_and_scanners_sorted = list(sorted(paths_and_scanners, key=lambda x: cmp_to_key(_my_scanner_comparator)(x[1])))
    print(f"{paths_and_scanners_sorted[-1][0]['mri_path']} {paths_and_scanners_sorted[-1][0]['gr_path']}")
    path = Path(paths_and_scanners_sorted[-1][0]['mri_path']).parent.name
    return path

def _my_scanner_comparator(x, y):
    '''
    Ingenia 1.5 < everything else 1.5 < everything else 3
    '''
    if x == y:
        return 0

    if x == ('Ingenia', '1.5'):
        return -1

    if y == ('Ingenia', '1.5'):
        return 1

    if float(x[1]) < float(y[1]):
        return -1

    if float(x[1]) > float(y[1]):
        return 1

    return 0


if __name__ == '__main__':
    other_part_path = '/export/scratch1/home/aleksand/s2/data/CervixRT_1492_part/'
    patient_dict = create_patient_dict(other_part_path)

    csv_filename = join(other_part_path, 'patient_status_preliminary.csv')
    last_patient_id_to_use = 297 #  the csv also includes the patients from the other part => ignore them

    ok_patients = get_all_ok_patients(csv_filename)
    print("Patients with all 'ok' statuses:", ok_patients)

    data = []
    pattern = re.compile(r'\d+')
    for p in ok_patients:
        patient_id = int(pattern.findall(p)[-1])
        if patient_id <= last_patient_id_to_use:
            if len(patient_dict[p]) > 1:
                path = _select_mri_by_scanner(patient_dict[p])
            else:
                path = Path(patient_dict[p][0]['mri_path']).parent.name

            data.append([p, path, 5])

    new_csv = pd.DataFrame(data, columns=['patient id', 'folder path', 'slices to remove'])
    new_csv.to_csv(join(other_part_path, 'dataset_v1.csv'), index=False)