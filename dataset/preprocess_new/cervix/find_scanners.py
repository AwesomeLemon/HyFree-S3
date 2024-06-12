'''
to look at scanners of the exact MRIs that are actually used for training,
store .csv with MRI paths when preprocessing the dataset,
and then run this script.
'''
import os
from pathlib import Path

import pydicom
from collections import defaultdict

import yaml
import tqdm
import pandas as pd

def extract_and_store_scanner_info(dir_with_csvs, scanner_info_path):
    info = extract_scanner_info(dir_with_csvs)
    # majority vote for each patient
    patient_to_scanner = {}
    for patient, scanners in info.items():
        scanner_to_count = defaultdict(int)
        for scanner in scanners:
            scanner_to_count[scanner] += 1
        patient_to_scanner[patient] = max(scanner_to_count, key=scanner_to_count.get)
    yaml.safe_dump(patient_to_scanner, open(scanner_info_path, 'w'))

def extract_scanner_info(dir_with_csvs):
    scanner_info = defaultdict(list)
    prev_patient = -1
    for item in Path(dir_with_csvs).glob('*.csv'):
        patient_id_new = item.stem # numeration starts from 1 (e.g. cervix_001); not used here
        path = pd.read_csv(item).iloc[0]['path']
        patient_id_old = path.split('/')[-4] # arbitrary numeration (e.g. CervixRT123)

        for root, dirs, files in os.walk(path):
            for file in tqdm.tqdm(files):
                if file.endswith('.dcm'):
                    try:
                        file_path = os.path.join(root, file)
                        dcm = pydicom.dcmread(file_path)
                        patient_id = dcm.PatientID
                        assert patient_id == patient_id_old
                        manufacturer = dcm.get("Manufacturer", "Unknown")
                        model = dcm.get("ManufacturerModelName", "Unknown")
                        magnetic_field_strength = dcm.get("MagneticFieldStrength", "Unknown")
                        scanner_info[patient_id].append(
                            (str(manufacturer), str(model), str(magnetic_field_strength)))
                        if patient_id != prev_patient:
                            if prev_patient != -1:
                                print(f"Patient {prev_patient}")
                            prev_patient = patient_id
                    except:
                        print(f"Error reading {file}")
    return scanner_info


if __name__ == '__main__':
    root_dir = '/export/scratch2/data/aleksand/data/Dataset508_cervix_DEBUG/'

    dir_with_csvs = os.path.join(root_dir, 'base', 'imagesTr')
    scanner_info_path = os.path.join(root_dir, 'scanner_info_from_dicoms.yaml')
    extract_and_store_scanner_info(dir_with_csvs, scanner_info_path)

    scanner_info = yaml.safe_load(open(scanner_info_path, 'r'))
    scanner_to_patients = defaultdict(list)
    for patient, scanner in scanner_info.items():
        scanner_to_patients[(scanner[1], scanner[2])].append(patient)

    for scanner, patients in scanner_to_patients.items():
        print(f'{scanner=}, {len(patients)=}')