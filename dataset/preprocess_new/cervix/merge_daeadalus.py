from pathlib import Path
import pandas as pd
import shutil
import os

if __name__ == '__main__':
    part_old_path = Path('/export/scratch1/home/aleksand/s2/data/daedalus0')
    part_new_path = Path('/export/scratch1/home/aleksand/s2/data/CervixRT_1492_part')

    out_path = Path('/export/scratch1/home/aleksand/s2/data/daedalus_full')
    out_path.mkdir(exist_ok=True)

    out_data = []

    df_newpart = pd.read_csv(part_new_path / 'dataset_v1.csv')
    for row in df_newpart.iterrows():
        patient_id = row[1]['patient id']
        print(f'{row=}')
        os.symlink(part_new_path / patient_id, out_path / patient_id)
        out_data.append(row[1])

    df_oldpart = pd.read_csv(part_old_path / 'dataset_final_v2.csv')
    for row in df_oldpart.iterrows():
        patient_id = row[1]['patient id']
        print(f'{row=}')
        os.symlink(part_old_path / patient_id, out_path / patient_id)

        # only the last part of the path is needed + remove trailing slash:
        row[1]['folder path'] = row[1]['folder path'].split('\\')[-2]
        out_data.append(row[1])

    df_out = pd.DataFrame(out_data)
    df_out.to_csv(out_path / 'dataset_v1.csv', index=False)