import shutil
import yaml

from pathlib import Path

if __name__ == '__main__':
    data_superdir = Path('/export/scratch2/data/aleksand/data')
    hyperkvasir_path = data_superdir / 'hyperkvasir_v1'
    clinicdb_path = data_superdir / 'clinicdb_v0'
    out_path = data_superdir / 'polyp_v0'
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)

    images_out = out_path / 'images'
    masks_out = out_path / 'masks'

    images_out.mkdir()
    masks_out.mkdir()

    hospital_to_cases = {'A': [], 'B': []}
    subject_id_map_hyperkvasir = yaml.safe_load(open(hyperkvasir_path / 'subject_id_map.yaml'))
    subject_id_map_clinicdb = yaml.safe_load(open(clinicdb_path / 'subject_id_map.yaml'))

    subject_id_map_merged = {}
    new_id = 0
    for case, slices in subject_id_map_hyperkvasir.items():
        cur_slices = []
        for i, slice_id in enumerate(slices):
            shutil.copy(hyperkvasir_path / 'images' / f"hyperkvasir_{str(slice_id).zfill(4)}.png",
                        images_out / f"polyp_{str(new_id + i).zfill(6)}.png")
            shutil.copy(hyperkvasir_path / 'masks' / f"hyperkvasir_{str(slice_id).zfill(4)}.png",
                        masks_out / f"polyp_{str(new_id + i).zfill(6)}.png")
            cur_slices.append(new_id + i)
        subject_id_map_merged[new_id] = cur_slices

        hospital_to_cases['A'].append(new_id)
        new_id += 100

    for case, slices in subject_id_map_clinicdb.items():
        cur_slices = []

        for i, slice_id in enumerate(slices):
            shutil.copy(clinicdb_path / 'images' / f"clinicdb_{str(slice_id).zfill(4)}.png",
                        images_out / f"polyp_{str(new_id + i).zfill(6)}.png")
            shutil.copy(clinicdb_path / 'masks' / f"clinicdb_{str(slice_id).zfill(4)}.png",
                        masks_out / f"polyp_{str(new_id + i).zfill(6)}.png")
            cur_slices.append(new_id + i)
        subject_id_map_merged[new_id] = cur_slices

        hospital_to_cases['B'].append(new_id)
        new_id += 100

    yaml.safe_dump(hospital_to_cases, open(out_path / 'hospital_to_cases.yaml', 'w'))
    yaml.safe_dump(subject_id_map_merged, open(out_path / 'subject_id_map.yaml', 'w'))
