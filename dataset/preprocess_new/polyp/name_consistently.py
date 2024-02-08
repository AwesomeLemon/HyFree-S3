import numpy as np
from pathlib import Path
import shutil
import yaml
from collections import defaultdict
from skimage import io
from tqdm import tqdm


def name_consistently(dataset_path, output_dir):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    images_path = dataset_path / 'Original'
    masks_path = dataset_path / 'Ground Truth'

    subject_id_map = defaultdict(list)

    patient_to_slices = {
        1: list(range(1, 26)),
        2: list(range(26, 51)),
        3: list(range(51, 68)),
        4: list(range(68, 79)),
        5: list(range(79, 104)),
        6: list(range(104, 127)),
        7: list(range(127, 152)),
        8: list(range(152, 178)),
        9: list(range(178, 200)),
        10: list(range(200, 206)),
        11: list(range(206, 228)),
        12: list(range(228, 253)),
        13: list(range(253, 278)),
        14: list(range(278, 298)),
        15: list(range(298, 318)),
        16: list(range(318, 343)),
        17: list(range(343, 364)),
        18: list(range(364, 384)),
        19: list(range(384, 409)),
        20: list(range(409, 429)),
        21: list(range(429, 448)),
        22: list(range(448, 467)),
        23: list(range(467, 479)),
        24: list(range(479, 504)),
        25: list(range(504, 529)),
        26: list(range(529, 547)),
        27: list(range(547, 572)),
        28: list(range(572, 592)),
        29: list(range(592, 613))
    }

    new_id = 10000

    for patient, slices in patient_to_slices.items():
        for slice_id in slices:
            img = io.imread(images_path / f"{slice_id}.tif")
            mask = io.imread(masks_path / f"{slice_id}.tif")
            print(f'{img.shape=}')
            print(f'{mask.shape=}')
            # crazy that tif has values other than 0 and 255
            mask[mask < 128] = 0
            mask[mask >= 128] = 1

            new_img_name = f"clinicdb_{str(new_id + slice_id).zfill(4)}.png"
            new_mask_name = new_img_name

            io.imsave(images_dir / new_img_name, img)
            io.imsave(masks_dir / new_mask_name, mask, check_contrast=False)

            subject_id_map[new_id].append(new_id + slice_id)
        new_id += 100

    with open(output_dir / 'subject_id_map.yaml', 'w') as file:
        yaml.safe_dump(dict(subject_id_map), file)

if __name__ == '__main__':
    data_superdir = Path('/export/scratch2/data/aleksand/data')
    name_consistently(data_superdir / 'CVC-ClinicDB', data_superdir / 'clinicdb_v0')