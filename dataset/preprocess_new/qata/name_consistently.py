from pathlib import Path
import shutil
import yaml
from collections import defaultdict
from skimage import io

def name_consistently_and_track_patients(dataset_path, output_dir):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    subject_id_map = defaultdict(list)
    new_id = 0
    # non_sub_id = 13000

    for subset in ['Train Set', 'Test Set']:
        subset_path = dataset_path / subset
        images_path = subset_path / 'Images'
        truths_path = subset_path / 'Ground-truths'

        for img_path in images_path.glob('*.png'):
            if 'sub' in img_path.name:
                subject_id = int(img_path.name.split('-')[1].split('_')[0][1:])
            else:
                continue # some images do not have subject id yet belong to the same subject
                # subject_id = non_sub_id
                # non_sub_id += 1

            mask_file = truths_path / f"mask_{img_path.name}"
            mask = io.imread(mask_file)
            mask[mask == 255] = 1

            new_img_name = f"qata_{str(new_id).zfill(4)}.png"
            new_mask_name = new_img_name

            shutil.copy(img_path, images_dir / new_img_name)
            io.imsave(masks_dir / new_mask_name, mask,
                      check_contrast=False)

            subject_id_map[subject_id].append(new_id)
            new_id += 1

    with open(output_dir / 'subject_id_map.yaml', 'w') as file:
        yaml.safe_dump(dict(subject_id_map), file)


if __name__ == '__main__':
    data_superdir = Path('/export/scratch2/data/aleksand/data')
    name_consistently_and_track_patients(data_superdir / 'QaTa-COV19' / 'QaTa-COV19-v2', data_superdir / 'qata_named_v1')