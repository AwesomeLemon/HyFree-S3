'''
convert nnunet-preprocessed arrays to pngs
'''
import shutil
import numpy as np
import tqdm
import yaml
from PIL import Image
from skimage.exposure import rescale_intensity

def to_png(in_path, out_path, patch_size, rescale_to_percentiles, keep_nth):
    in_path = in_path / 'nnUNetPlans_2d'
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir()

    for case in tqdm.tqdm(in_path.glob('*.npz')):
        case_name = case.stem
        data_and_seg = np.load(case)
        yaml_path = case.parent / f'{case_name}.yaml'
        stats = yaml.safe_load(open(yaml_path, 'r'))

        data = data_and_seg['data']
        for channel in range(data.shape[0]):
            mean, std = stats[channel]
            data[channel] = data[channel] * std + mean

        data[data < 0] = 0

        if rescale_to_percentiles:
            lb = np.percentile(data, 10)
            ub = np.percentile(data, 99)
        else:
            lb = data.min()
            ub = data.max()

        data = rescale_intensity(data, in_range=(lb, ub))

        data = (data * 255).astype('uint8')
        color_mode = 'L' if data.shape[0] == 1 else 'RGB'
        # cur data shape CDHW
        for i in range(data.shape[1]):
            if i % keep_nth != 0:
                continue
            data_slice = data[:, i]  # CHW
            if color_mode == 'RGB':
                data_slice = np.transpose(data_slice, (1, 2, 0))  # HWC
            else:
                assert color_mode == 'L'
                data_slice = data_slice[0]  # HW
            pil_im = Image.fromarray(data_slice, mode=color_mode)
            pil_im = pil_im.resize(patch_size, Image.Resampling.BILINEAR)
            pil_im.save(out_path / f'{case_name}_{i:04d}.png')