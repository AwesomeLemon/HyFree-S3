from collections import defaultdict

import numpy as np
from typing import List

import yaml
from pathlib import Path

import open_clip

import os

import faiss
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

from dataset.memorization.dataset_png_paths_paths import DatasetPngPathsReturnPaths


def search_paths_in_index(paths_in: List[Path], index_dir_path: Path, path_out: Path, extension='.png', n_search=3):
    model_name = 'ViT-g-14'
    weights_name = 'laion2b_s12b_b42k'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name,
                                                                 pretrained=weights_name)

    model = model.cuda()
    model.eval()

    dataset = DatasetPngPathsReturnPaths(paths_in, preprocess, extension)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    index = faiss.read_index(os.path.join(index_dir_path, 'index.faiss'))
    paths_in_index = yaml.safe_load(open(os.path.join(index_dir_path, 'paths.yaml'), 'r'))['paths']

    path_out.mkdir(parents=True, exist_ok=True)
    info_closest = defaultdict(defaultdict)

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            images, paths = batch
            images = images.cuda()
            features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)
            features = features.detach().cpu().numpy().astype('float32')  # previously, code worked w/ float16, not sure why that started throuwing errors

            D, I = index.search(features, n_search)

            for j in range(len(paths)):
                info_closest[paths[j]]['paths'] = [paths_in_index[k] for k in I[j]]
                info_closest[paths[j]]['distances'] = D[j].tolist()

    # defaultdicts to dicts
    info_closest = {k: dict(v) for k, v in info_closest.items()}
    yaml.safe_dump(info_closest, open(os.path.join(path_out, 'info_closest.yaml'), 'w'), default_flow_style=False)


def histogram_closest(path, index_type: str, search_type: str, suffix='', threshold_path=None, title=None):
    info_closest = yaml.safe_load(open(os.path.join(path, f'info_closest{suffix}.yaml'), 'r'))
    distances = [closest_dict['distances'][0] for path_query, closest_dict in tqdm(info_closest.items())]

    plt.style.use('seaborn')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    sns.set(font_scale=1.2)

    plt.figure(figsize=(6, 2.5))
    sns.histplot(distances, bins=30, log_scale=True)
    plt.xlim(left=1e-2, right=1)
    plt.xlabel('OpenCLIP distance')
    plt.ylabel('Number of images')
    # plt.title(f'Smallest distance from each {search_type} image to all {index_type} images')
    if title is not None:
        plt.title(title)
    if threshold_path is not None:
        threshold = yaml.safe_load(open(threshold_path, 'r'))['threshold']
        plt.axvline(threshold, 0, 1, color='red')
    plt.savefig(os.path.join(path, f'histogram_closest.png'), bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def percentile_dist_from_index(path, percentile=5):
    info_closest = yaml.safe_load(open(os.path.join(path, 'info_closest.yaml'), 'r'))
    distances = [closest_dict['distances'][0] for path_query, closest_dict in info_closest.items()]
    return float(np.percentile(distances, percentile))


def paths_with_dist_below_threshold_from_index(path, threshold):
    info_closest = yaml.safe_load(open(os.path.join(path, 'info_closest.yaml'), 'r'))
    paths = [path_query for path_query, closest_dict in info_closest.items() if closest_dict['distances'][0] < threshold]

    info_closest_filtered = {path_query: closest_dict for path_query, closest_dict in info_closest.items() if path_query not in paths}
    yaml.safe_dump(info_closest_filtered, open(os.path.join(path, 'info_closest_filtered.yaml'), 'w'), default_flow_style=False)

    info_closest_removed = {path_query: closest_dict for path_query, closest_dict in info_closest.items() if path_query in paths}
    yaml.safe_dump(info_closest_removed, open(os.path.join(path, 'info_closest_removed.yaml'), 'w'), default_flow_style=False)
    return paths