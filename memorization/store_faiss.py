from typing import List

import yaml
from pathlib import Path

import open_clip

import faiss
import torch
from tqdm import tqdm

from dataset.memorization.dataset_png_paths_paths import DatasetPngPathsReturnPaths

def create_index_from_paths(paths_in: List[Path], path_out: Path, extension='.png'):
    model_name = 'ViT-g-14'
    weights_name = 'laion2b_s12b_b42k'
    d = 1024
    model, _, preprocess = open_clip.create_model_and_transforms(model_name,
                                                                 pretrained=weights_name)
    model = model.cuda()
    model.eval()

    dataset = DatasetPngPathsReturnPaths(paths_in, preprocess, extension)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    index = faiss.IndexFlatL2(d)

    paths_all = []

    print('Starting iteration over dataset')
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            images, paths = batch
            images = images.cuda()
            features = model.encode_image(images)
            print(f'{features.shape=}')
            features /= features.norm(dim=-1, keepdim=True)
            features = features.detach().cpu().numpy().astype('float32') #previously, code worked w/ float16, not sure why that started throuwing errors
            index.add(features)
            paths_all.extend(paths)

    path_out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str((path_out / 'index.faiss').absolute()))
    yaml.safe_dump({'paths': paths_all}, open(path_out / 'paths.yaml', 'w'), default_flow_style=False)