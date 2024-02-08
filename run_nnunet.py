import math
import os
import copy
import pickle
import numpy as np
import json
import shutil
import tempfile
from collections import defaultdict

from PIL import Image
from datetime import datetime
from os.path import join

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from pathlib import Path

import cv2
import hydra
import ray
import torch
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from omegaconf import DictConfig
import pandas as pd
import yaml
from nnunetv2.run.run_training import run_training

from dataset.preprocess_new.common.random_split import get_split_fn
from dataset.preprocess_new.common.to_png import to_png
from memorization.search_faiss import search_paths_in_index, histogram_closest, \
    paths_with_dist_below_threshold_from_index, percentile_dist_from_index
from memorization.store_faiss import create_index_from_paths
from utils.general import setup_logging, set_random_seeds
from utils.rsync_wrapper import RsyncWrapper

from stylegan2_mod.DiffAugment.train import main as train_gan
from stylegan2_mod.DiffAugment.generate import generate_images


@ray.remote(num_gpus=1)
def train_gan_ray(out_dir, cfg, rsync_w, df_dataset_id, hospital, seed, gpus=1):
    set_random_seeds(seed)
    out_dir.mkdir(exist_ok=True, parents=True)

    row = _get_dataset_row_from_df(df_dataset_id, hospital, 'real', cfg.data.fold)
    nnunet_dataset_name = row['dataset_name']

    # get the png dataset that the gan will be trained on
    png_dir_name = nnunet_dataset_name + '_png'
    in_dir = Path(cfg.path.data) / png_dir_name
    print(f'{in_dir=}')
    rsync_w.download(in_dir, if_dir=True)

    # need strides & kernel sizes
    dataset_path = Path(nnUNet_preprocessed) / nnunet_dataset_name
    rsync_w.download(dataset_path, if_dir=True)
    nnunet_plans_json = dataset_path / 'nnUNetPlans.json'
    nnunet_plans = json.load(open(nnunet_plans_json, 'r'))

    kimg = cfg.train.gan.kimg
    if kimg == 'auto':
        kimg = len(list(filter(lambda x: 'mask' not in x.name,
                               in_dir.rglob('*.png'))))
    print(f'{str(in_dir.absolute())=} {kimg=}')
    config_kwargs = {'runname': 'gan',
        'data': str(in_dir.absolute()),
        'mirror': True,
        'kimg': kimg,
        'gpus': gpus,
        'snap': cfg.train.gan.snap,
        'seed': seed,
        'nnunet_strides': nnunet_plans['configurations']['2d']['pool_op_kernel_sizes'],
        'nnunet_kernel_sizes': nnunet_plans['configurations']['2d']['conv_kernel_sizes'],
    }
    train_gan(None, str(out_dir.absolute()), False, **config_kwargs)

    rsync_w.upload(str(out_dir.absolute()), if_dir=True)


@ray.remote(num_cpus=4)
def prepare_dataset_for_gan(cfg, rsync_w, df_dataset_id, hospital) -> None:
    row = _get_dataset_row_from_df(df_dataset_id, hospital, 'real', cfg.data.fold)
    dataset_name = row['dataset_name']
    dataset_path = Path(nnUNet_preprocessed) / dataset_name
    rsync_w.download(dataset_path, if_dir=True)

    nnunet_plans_json = dataset_path / 'nnUNetPlans.json'
    nnunet_plans = json.load(open(nnunet_plans_json, 'r'))
    patch_size = nnunet_plans['configurations']['2d']['patch_size']

    data_root = Path(cfg.path.data)
    out_dataset_name = dataset_name + '_png'
    out_path = data_root / out_dataset_name

    dataset_fingerprint = json.load(open(dataset_path / 'dataset_fingerprint.json', 'r'))
    spacings = dataset_fingerprint['spacings']
    median_spacing_z = np.median(np.array(spacings)[:, 0])
    keep_nth = 1 if median_spacing_z >= 4 else math.ceil(4 / median_spacing_z)
    to_png(dataset_path, out_path, patch_size, not cfg.data.original_data_was_png, keep_nth)

    rsync_w.upload(out_path, if_dir=True)


@ray.remote(num_gpus=0.5)
def find_similarity_threshold(cfg, rsync_w, df_dataset_id, hospital) -> None:
    row = _get_dataset_row_from_df(df_dataset_id, hospital, 'real', cfg.data.fold)
    dataset_name = row['dataset_name']
    dataset_path = Path(nnUNet_preprocessed) / dataset_name
    rsync_w.download(dataset_path, if_dir=True)

    data_root = Path(cfg.path.data)
    png_dataset_name = dataset_name + '_png'
    png_path = data_root / png_dataset_name
    rsync_w.download(png_path, if_dir=True)

    split_fn = get_split_fn(cfg.data.dataset_descriptor)
    part0_paths, part1_paths = split_fn(dataset_name, png_path)

    index_dir = data_root / (png_dataset_name + '_index')
    create_index_from_paths(part0_paths, index_dir)

    index_search_dir = data_root / (png_dataset_name + '_index_search')
    search_paths_in_index(part1_paths, index_dir, index_search_dir)

    histogram_closest(index_search_dir, 'real', 'real')
    threshold = percentile_dist_from_index(index_search_dir)

    yaml.safe_dump({'threshold': threshold}, open(dataset_path / 'threshold.yaml', 'w'))

    rsync_w.upload(index_dir, if_dir=True)
    rsync_w.upload(index_search_dir, if_dir=True)
    rsync_w.upload(dataset_path / 'threshold.yaml')


@ray.remote(num_gpus=0.5)
def filter_generated_images(cfg, rsync_w, df_dataset_id, hospital, exp_dir_syn_cur) -> None:
    row = _get_dataset_row_from_df(df_dataset_id, hospital, 'real', cfg.data.fold)
    dataset_name = row['dataset_name']
    dataset_path = Path(nnUNet_preprocessed) / dataset_name
    rsync_w.download(dataset_path, if_dir=True)

    data_root = Path(cfg.path.data)
    png_dataset_name = dataset_name + '_png'
    png_path = data_root / png_dataset_name
    rsync_w.download(png_path, if_dir=True)
    png_images_paths = list(png_path.rglob('*.png'))

    generated_path = exp_dir_syn_cur / 'generated'
    rsync_w.download(generated_path, if_dir=True)
    generated_images_paths = list(generated_path.rglob('*.png'))

    index_dir = exp_dir_syn_cur / ('generated' + '_index')
    create_index_from_paths(png_images_paths, index_dir)

    index_search_dir = exp_dir_syn_cur / ('generated' + '_index_search')
    search_paths_in_index(generated_images_paths, index_dir, index_search_dir)

    histogram_closest(index_search_dir, 'real', 'syn')
    threshold = yaml.safe_load(open(dataset_path / 'threshold.yaml', 'r'))['threshold']

    paths_below_threshold = paths_with_dist_below_threshold_from_index(index_search_dir, threshold)
    print(f'{len(paths_below_threshold)} images were below the threshold of {threshold:.4f}')

    generated_filtered_path = exp_dir_syn_cur / 'generated_filtered'
    if generated_filtered_path.exists():
        shutil.rmtree(generated_filtered_path)
    generated_filtered_path.mkdir(parents=True)

    for p in paths_below_threshold:
        shutil.move(p, generated_filtered_path)

    rsync_w.upload(index_dir, if_dir=True)
    rsync_w.upload(index_search_dir, if_dir=True)
    rsync_w.upload(generated_path, if_dir=True, if_delete=True)
    rsync_w.upload(generated_filtered_path, if_dir=True)


def _get_dataset_row_from_df(df_dataset_id, hospital, real, fold):
    return df_dataset_id[(df_dataset_id['hosp'] == hospital)
                         & (df_dataset_id['fold'] == fold)
                         & (df_dataset_id['real'] == real)
                         ].iloc[0]


@ray.remote(num_gpus=1)
def generate_images_ray(rsync_w, gan_path, n_images, out_dir, seed):
    set_random_seeds(seed)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rsync_w.download(str(gan_path.absolute()))

    seeds = list(range(n_images))

    generate_images(None, str(gan_path.absolute()), seeds, 1, 'const',
                    str(out_dir.absolute()), None, None)#, False)

    rsync_w.upload(str(out_dir.absolute()), if_dir=True)



@ray.remote(num_gpus=1)
def prepare_dataset_for_syn_nnunet(cfg, rsync_w, df_dataset_id, hospital, in_path_pngs):
    n_slices_per_case = 1 if cfg.data.original_data_was_png else 50

    row_real = _get_dataset_row_from_df(df_dataset_id, hospital, 'real', cfg.data.fold)
    dataset_id_real, dataset_name_real = row_real['dataset_id'], row_real['dataset_name']

    row_syn = _get_dataset_row_from_df(df_dataset_id, hospital, 'syn', cfg.data.fold)
    dataset_id_syn, dataset_name_syn = row_syn['dataset_id'], row_syn['dataset_name']

    rsync_w.download(in_path_pngs, if_dir=True)
    rsync_w.download(Path(nnUNet_preprocessed) / dataset_name_real, if_dir=True)
    rsync_w.download(Path(nnUNet_results) / dataset_name_real, if_dir=True)

    hsh = str(abs(hash((in_path_pngs, hospital))))
    print(f'{tempfile.gettempdir()=}')
    tmp_path = Path(tempfile.gettempdir())
    if (tmp_path / 'redirect_tmp').exists():
        tmp_path = tmp_path / 'redirect_tmp'
    out_path_npzs = tmp_path / f'pred_{hsh}'
    if out_path_npzs.exists():
        shutil.rmtree(out_path_npzs)
    out_path_npzs.mkdir(parents=True)

    now = datetime.now()
    _segment_ray(in_path_pngs, dataset_name_real, out_path_npzs, n_slices_per_case)
    print(f'Segmented images in {datetime.now() - now}')

    now = datetime.now()
    _create_syn_mod_of_nnunet_dataset(out_path_npzs, dataset_id_real, dataset_id_syn,
                                      cfg.data.dataset_descriptor, n_slices_per_case)
    print(f'Created syn mode of nnunet dataset in {datetime.now() - now}')

    shutil.rmtree(out_path_npzs)
    rsync_w.upload(Path(nnUNet_preprocessed) / dataset_name_syn, if_dir=True)


def _segment_ray(in_path_pngs, dataset_name_segmentor, out_path_npzs, n_slices_per_case) -> None:
    '''
    follow the procedure in nnUNetTrainer->perform_actual_validation
    '''
    if out_path_npzs.exists():
        shutil.rmtree(out_path_npzs)
    out_path_npzs.mkdir(parents=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    results_dir = join(nnUNet_results, dataset_name_segmentor, 'nnUNetTrainer__nnUNetPlans__2d')
    predictor.initialize_from_trained_model_folder(
        results_dir,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    predictor.network.load_state_dict(predictor.list_of_parameters[0]) # it doesn't load the parameters into the model otherwise

    png_paths = list(in_path_pngs.glob('*.png'))
    n_generated = len(png_paths)
    n_cases = n_generated // n_slices_per_case
    for i_case in range(n_cases):
        case_cur_data = []
        while len(case_cur_data) < n_slices_per_case:
            im_path = png_paths.pop()
            im = Image.open(im_path)
            im_np = np.array(im)
            if len(im_np.shape) == 2:
                im_np = im_np[None, ...]  # CHW
            else:
                im_np = im_np.transpose((2, 0, 1))  # CHW
            im_np = im_np[:, None, ...]  # CDHW
            case_cur_data.append(im_np)

        case_cur_data = np.concatenate(case_cur_data, axis=1)
        case_cur_data = (case_cur_data - case_cur_data.mean()) / case_cur_data.std()

        im_torch = torch.from_numpy(case_cur_data).half()

        prediction = predictor.predict_sliding_window_return_logits(im_torch)
        predicted_logits = prediction.cpu()

        label_manager = predictor.label_manager
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities).cpu().numpy()

        np.savez(out_path_npzs / f'{i_case}.npz',
                 data=im_torch.cpu().numpy().astype('float32'),
                 seg=segmentation.astype('int8')[None, ...])


def _create_syn_mod_of_nnunet_dataset(in_path_npzs, dataset_id_source, dataset_id_new, dataset_descriptor, n_slices_per_case):
    path_preprocessed = Path(nnUNet_preprocessed)
    name_orig = f'Dataset{dataset_id_source}_{dataset_descriptor}'
    path_orig = path_preprocessed / name_orig
    name_out = f'Dataset{dataset_id_new}_{dataset_descriptor}'
    path_out = path_preprocessed / name_out
    if path_out.exists():
        shutil.rmtree(path_out)
    path_out.mkdir(parents=True)

    dataset_json = json.load(open(path_orig / 'dataset.json', 'r'))
    foreground_labels = [v for k, v in dataset_json['labels'].items() if k != 'background']

    path_out_2d = path_out / 'nnUNetPlans_2d'
    path_out_2d.mkdir()

    pkl_reference = pickle.load(open(list(path_orig.glob('nnUNetPlans_2d/*.pkl'))[0], 'rb'))

    n_cases = len(list(in_path_npzs.glob('*.npz')))
    all_cases = []
    for i_case in range(n_cases):
        case_path = in_path_npzs / f'{i_case}.npz'
        case_cur = np.load(case_path)
        case_name = f'{dataset_descriptor}_{i_case + 1:03d}.npz'
        np.savez(path_out_2d / case_name, **case_cur)
        all_cases.append(case_name.split('.')[0])

        pkl_cur = copy.deepcopy(pkl_reference)
        pkl_cur['shape_before_cropping'] = (n_slices_per_case, *pkl_cur['shape_before_cropping'][1:])
        pkl_cur['bbox_used_for_cropping'][0][1] = n_slices_per_case
        pkl_cur['shape_after_cropping_and_before_resampling'] = (n_slices_per_case, *pkl_cur['shape_after_cropping_and_before_resampling'][1:])
        pkl_cur['class_locations'] = DefaultPreprocessor._sample_foreground_locations(case_cur['seg'], foreground_labels)
        pickle.dump(pkl_cur, open(path_out_2d / f'{dataset_descriptor}_{i_case + 1:03d}.pkl', 'wb'))


    n_val = min(int(0.2 * n_cases), 2500 // n_slices_per_case) # no sense in having too large a validation set, it will only slow down training
    n_train = n_cases - n_val
    # no need to shuffle because these cases are just randomly synthesized slices
    train_cases = all_cases[:n_train]
    val_cases = all_cases[n_train:]
    splits_final = [{'train': train_cases, 'val': val_cases}]
    json.dump(splits_final, open(path_out / 'splits_final.json', 'w'), indent=2)

    dataset_json['name'] = name_out
    dataset_json['numTest'] = 0
    dataset_json['numTraining'] = n_cases
    json.dump(dataset_json, open(path_out / 'dataset.json', 'w'), indent=2)

    shutil.copy(path_orig / 'dataset_fingerprint.json', path_out / 'dataset_fingerprint.json')

    nnunet_plans_json = json.load(open(path_orig / 'nnUNetPlans.json', 'r'))
    nnunet_plans_json['dataset_name'] = name_out
    json.dump(nnunet_plans_json, open(path_out / 'nnUNetPlans.json', 'w'), indent=2)


def merge_syn_nnunet_datasets(cfg, rsync_w, df_dataset_id, hospitals_except_all):
    # get ids, names & download.
    dataset_descriptor = cfg.data.dataset_descriptor
    row_real_all = _get_dataset_row_from_df(df_dataset_id, 'all', 'real', cfg.data.fold)
    id_real_all, name_real_all = row_real_all['dataset_id'], row_real_all['dataset_name']

    row_syn_all = _get_dataset_row_from_df(df_dataset_id, 'all', 'syn', cfg.data.fold)
    id_syn_all, name_syn_all = row_syn_all['dataset_id'], row_syn_all['dataset_name']

    rsync_w.download(Path(nnUNet_preprocessed) / name_real_all, if_dir=True)
    ids_syn_individ, names_syn_individ = [], []
    for h in hospitals_except_all:
        row = _get_dataset_row_from_df(df_dataset_id, h, 'syn', cfg.data.fold)
        ids_syn_individ.append(row['dataset_id'])
        names_syn_individ.append(row['dataset_name'])
        rsync_w.download(Path(nnUNet_preprocessed) / row['dataset_name'], if_dir=True)

    # create dirs
    path_preprocessed = Path(nnUNet_preprocessed)
    path_orig = path_preprocessed / name_real_all
    path_out = path_preprocessed / name_syn_all
    if path_out.exists():
        shutil.rmtree(path_out)
    path_out.mkdir(parents=True)
    path_out_2d = path_out / 'nnUNetPlans_2d'
    path_out_2d.mkdir()

    dataset_json = json.load(open(path_orig / 'dataset.json', 'r'))
    '''
    in practice, we will not have access to real_all, but it will still be possible to construct
    its parameters. Here, for convenience, just use it.
    '''

    i_case_out = 0
    train_cases, val_cases = [], []

    for name in names_syn_individ:
        path_cur = path_preprocessed / name
        n_cases = len(list(path_cur.glob('nnUNetPlans_2d/*.npz')))
        old_to_new_name = {}
        for i_case in range(n_cases):
            old_name = f'{dataset_descriptor}_{i_case + 1:03d}'
            case_path = path_cur / 'nnUNetPlans_2d' / old_name

            new_case_name = f'{dataset_descriptor}_{i_case_out + 1:03d}'
            shutil.copy(str(case_path.absolute()) + '.npz', path_out_2d / (new_case_name + '.npz'))
            shutil.copy(str(case_path.absolute()) + '.pkl', path_out_2d / (new_case_name + '.pkl'))

            old_to_new_name[old_name] = new_case_name
            i_case_out += 1
        splits_old = json.load(open(path_cur / 'splits_final.json', 'r'))
        train_cases.extend([old_to_new_name[case] for case in splits_old[0]['train']])
        val_cases.extend([old_to_new_name[case] for case in splits_old[0]['val']])

    splits_final = [{'train': train_cases, 'val': val_cases}]
    json.dump(splits_final, open(path_out / 'splits_final.json', 'w'), indent=2)

    dataset_json['name'] = name_syn_all
    dataset_json['numTest'] = 0
    dataset_json['numTraining'] = len(train_cases) + len(val_cases)
    json.dump(dataset_json, open(path_out / 'dataset.json', 'w'), indent=2)

    shutil.copy(path_orig / 'dataset_fingerprint.json', path_out / 'dataset_fingerprint.json')

    nnunet_plans_json = json.load(open(path_orig / 'nnUNetPlans.json', 'r'))
    nnunet_plans_json['dataset_name'] = name_syn_all
    json.dump(nnunet_plans_json, open(path_out / 'nnUNetPlans.json', 'w'), indent=2)

    rsync_w.upload(path_out, if_dir=True)

@ray.remote
def create_syn_real_nnunet_dataset(cfg, rsync_w, df_dataset_id, hospital):
    # get ids, names & download.
    dataset_descriptor = cfg.data.dataset_descriptor
    row_real_h = _get_dataset_row_from_df(df_dataset_id, hospital, 'real', cfg.data.fold)
    id_real_h, name_real_h = row_real_h['dataset_id'], row_real_h['dataset_name']

    row_syn_all = _get_dataset_row_from_df(df_dataset_id, 'all', 'syn', cfg.data.fold)
    id_syn_all, name_syn_all = row_syn_all['dataset_id'], row_syn_all['dataset_name']

    row_syn_real_h = _get_dataset_row_from_df(df_dataset_id, hospital, 'syn-real', cfg.data.fold)
    id_syn_real_h, name_syn_real_h = row_syn_real_h['dataset_id'], row_syn_real_h['dataset_name']

    rsync_w.download(Path(nnUNet_preprocessed) / name_syn_all, if_dir=True)
    rsync_w.download(Path(nnUNet_preprocessed) / name_real_h, if_dir=True)

    path_results = Path(nnUNet_results)
    rsync_w.download(path_results / name_syn_all, if_dir=True)

    # create dir
    path_preprocessed = Path(nnUNet_preprocessed)
    path_real_h = path_preprocessed / name_real_h
    path_syn_all = path_preprocessed / name_syn_all
    path_syn_real_h = path_preprocessed / name_syn_real_h
    if path_syn_real_h.exists():
        shutil.rmtree(path_syn_real_h)
    shutil.copytree(path_real_h, path_syn_real_h)

    # create updated nnUNetPlans.json: everything from real_h, except for configurations, which are from syn_all
    # (to be able to use pretrained weights)
    os.remove(path_syn_real_h / 'nnUNetPlans.json')
    nnUNetPlans_real_h = json.load(open(path_real_h / 'nnUNetPlans.json', 'r'))
    nnUNetPlans_syn_all = json.load(open(path_syn_all / 'nnUNetPlans.json', 'r'))

    nnUNetPlans_syn_real_h = copy.deepcopy(nnUNetPlans_real_h)
    nnUNetPlans_syn_real_h['dataset_name'] = name_syn_real_h
    nnUNetPlans_syn_real_h['configurations'] = nnUNetPlans_syn_all['configurations']

    json.dump(nnUNetPlans_syn_real_h, open(path_syn_real_h / 'nnUNetPlans.json', 'w'), indent=2)

    dataset_json = json.load(open(path_syn_real_h / 'dataset.json', 'r'))
    dataset_json['name'] = name_syn_real_h
    json.dump(dataset_json, open(path_syn_real_h / 'dataset.json', 'w'), indent=2)

    # copy checkpoint to the 'preprocessed' dir of the new dataset, because the 'results' dir for it
    # doesn't exist yet and because this makes my life easier
    shutil.copy(path_results / name_syn_all / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0' / 'checkpoint_best.pth',
                path_syn_real_h / 'checkpoint_best.pth')

    rsync_w.upload(path_syn_real_h, if_dir=True)


@ray.remote(num_gpus=1)
def train_nnunet(cfg, rsync_w, df_dataset_id,
                 hospital, real,
                 exp_dir_cur,
                 rsync_more_dirs=(),
                 use_pretrained_syn_all_weights=False # they will be provided in the preprocessed dir
                 ) -> None:
    exp_dir_cur.mkdir(exist_ok=True, parents=True)
    row = _get_dataset_row_from_df(df_dataset_id, hospital, real, cfg.data.fold)
    dataset_id = str(row['dataset_id'])
    nnunet_dataset_name = str(row['dataset_name'])
    rsync_w.download(join(nnUNet_preprocessed, nnunet_dataset_name), if_dir=True)
    for d in rsync_more_dirs:
        rsync_w.download(d, if_dir=True)
    fold_nnunet = 0  # always 0; my folds correspond to different nnunet datasets, each with one inner fold
    if use_pretrained_syn_all_weights:
        weights_path = join(nnUNet_preprocessed, nnunet_dataset_name, 'checkpoint_best.pth')
        schedule = cfg.train.get('nnunet_schedule_finetune', 'poly')
    else:
        weights_path = None
        schedule = 'poly'
    run_training(dataset_id, '2d', fold_nnunet, 'nnUNetTrainer',
                 'nnUNetPlans', weights_path, 1, False,
                 False, False, False,
                 False, True,
                 torch.device('cuda', 0),
                 cfg.train.get('nnunet_epochs', 1000),
                 schedule,
                 )
    # remove .npy files from preprocessed
    for p in Path(join(nnUNet_preprocessed, nnunet_dataset_name)).rglob('*.npy'):
        os.remove(p)
    rsync_w.upload(join(nnUNet_results, nnunet_dataset_name), if_dir=True)

    # copy summary.json to the exp dir just in case
    val_summary_path = join(nnUNet_results, nnunet_dataset_name, 'nnUNetTrainer__nnUNetPlans__2d', 'fold_0', 'validation',
             'summary.json')
    if Path(val_summary_path).exists():
        shutil.copy(val_summary_path, join(exp_dir_cur, 'summary_val.json'))
    rsync_w.upload(exp_dir_cur, if_dir=True)


@ray.remote(num_gpus=1)
def test_nnunet(cfg, rsync_w, df_dataset_id,
                hospital, real,
                hospital_target, # target is always real
                exp_dir_cur) -> None:
    row_for_hosp_and_fold = _get_dataset_row_from_df(df_dataset_id, hospital, real, cfg.data.fold)
    dataset_id_this = str(row_for_hosp_and_fold['dataset_id'])
    dataset_name_this = str(row_for_hosp_and_fold['dataset_name'])

    row_for_hosp_and_fold = _get_dataset_row_from_df(df_dataset_id, hospital_target, 'real', cfg.data.fold)
    dataset_id_target = str(row_for_hosp_and_fold['dataset_id'])
    dataset_name_target = str(row_for_hosp_and_fold['dataset_name'])

    rsync_w.download(join(nnUNet_results, dataset_name_this), if_dir=True)
    rsync_w.download(join(nnUNet_raw, dataset_name_target), if_dir=True)
    rsync_w.download(exp_dir_cur, if_dir=True)

    hsh = str(abs(hash((cfg, hospital, hospital_target, exp_dir_cur))))
    tmp_pred_store_path = Path(tempfile.gettempdir()) / f'pred_{hsh}'
    if tmp_pred_store_path.exists():
        shutil.rmtree(tmp_pred_store_path)
    tmp_pred_store_path.mkdir(parents=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    results_dir = join(nnUNet_results, dataset_name_this, 'nnUNetTrainer__nnUNetPlans__2d')
    predictor.initialize_from_trained_model_folder(
        results_dir,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    test_images_path = join(nnUNet_raw, dataset_name_target, 'imagesTs')
    test_labels_path = join(nnUNet_raw, dataset_name_target, 'labelsTs')
    predictor.predict_from_files(test_images_path,
                                 str(tmp_pred_store_path.absolute()),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    print('#### Predictions stored ####')

    # get labels from dataset.json
    dataset_json = join(nnUNet_raw, dataset_name_target, 'dataset.json')
    label_indices_non_bg = [name_and_index[1] for name_and_index in json.load(
                                open(dataset_json, 'r'))['labels'].items()
                            if name_and_index[0] != 'background']
    summary_path = join(exp_dir_cur, f'summary_{hsh}.json')
    compute_metrics_on_folder_simple(
        test_labels_path,
        str(tmp_pred_store_path.absolute()),
        label_indices_non_bg,
        summary_path
    )

    with open(summary_path) as f:
        summary = json.load(f)

    phase_metrics = {
        'dice': {
            'avg': summary['foreground_mean']['Dice'],
        },
        'hd95': {
            'avg': summary['foreground_mean']['HD95'],
        }
    }
    for label_id, values in summary['mean'].items():
        phase_metrics['dice'][label_id] = values['Dice']
        phase_metrics['hd95'][label_id] = values['HD95']

    shutil.rmtree(tmp_pred_store_path)

    # save as yaml
    with open(exp_dir_cur / f'info_evaluate_{hospital_target}.yml', 'w') as f:
        info = {
            'test': phase_metrics,
            'hospital': hospital_target
        }
        yaml.safe_dump(info, f)
    rsync_w.upload(str((exp_dir_cur / f'info_evaluate_{hospital_target}.yml').absolute()), if_dir=False)


@hydra.main(version_base=None, config_path="config/nnunet", config_name="cervix_debug0")
def main(cfg: DictConfig) -> None:
    cv2.setNumThreads(0)
    ray.init(address=cfg.general.ray_address, _temp_dir='/export/scratch1/home/aleksand/s2/tmp/ray')

    rsync_w = RsyncWrapper(cfg.general.ssh_user, cfg.general.ray_head_node,
                           cfg.general.if_shared_fs, cfg.general.final_upload_node)

    exp_dir = prepare_dir(cfg)
    setup_logging(exp_dir / '_log.txt')

    hospitals = list(cfg.skip.real.keys())
    hospitals_except_all = [h for h in hospitals if h != 'all']

    seed = cfg.general.seed_base + cfg.data.fold * 1000
    def next_seed():
        nonlocal seed
        seed += 1
        return seed

    seeds = {}
    # seeds need to be the same for both real & fake data:
    seeds['appliedUnet'] = {h: next_seed() for h in hospitals} #  nnunet uses a non-deterministic dataloader, making seeding meaningless.
    # in general, seeds should be fixed so that if only a part of a pipeline is run, it will still have a fixed seed
    seeds['GAN'] = {h: next_seed() for h in hospitals_except_all}
    seeds['generate'] = {h: next_seed() for h in hospitals_except_all}
    seeds['segment_generated'] = {h: next_seed() for h in hospitals_except_all}

    exp_dirs = {}
    for h in hospitals:
        exp_dirs[h] = exp_dir / h
        (exp_dir / h).mkdir(exist_ok=True)

    futures = defaultdict(list)

    df_dataset_id = pd.DataFrame(yaml.safe_load(
        open(Path(cfg.path.data) / cfg.data.base_dataset / 'df_dataset_id.yaml', 'r')))

    print('0##| prepare_data_for_gan')
    for h in hospitals_except_all:
        if not cfg.skip.syn[h]['prepare_data_for_gan']:
            f = prepare_dataset_for_gan.remote(cfg, rsync_w, df_dataset_id, h)
            futures['prepare_data_for_gan'].append(f)

    print('#0#| get prepare_data_for_gan')
    for f in futures['prepare_data_for_gan']:
        ray.get(f)
    print('##0| got prepare_data_for_gan')

    print('1##| determine thresholds')
    for h in hospitals_except_all:
        if not cfg.skip.syn[h]['determine_threshold']:
            f = find_similarity_threshold.remote(cfg, rsync_w, df_dataset_id, h)
            futures['determine_threshold'].append(f)

    print('#1#| get determine_threshold')
    for f in futures['determine_threshold']:
        ray.get(f)
    print('##1| got determine_threshold')

    print('2##| GANs')
    for h in hospitals_except_all:
        if not cfg.skip.syn[h]['gan']:
            n_gpus = 1
            f = train_gan_ray.options(**{'num_gpus': n_gpus}).remote(exp_dirs[h] / 'syn',
                                     cfg, rsync_w,
                                     df_dataset_id, h,
                                     seeds['GAN'][h], n_gpus)
            futures['gan'].append(f)


    print('3##| U-Net-real')
    for h in hospitals:
        if not cfg.skip.real[h].applied_unet_train:
            f = train_nnunet.options(**{'num_cpus': cfg.train.get('num_cpus_real', 1)}).remote(cfg, rsync_w, df_dataset_id, h, 'real', exp_dirs[h] / 'real' / 'appliedUnet')
            futures['unet_real'].append(f)

    print('#2#| get GANs')
    for f in futures['gan']:
        ray.get(f)
    print('##2| got GANs')

    print('4##| generate images')
    for h in hospitals_except_all:
        if not cfg.skip.syn[h]['generate']:
            n_to_generate = cfg.generate.n_images
            if n_to_generate == 'auto':
                row = _get_dataset_row_from_df(df_dataset_id, h, 'real', cfg.data.fold)
                nnunet_dataset_name = row['dataset_name']
                in_dir = Path(cfg.path.data) / (nnunet_dataset_name + '_png')
                n_to_generate = len(list(filter(lambda x: 'mask' not in x.name,
                                                     in_dir.rglob('*.png'))))
                n_to_generate *= 10  # 10 times more than the number of real images
                n_to_generate = n_to_generate // 100 * 100  # make divisible by 100

            f = generate_images_ray.remote(rsync_w,
                                exp_dirs[h] / 'syn' / 'gan' / 'network-snapshot-best.pkl',
                                n_to_generate,
                                exp_dirs[h] / 'syn' / 'generated', seeds['generate'][h])

            futures['generate'].append(f)


    print('#4#| get generate images')
    for f in futures['generate']:
        ray.get(f)
    print('##4| got generate images')

    print('5##| filter generated')
    for h in hospitals_except_all:
        if not cfg.skip.syn[h]['filter_generated']:
            f = filter_generated_images.remote(cfg, rsync_w, df_dataset_id, h,
                                               exp_dirs[h] / 'syn')
            futures['filter_generated'].append(f)

    print('#5#| get filter generated')
    for f in futures['filter_generated']:
        ray.get(f)
    print('##5| got filter generated')

    print('#3#| get U-Net-real')
    for f in futures['unet_real']:
        ray.get(f)
    print('##3| got U-Net-real')

    print('6##| segment generated images & create a nnunet dataset')
    for h in hospitals_except_all:
        if not cfg.skip['syn'][h]['segment']:
            f = prepare_dataset_for_syn_nnunet.remote(cfg, rsync_w, df_dataset_id, h,
                                                      exp_dirs[h] / 'syn' / 'generated')

            futures['segment_generated'].append(f)

    print('#6#| get segmentations')
    for f in futures['segment_generated']:
        ray.get(f)
    print('##6| got segmentations')

    print('7##| merge syn datasets')
    if not cfg.skip.syn['all']['merge']:
        merge_syn_nnunet_datasets(cfg, rsync_w, df_dataset_id, hospitals_except_all)
    print('##7| merged syn datasets')

    print('8##| U-Net-syn')
    for h in hospitals:
        if not cfg.skip.syn[h].applied_unet_train:
            f = train_nnunet.options(**{'num_cpus': cfg.train.get('num_cpus_syn', 1)}).remote(cfg, rsync_w,
                                    df_dataset_id, h, 'syn',
                                    exp_dirs[h] / 'syn' / 'appliedUnet')#, rsync_more_dirs)
            futures['unet_syn'].append(f)

    print('#8#| get U-Net-syn')
    for f in futures['unet_syn']:
        ray.get(f)
    print('##8| got U-Net-syn')

    print('9##| create_syn_real_nnunet_dataset')
    for h in hospitals_except_all:
        if not cfg.skip['syn-real'][h]['prepare_data_syn_real']:
            f = create_syn_real_nnunet_dataset.remote(cfg, rsync_w, df_dataset_id, h)
            futures['unet_syn_real'].append(f)

    print('#9#| get create_syn_real_nnunet_dataset')
    for f in futures['unet_syn_real']:
        ray.get(f)
    print('##9| got create_syn_real_nnunet_dataset')

    print('10##| U-Net-real-from-syn-pretrain')
    for h in hospitals_except_all:
        if not cfg.skip['syn-real'][h]['applied_unet_from_syn_pretrain']:
            f = train_nnunet.options(**{'num_cpus': cfg.train.num_cpus_syn_real}).remote(cfg, rsync_w, df_dataset_id, h, 'syn-real',
                                    exp_dirs[h] / 'real' / 'appliedUnet_from_syn_pretrain',
                                    use_pretrained_syn_all_weights=True)
            futures['unet_real_from_syn_pretrain'].append(f)

    print('#10#| get U-Net-real-from-syn-pretrain')
    for f in futures['unet_real_from_syn_pretrain']:
        ray.get(f)
    print('##10| got U-Net-real-from-syn-pretrain')

    print('11##| eval')
    for h in hospitals:
        for h_t in hospitals_except_all:  # (target)
            modes_list = ['real', 'syn', 'syn-real']
            if h == 'all':
                modes_list.remove('syn-real')
            for mode in modes_list:
                if not cfg.skip[mode][h]['eval'][h_t]:
                    mode_dir = {'real': 'real', 'syn': 'syn', 'syn-real': 'real'}[mode]
                    model_dir = 'appliedUnet' if mode != 'syn-real' else 'appliedUnet_from_syn_pretrain'
                    dir_path = exp_dirs[h] / mode_dir
                    dir_path.mkdir(exist_ok=True)
                    dir_path = dir_path / model_dir
                    dir_path.mkdir(exist_ok=True)
                    f = test_nnunet.remote(cfg, rsync_w, df_dataset_id,
                                           h, mode, h_t,
                                           dir_path)
                    futures['eval'].append(f)

    print('#11#| get eval')
    for f in futures['eval']:
        ray.get(f)
    print('##11| got eval')

    print('upload_final')
    rsync_w.upload_final(str(exp_dir.absolute()), if_dir=True)
    for h in hospitals:
        for real in ['real', 'syn', 'syn-real']:
            if h == 'all' and real == 'syn-real':
                continue
            try:
                row = _get_dataset_row_from_df(df_dataset_id, h, real, cfg.data.fold)
            except:
                continue
            nnunet_dataset_name = row['dataset_name']
            if os.path.exists(join(nnUNet_preprocessed, nnunet_dataset_name)):
                rsync_w.upload(join(nnUNet_preprocessed, nnunet_dataset_name), if_dir=True)
            if os.path.exists(join(nnUNet_results, nnunet_dataset_name)):
                rsync_w.upload(join(nnUNet_results, nnunet_dataset_name), if_dir=True)

    print('Success', datetime.now().strftime('%H:%M:%S'))

def prepare_dir(cfg):
    exp_dir = Path(cfg.path.logs) / cfg.general.exp_name
    if 'copy_from' in cfg.general:
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_dir = exp_dir / f'fold_{cfg.data.fold}'
        if exp_dir.exists():
            shutil.rmtree(str(exp_dir.absolute()))
        exp_dir.mkdir(parents=True)

        copy_from = Path(cfg.path.logs) / cfg.general.copy_from / f'fold_{cfg.data.fold}'
        print(f'Copying from {copy_from} to {exp_dir}')
        if not cfg.general.get('copy_only_gans', False):
            shutil.copytree(str(copy_from.absolute()), str(exp_dir.absolute()), dirs_exist_ok=True)
        else:
            for h in cfg.general.hospitals:
                shutil.copytree(str((copy_from / h / 'syn' / 'gan').absolute()),
                                str((exp_dir / h / 'syn' / 'gan').absolute()), dirs_exist_ok=True)
                shutil.copytree(str((copy_from / h / 'syn' / 'generated').absolute()),
                                str((exp_dir / h / 'syn' / 'generated').absolute()), dirs_exist_ok=True)

        print('Removing all tensorboard files')
        event_files = exp_dir.rglob('events.*')
        for f in event_files:
            f.unlink()
    else:
        exp_dir = exp_dir / f'fold_{cfg.data.fold}'
        exp_dir.mkdir(exist_ok=True, parents=True)
    return exp_dir

if __name__ == '__main__':
    main()
