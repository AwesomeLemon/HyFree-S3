import os
import copy

import numpy as np

import json
import shutil
import tempfile
from datetime import datetime
from os.path import join

from pathlib import Path

import cv2
import hydra
import ray
import torch
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import yaml
from nnunetv2.run.run_training import run_training

from nnunetv2_mod.nnunetv2.run.run_training import get_trainer_from_args
from run_nnunet import _get_dataset_row_from_df
from utils.general import setup_logging, set_random_seeds, resize_images
from utils.rsync_wrapper import RsyncWrapper

def create_init_checkpoint(cfg, rsync_w, df_dataset_id,
                 hospital, real,
                 exp_dir_cur):
    exp_dir_cur.mkdir(exist_ok=True, parents=True)
    row = _get_dataset_row_from_df(df_dataset_id, hospital, real, cfg.data.fold)
    dataset_id = str(row['dataset_id'])
    nnunet_dataset_name = str(row['dataset_name'])
    rsync_w.download(join(nnUNet_preprocessed, nnunet_dataset_name), if_dir=True)
    nnunet_trainer = get_trainer_from_args(dataset_id, '2d', 0, 'nnUNetTrainer',
                              'nnUNetPlans', False, device=torch.device('cpu'),
                                           num_epochs=cfg.train.get('nnunet_epochs', 1000),
                                           schedule_name='poly', continue_for=0)
    nnunet_trainer.initialize()
    nnunet_trainer.current_epoch = -1 # save_checkpoint adds 1 automatically

    nnunet_trainer.save_checkpoint(join(exp_dir_cur, 'checkpoint_init.pth'))

@ray.remote(num_gpus=1)
def train_nnunet(cfg, rsync_w, df_dataset_id,
                 hospital, real,
                 exp_dir_cur,
                 rsync_more_dirs=(),
                 pretrained_weights_path=None,
                 remove_npy=True,
                 if_continue=False,
                 continue_for=None
                 ) -> None:
    exp_dir_cur.mkdir(exist_ok=True, parents=True)
    row = _get_dataset_row_from_df(df_dataset_id, hospital, real, cfg.data.fold)
    dataset_id = str(row['dataset_id'])
    nnunet_dataset_name = str(row['dataset_name'])
    rsync_w.download(join(nnUNet_preprocessed, nnunet_dataset_name), if_dir=True)
    rsync_w.download(join(nnUNet_results, nnunet_dataset_name), if_dir=True, repeat_on_fail=False) # to continue training from the prev round, if available
    for d in rsync_more_dirs:
        rsync_w.download(d, if_dir=True)
    fold_nnunet = 0  # always 0; my folds correspond to different nnunet datasets, each with one inner fold
    schedule = 'poly'
    print(f'{if_continue=} {continue_for=}')
    run_training(dataset_id, '2d', fold_nnunet, 'nnUNetTrainer',
                 'nnUNetPlans', pretrained_weights_path, 1, False,
                 False, if_continue,
                 False,
                 False,
                 False, # validate with last, not best
                 torch.device('cuda', 0),
                 cfg.train.get('nnunet_epochs', 1000),
                 schedule,
                 continue_for
                 )
    # remove .npy files from preprocessed
    if remove_npy:
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
                results_dir,
                hospital_target, # target is always real
                ) -> None:
    row_for_hosp_and_fold = _get_dataset_row_from_df(df_dataset_id, hospital_target, 'real', cfg.data.fold)
    dataset_name_target = str(row_for_hosp_and_fold['dataset_name'])

    rsync_w.download(results_dir, if_dir=True)
    rsync_w.download(join(nnUNet_raw, dataset_name_target), if_dir=True)

    hsh = str(abs(hash((cfg, hospital_target, results_dir))))
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
    dataset_json = join(nnUNet_preprocessed, dataset_name_target, 'dataset.json')
    label_indices_non_bg = [name_and_index[1] for name_and_index in json.load(
                                open(dataset_json, 'r'))['labels'].items()
                            if name_and_index[0] != 'background']
    summary_path = join(results_dir, f'summary_{hsh}.json')
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
    with open(results_dir / f'info_evaluate_{hospital_target}.yml', 'w') as f:
        info = {
            'test': phase_metrics,
            'hospital': hospital_target
        }
        yaml.safe_dump(info, f)
    rsync_w.upload(str((results_dir / f'info_evaluate_{hospital_target}.yml').absolute()), if_dir=False)


@hydra.main(version_base=None, config_path="config/federated", config_name="cervix_00")
def main(cfg: DictConfig) -> None:
    cv2.setNumThreads(0)
    ray.init(address=cfg.general.ray_address, _temp_dir='/export/scratch1/home/aleksand/s2/tmp/ray')

    rsync_w = RsyncWrapper(cfg.general.ssh_user, cfg.general.ray_head_node,
                           cfg.general.if_shared_fs, cfg.general.final_upload_node)

    exp_dir = Path(cfg.path.logs) / cfg.general.exp_name
    exp_dir = exp_dir / f'fold_{cfg.data.fold}'
    exp_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(exp_dir / '_log.txt')

    hosps_except_all = ['A', 'B']
    hosps = hosps_except_all + ['all']
    exp_dirs = {}
    for h in hosps:
        exp_dirs[h] = exp_dir / h / 'real' / 'appliedUnet'
        exp_dirs[h].mkdir(exist_ok=True, parents=True)

    with open(exp_dir / 'cfg.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    df_dataset_id = pd.DataFrame(yaml.safe_load(
        open(Path(cfg.path.data) / cfg.data.base_dataset / 'df_dataset_id_federated.yaml', 'r')))

    # federated learning for 1 fold
    # need to have prepared datasets & df_dataset_id.yaml
    # in the main process, have a for loop for 1000 iterations
    # in each one, train local models in a ray function (starts with download, ends with upload),
    # then merge on the central node, and repeat

    epochs_total = cfg.train.nnunet_epochs
    epochs_step = cfg.train.nnunet_epochs_step

    if not cfg.skip.train:
        # start by creating a common init (using the A hospital dataset, it doesn't matter; but save in 'all')
        create_init_checkpoint(cfg, rsync_w, df_dataset_id, 'A', 'real', exp_dirs['all'])
        ema_fg_dice_best = -np.inf
        ema_fg_dice_history = []
        assert epochs_total % epochs_step == 0
        for round in range(epochs_total // epochs_step):
            print(f'Round {round+1}/{epochs_total // epochs_step}')
            if_last = round == epochs_total // epochs_step - 1
            futures = []
            if round == 0:
                pretrained_weights_path = exp_dirs['all'] / 'checkpoint_init.pth'
                rsync_more_dirs = [exp_dirs['all']]
                # potentially delete checkpoints from previous runs because nnunet tries to load _final before _latest
                for h in hosps_except_all:
                    row = _get_dataset_row_from_df(df_dataset_id, h, 'real', cfg.data.fold)
                    dataset_name = str(row['dataset_name'])
                    ckpt_cur_dir = Path(nnUNet_results) / dataset_name / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0'
                    for ckpt in ckpt_cur_dir.glob('checkpoint*'):
                        os.remove(ckpt)
            else:
                pretrained_weights_path = None
                rsync_more_dirs = []
            for h in hosps_except_all:
                f = train_nnunet.remote(cfg, rsync_w, df_dataset_id, h, 'real', exp_dirs[h],
                                        rsync_more_dirs, pretrained_weights_path,
                                        remove_npy=if_last,
                                        if_continue=round > 0, continue_for=epochs_step)
                futures.append(f)
            ray.get(futures)

            # merge, no need to download because this is the central node

            ckpt_name = 'checkpoint_latest.pth' if not if_last else 'checkpoint_final.pth'
            avg_state = None
            emas_fg_dice = []
            for h in hosps_except_all:
                dataset_name = _get_dataset_row_from_df(df_dataset_id, h, 'real', cfg.data.fold)['dataset_name']
                ckpt_cur = Path(nnUNet_results) / dataset_name / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0' / ckpt_name
                ckpt = torch.load(ckpt_cur, map_location='cpu')
                state = ckpt['network_weights']
                if avg_state is None:
                    avg_state = copy.deepcopy(state)
                else:
                    for k in avg_state.keys():
                        avg_state[k] += state[k]
                emas_fg_dice.append(ckpt['logging']['ema_fg_dice'][-1])
                del ckpt

            for k in avg_state.keys():
                avg_state[k] /= len(hosps_except_all)

            for h in hosps_except_all:
                dataset_name = _get_dataset_row_from_df(df_dataset_id, h, 'real', cfg.data.fold)['dataset_name']
                ckpt_cur = Path(nnUNet_results) / dataset_name / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0' / ckpt_name
                state = torch.load(ckpt_cur, map_location='cpu')
                state['network_weights'] = avg_state
                torch.save(state, ckpt_cur)

            ema_fg_dice = np.mean(emas_fg_dice).item()
            if ema_fg_dice > ema_fg_dice_best:
                ema_fg_dice_best = ema_fg_dice
                # copy any of the checkpoints, they all have the same weights
                dataset_name = _get_dataset_row_from_df(df_dataset_id, 'A', 'real', cfg.data.fold)['dataset_name']
                ckpt_cur = Path(nnUNet_results) / dataset_name / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0' / ckpt_name
                shutil.copy(ckpt_cur, exp_dirs['all'] / f'checkpoint_best.pth')

            ema_fg_dice_history.append(ema_fg_dice)
            yaml.safe_dump(ema_fg_dice_history, open(exp_dirs['all'] / 'ema_fg_dice_history.yaml', 'w'))

        # copy final checkpoint
        dataset_name = _get_dataset_row_from_df(df_dataset_id, 'A', 'real', cfg.data.fold)['dataset_name']
        ckpt_cur = Path(nnUNet_results) / dataset_name / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0' / 'checkpoint_final.pth'
        shutil.copy(ckpt_cur, exp_dirs['all'] / 'checkpoint_final.pth')

    if not cfg.skip.test:
        # prepare results_dir for test
        results_dir = exp_dirs['all']
        results_dir.mkdir(exist_ok=True, parents=True)
        # copy dataset.json, plans.json from A
        dataset_name = _get_dataset_row_from_df(df_dataset_id, 'A', 'real', cfg.data.fold)['dataset_name']
        shutil.copy(Path(nnUNet_results) / dataset_name / 'nnUNetTrainer__nnUNetPlans__2d' / 'dataset.json', results_dir)
        shutil.copy(Path(nnUNet_results) / dataset_name / 'nnUNetTrainer__nnUNetPlans__2d'/ 'plans.json', results_dir)
        # move checkpoints
        ckpt_subdir = results_dir / 'fold_0'
        ckpt_subdir.mkdir(exist_ok=True, parents=True)
        for ckpt in results_dir.glob('checkpoint*'):
            shutil.copy(ckpt, ckpt_subdir)

        # test
        futures = []
        for h in hosps_except_all:
            f = test_nnunet.remote(cfg, rsync_w, df_dataset_id, results_dir, h)
            futures.append(f)
        ray.get(futures)

    print('upload_final')
    rsync_w.upload_final(str(exp_dir.absolute()), if_dir=True)
    print('Success', datetime.now().strftime('%H:%M:%S'))

if __name__ == '__main__':
    main()
