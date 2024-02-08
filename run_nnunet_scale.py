import os
import copy
import json
import shutil
from collections import defaultdict

from datetime import datetime
from os.path import join

from pathlib import Path

import cv2
import hydra
import ray
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from omegaconf import DictConfig
import pandas as pd
import yaml

from run_nnunet import prepare_dir, _get_dataset_row_from_df, prepare_dataset_for_gan, find_similarity_threshold, \
    train_gan_ray, train_nnunet, generate_images_ray, filter_generated_images, prepare_dataset_for_syn_nnunet, \
    test_nnunet
from utils.general import setup_logging, set_random_seeds
from utils.rsync_wrapper import RsyncWrapper



def merge_syn_nnunet_datasets(cfg, rsync_w, df_dataset_id, hospitals_to_merge, hospital_target):
    # get ids, names & download.
    dataset_descriptor = cfg.data.dataset_descriptor
    name_real_target = _get_dataset_row_from_df(df_dataset_id, hospital_target, 'real', cfg.data.fold)['dataset_name']
    name_syn_target = _get_dataset_row_from_df(df_dataset_id, hospital_target, 'syn', cfg.data.fold)['dataset_name']

    rsync_w.download(Path(nnUNet_preprocessed) / name_real_target, if_dir=True)
    ids_syn_individ, names_syn_individ = [], []
    for h in hospitals_to_merge:
        row = _get_dataset_row_from_df(df_dataset_id, h, 'syn', cfg.data.fold)
        ids_syn_individ.append(row['dataset_id'])
        names_syn_individ.append(row['dataset_name'])
        rsync_w.download(Path(nnUNet_preprocessed) / row['dataset_name'], if_dir=True)

    path_preprocessed = Path(nnUNet_preprocessed)
    path_orig = path_preprocessed / name_real_target
    path_out = path_preprocessed / name_syn_target
    if path_out.exists():
        shutil.rmtree(path_out)
    path_out.mkdir(parents=True)
    path_out_2d = path_out / 'nnUNetPlans_2d'
    path_out_2d.mkdir()

    dataset_json = json.load(open(path_orig / 'dataset.json', 'r'))
    '''
    in practice, we will not have access to real_all, but it will still be possible to construct
    its parameters based on the statistics of the datasets. Here, for convenience, just use it.
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

    dataset_json['name'] = name_syn_target
    dataset_json['numTest'] = 0
    dataset_json['numTraining'] = len(train_cases) + len(val_cases)
    json.dump(dataset_json, open(path_out / 'dataset.json', 'w'), indent=2)

    shutil.copy(path_orig / 'dataset_fingerprint.json', path_out / 'dataset_fingerprint.json')

    nnunet_plans_json = json.load(open(path_orig / 'nnUNetPlans.json', 'r'))
    nnunet_plans_json['dataset_name'] = name_syn_target
    json.dump(nnunet_plans_json, open(path_out / 'nnUNetPlans.json', 'w'), indent=2)

    rsync_w.upload(path_out, if_dir=True)

@ray.remote
def create_syn_real_nnunet_dataset(cfg, rsync_w, df_dataset_id, hospital, syn_real_type, hospital_syn_pretrain):
    # get ids, names & download.
    dataset_descriptor = cfg.data.dataset_descriptor
    row_real_h = _get_dataset_row_from_df(df_dataset_id, hospital, 'real', cfg.data.fold)
    id_real_h, name_real_h = row_real_h['dataset_id'], row_real_h['dataset_name']

    name_syn_pretrain = _get_dataset_row_from_df(df_dataset_id, hospital_syn_pretrain, 'syn', cfg.data.fold)['dataset_name']

    row_syn_real_h = _get_dataset_row_from_df(df_dataset_id, hospital, syn_real_type, cfg.data.fold)
    id_syn_real_h, name_syn_real_h = row_syn_real_h['dataset_id'], row_syn_real_h['dataset_name']

    rsync_w.download(Path(nnUNet_preprocessed) / name_syn_pretrain, if_dir=True)
    rsync_w.download(Path(nnUNet_preprocessed) / name_real_h, if_dir=True)

    path_results = Path(nnUNet_results)
    rsync_w.download(path_results / name_syn_pretrain, if_dir=True)

    path_preprocessed = Path(nnUNet_preprocessed)
    path_real_h = path_preprocessed / name_real_h
    path_syn_pretrain = path_preprocessed / name_syn_pretrain
    path_syn_real_h = path_preprocessed / name_syn_real_h
    if path_syn_real_h.exists():
        shutil.rmtree(path_syn_real_h)
    shutil.copytree(path_real_h, path_syn_real_h)

    # create updated nnUNetPlans.json: everything from real_h, except for configurations, which are from syn_[AB|ABCD|all)
    # (to be able to use pretrained weights)
    os.remove(path_syn_real_h / 'nnUNetPlans.json')
    nnUNetPlans_real_h = json.load(open(path_real_h / 'nnUNetPlans.json', 'r'))
    nnUNetPlans_syn_all = json.load(open(path_syn_pretrain / 'nnUNetPlans.json', 'r'))

    nnUNetPlans_syn_real_h = copy.deepcopy(nnUNetPlans_real_h)
    nnUNetPlans_syn_real_h['dataset_name'] = name_syn_real_h
    nnUNetPlans_syn_real_h['configurations'] = nnUNetPlans_syn_all['configurations']

    json.dump(nnUNetPlans_syn_real_h, open(path_syn_real_h / 'nnUNetPlans.json', 'w'), indent=2)

    dataset_json = json.load(open(path_syn_real_h / 'dataset.json', 'r'))
    dataset_json['name'] = name_syn_real_h
    json.dump(dataset_json, open(path_syn_real_h / 'dataset.json', 'w'), indent=2)

    # copy checkpoint to the 'preprocessed' dir of the new dataset, because the 'results' dir for it
    # doesn't exist yet and because this makes my life easier
    shutil.copy(path_results / name_syn_pretrain / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0' / 'checkpoint_best.pth',
                path_syn_real_h / 'checkpoint_best.pth')

    rsync_w.upload(path_syn_real_h, if_dir=True)


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
    seeds['appliedUnet'] = {h: next_seed() for h in hospitals} #  nnunet by default uses a non-deterministic dataloader, making seeding meaningless.
    # in general, seeds should be fixed so that if only a part of a pipeline is run, it will still have a fixed seed
    seeds['GAN'] = {h: next_seed() for h in hospitals_except_all}
    seeds['generate'] = {h: next_seed() for h in hospitals_except_all}
    seeds['segment_generated'] = {h: next_seed() for h in hospitals_except_all}

    exp_dirs = {}
    for h in hospitals + ['AB', 'ABCD']:
        exp_dirs[h] = exp_dir / h
        (exp_dir / h).mkdir(exist_ok=True)

    futures = defaultdict(list)

    df_dataset_id = pd.DataFrame(yaml.safe_load(
        open(Path(cfg.path.data) / cfg.data.base_dataset / 'df_dataset_id_scaling.yaml', 'r')))

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
            n_gpus = 1#2 if h == 'A' else 1
            f = train_gan_ray.options(**{'num_gpus': n_gpus}).remote(exp_dirs[h] / 'syn',
                                     cfg, rsync_w,
                                     df_dataset_id, h,
                                     seeds['GAN'][h], n_gpus)
            # if h == 'A':
            #     time.sleep(300)
            #     print('AAAAAAAAAA, added sleeping')
            futures['gan'].append(f)

    # time.sleep(600) # wait for the GANs to start training

    print('3##| U-Net-real')
    for h in hospitals:
        if not cfg.skip.real[h].applied_unet_train:
            f = train_nnunet.remote(cfg, rsync_w, df_dataset_id, h, 'real', exp_dirs[h] / 'real' / 'appliedUnet')
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
                n_to_generate = n_to_generate // 100 * 100  # make divisible by 100
                n_to_generate *= 10  # 10 times more than the number of real images

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

    print('7##| merge syn AB, ABCD, all')
    for merge_target, hospitals_to_be_merged in zip(['AB', 'ABCD', 'all'], [['A', 'B'], ['A', 'B', 'C', 'D'], hospitals_except_all]):
        if not cfg.skip.syn[merge_target]['merge']:
            merge_syn_nnunet_datasets(cfg, rsync_w, df_dataset_id, hospitals_to_be_merged, merge_target)
    print('##7| merged syn AB, ABCD, all')

    print('8##| U-Net-syn for merged')
    for h in ['AB', 'ABCD', 'all']:
        if not cfg.skip.syn[h].applied_unet_train:
            f = train_nnunet.remote(cfg, rsync_w,
                                    df_dataset_id, h, 'syn',
                                    exp_dirs[h] / 'syn' / 'appliedUnet')
            futures['unet_syn'].append(f)

    print('#8#| get U-Net-syn for merged')
    for f in futures['unet_syn']:
        ray.get(f)
    print('##8| got U-Net-syn for merged')

    print('9##| create all syn_real nnunet datasets')
    for syn_real_type, merge_target in zip(['syn2-real', 'syn4-real', 'syn-real'], ['AB', 'ABCD', 'all']):
        for h in hospitals_except_all:
            if not cfg.skip[syn_real_type][h]['prepare_data_syn_real']:
                f = create_syn_real_nnunet_dataset.remote(cfg, rsync_w, df_dataset_id, h, syn_real_type, merge_target)
                futures['unet_syn_real'].append(f)

    print('#9#| get create all syn_real nnunet datasets')
    for f in futures['unet_syn_real']:
        ray.get(f)
    print('##9| got create all syn_real nnunet datasets')

    print('10##| U-Net-real-from-syn-pretrain')
    for syn_real_type, out_siffix in zip(['syn2-real', 'syn4-real', 'syn-real'], ['syn2_pretrain', 'syn4_pretrain', 'syn_pretrain']):
        for h in hospitals_except_all:
            if not cfg.skip[syn_real_type][h]['applied_unet_from_syn_pretrain']:
                f = train_nnunet.options(**{'num_cpus': cfg.train.num_cpus_syn_real}).remote(cfg, rsync_w, df_dataset_id, h, syn_real_type,
                                        exp_dirs[h] / 'real' / f'appliedUnet_from_{out_siffix}',
                                        use_pretrained_syn_all_weights=True)
                futures['unet_real_from_syn_pretrain'].append(f)

    print('#10#| get U-Net-real-from-syn-pretrain')
    for f in futures['unet_real_from_syn_pretrain']:
        ray.get(f)
    print('##10| got U-Net-real-from-syn-pretrain')

    print('11##| eval')
    for h in hospitals + ['AB', 'ABCD']:
        for h_t in hospitals_except_all:  # (target)
            modes_list = ['real', 'syn', 'syn-real', 'syn2-real', 'syn4-real']
            if h in ['all', 'AB', 'ABCD']:
                modes_list.remove('syn-real')
                modes_list.remove('syn2-real')
                modes_list.remove('syn4-real')
            if h in ['AB', 'ABCD']:
                modes_list.remove('real')
            for mode in modes_list:
                if not cfg.skip[mode][h]['eval'][h_t]:
                    mode_dir = {'real': 'real', 'syn': 'syn',
                                'syn-real': 'real',
                                'syn2-real': 'real',
                                'syn4-real': 'real'
                                }[mode]
                    model_dir = {'real': 'appliedUnet', 'syn': 'appliedUnet',
                                 'syn-real': 'appliedUnet_from_syn_pretrain',
                                 'syn2-real': 'appliedUnet_from_syn2_pretrain',
                                 'syn4-real': 'appliedUnet_from_syn4_pretrain'
                                 }[mode]
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
        for real in ['real', 'syn', 'syn-real', 'syn2-real', 'syn4-real']:
            if h == 'all' and real in ['syn-real', 'syn2-real', 'syn4-real']:
                continue
            row = _get_dataset_row_from_df(df_dataset_id, h, real, cfg.data.fold)
            nnunet_dataset_name = row['dataset_name']
            if os.path.exists(join(nnUNet_preprocessed, nnunet_dataset_name)):
                rsync_w.upload(join(nnUNet_preprocessed, nnunet_dataset_name), if_dir=True)
            if os.path.exists(join(nnUNet_results, nnunet_dataset_name)):
                rsync_w.upload(join(nnUNet_results, nnunet_dataset_name), if_dir=True)

    print('Success', datetime.now().strftime('%H:%M:%S'))

if __name__ == '__main__':
    main()
