from PIL import Image
from pathlib import Path

import copy
import logging
import os
import random
import shutil
import sys
import time
import warnings
from itertools import chain

import cv2
import numpy as np
import ray
import torch
from ray.util import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from torch.optim import AdamW
from collections import OrderedDict, defaultdict, abc as container_abcs


def crop_sample(x):
    volume, mask = x
    volume[volume < np.max(volume) * 0.1] = 0 # I understand why this is here: to not take
    #                                           background noise into account when cropping;
    #                                           but proper percentile thresholding is done after this
    #                                           Maybe don't modify the volume in-place: do it on a copy for cropping,
    #                                           and then do the thresholding on the original volume?
    #                                           In any case, cropping doesn't appear to do anything, so I don't
    #                                           do this step.
    z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_projection)
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1
    y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
    y_nonzero = np.nonzero(y_projection)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1
    x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_projection)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1
    return (
        volume[z_min:z_max, y_min:y_max, x_min:x_max],
        mask[z_min:z_max, y_min:y_max, x_min:x_max],
    )


def pad_sample(x):
    volume, mask = x
    a = volume.shape[1]
    b = volume.shape[2]
    if a == b:
        return volume, mask
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    mask = np.pad(mask, padding, mode="constant", constant_values=0)
    padding = padding + ((0, 0),)
    volume = np.pad(volume, padding, mode="constant", constant_values=0)
    return volume, mask

# import cv2
import torchvision
def resize_sample(x, size=256, permute_image=True):
    st = time.time()
    volume, mask = x
    v_shape = volume.shape
    out_shape = (v_shape[0], size, size)
    print(f'{mask.shape=}')
    # cv2.resize()
    if_torch_resize = True
    if not if_torch_resize:
        mask = resize(
            mask,
            output_shape=out_shape,
            order=0,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
    else:
        old_mask_dtype = copy.deepcopy(mask.dtype)
        mask = torch.Tensor(mask)
        if len(mask.shape) > 3:
            mask = torch.permute(mask, (0, 3, 1, 2))

        mask = torchvision.transforms.functional.resize(mask, (size, size),
                                                    torchvision.transforms.InterpolationMode.NEAREST_EXACT,
                                                    antialias=False
                                                    )

        if len(mask.shape) > 3:
            mask = torch.permute(mask, (0, 2, 3, 1))

        mask = mask.numpy().astype(old_mask_dtype)

    print(f'{mask.shape=}')
    out_shape = out_shape + (v_shape[3],)
    print(f'{volume.shape=}')
    if not if_torch_resize:
        volume = resize(
            volume,
            output_shape=out_shape,
            order=2,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
    else:
        old_volume_dtype = copy.deepcopy(volume.dtype)
        volume = torch.Tensor(volume)
        if permute_image:
            volume = torch.permute(volume, (0, 3, 1, 2))
        volume = torchvision.transforms.functional.resize(volume, (size, size),
                                                    torchvision.transforms.InterpolationMode.BILINEAR,
                                                    antialias=False
                                                    )
        if permute_image:
            volume = torch.permute(volume, (0, 2, 3, 1))
        volume = volume.numpy().astype(old_volume_dtype)

    print(f'{volume.shape=}')
    print(f'resize time {time.time() - st:.4f}')
    return volume, mask

def resize_sample_only_mask(mask, size=256):
    st = time.time()
    # print(f'Before {mask.shape=}')
    old_mask_dtype = copy.deepcopy(mask.dtype)
    mask = torchvision.transforms.functional.resize(torch.Tensor(mask), size,
                                                torchvision.transforms.InterpolationMode.NEAREST_EXACT,
                                                antialias=False
                                                ).numpy().astype(old_mask_dtype)
    # print(f'After {mask.shape=}')
    # print(f'resize time {time.time() - st:.4f}')
    return mask

def normalize_volume(volume):
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume

def normalize_volume_my1(volume):
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    return volume

def normalize_volume_my_1_5_make_positive(volume):
    '''
    unlike brain data which after normalize_my1 is in [0, 1], cervix data is in [-1, 1]
    '''
    volume = volume + 1
    volume /= 2
    return volume


def normalize_volume_my2(volume):
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume

def normalize_volume_my2_return_mean_and_std(volume):
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume, m, s

def log_images(x, y_true, y_pred, n_to_log, channel=1):
    images = []
    if x.shape[1] == 1:
        channel = 0
    x_np = x[:, channel].cpu().numpy()
    # y_true_np = y_true[:, 0].cpu().numpy()
    # y_pred_np = y_pred[:, 0].cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    mask_channels = [0] if y_true_np.shape[1] == 1 else list(range(1, y_true_np.shape[1]))
    for i in range(min(x_np.shape[0], n_to_log)):
        image = gray2rgb(np.squeeze(x_np[i]))
        for ch in mask_channels:
            image = outline(image, y_pred_np[i][ch], color=[255, 0, 0])
            image = outline(image, y_true_np[i][ch], color=[0, 255, 0])
        images.append(image)
    return images


def gray2rgb(image):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret


def outline(image, mask, color):
    mask = np.round(mask)
    if False: # this is hella slow
        yy, xx = np.nonzero(mask)
        for y, x in zip(yy, xx):
            if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
                image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    else:
        mask_eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        image[mask != mask_eroded] = color
    return image


def normalize_mask(mask):
    #https://github.com/mateuszbuda/brain-segmentation-pytorch/issues/36

    if np.max(mask) == 0:
        return mask

    mask = mask // (np.max(mask))
    return mask

def set_random_seeds(seed=42):
    # for reproducibility, need to flip the flags below. But that hurts GAN performance.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f'random seed set to {seed}')

class LogWriter:
    def __init__(self, log_fun):
        self.log_fun = log_fun
        self.buf = []
        self.is_tqdm_msg_fun = lambda msg: '%|' in msg
        # annoyingly, ray doesn't allow to disable colors in output, and they make logs unreadable, so:
        self.replace_garbage = lambda msg: msg.replace('[2m[36m', '').replace('[0m', '').replace('[32m', '')

    def write(self, msg):
        is_tqdm = self.is_tqdm_msg_fun(msg)
        has_newline = msg.endswith('\n')
        if has_newline or is_tqdm:
            self.buf.append(msg)  # .rstrip('\n'))
            self.log_fun(self.replace_garbage(''.join(self.buf)))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass

    def close(self):
        self.log_fun.close()


def setup_logging(log_path):
    from importlib import reload
    reload(logging)
    logging.StreamHandler.terminator = ''  # don't add new line, I'll do it myself; this line affects both handlers
    stream_handler = logging.StreamHandler(sys.__stdout__)
    file_handler = logging.FileHandler(log_path, mode='a')
    # don't want a bazillion tqdm lines in the log:
    # file_handler.filter = lambda record: '%|' not in record.msg or '100%|' in record.msg
    file_handler.filter = lambda record: '[A' not in record.msg and ('%|' not in record.msg or '100%|' in record.msg)
    handlers = [
        file_handler,
        stream_handler]
    logging.basicConfig(level=logging.INFO,
                        # format='%(asctime)s %(message)s',
                        # https://docs.python.org/3/library/logging.html#logrecord-attributes
                        # https://docs.python.org/3/library/logging.html#logging.Formatter
                        # format='%(process)d %(message)s',
                        format='%(message)s',
                        handlers=handlers,
                        datefmt='%H:%M')
    sys.stdout = LogWriter(logging.info)
    sys.stderr = LogWriter(logging.error)

def make_mask_onehot(mask, unique_values):
    # unique_values = np.unique(mask)
    # unique_values = unique_values[unique_values != 0]  # Exclude 0

    # Create a one-hot encoded representation
    c, w, h = len(unique_values), mask.shape[0], mask.shape[1]
    one_hot = np.zeros((c, w, h), dtype=int)

    for i, value in enumerate(unique_values):
        one_hot[i][mask == value] = 1
    one_hot = np.transpose(one_hot, (1, 2, 0))
    return one_hot

def argmax_and_onehotencode(t):
    if t.shape[1] == 1:
        # create prediction for background by substracting from 1
        t = torch.cat((1 - t, t), dim=1)
    am = torch.argmax(t, dim=1, keepdim=True)
    ohe = torch.zeros_like(t)
    ohe.scatter_(1, am, 1)
    return ohe


@ray.remote(num_cpus=1, max_calls=1)  # this num_cpus shouldn't be changed!
def ray_wrap(fun, *args):
    fun(*args)


def ray_run_fun_once_per_node(fun, *args):
    # st = time.time()
    bundles = [{"CPU": 1} for _ in ray.nodes()]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        pg = placement_group(bundles, "STRICT_SPREAD")
        ray.get(pg.ready())
        tasks = [ray_wrap.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg)).remote(fun, *args)
                 for i in range(len(ray.nodes()))]
        for t in tasks:
            ray.get(t)
        remove_placement_group(pg)

def print_individual(individual):
    hps = '|'.join(map(str, individual['hyperparameters']))
    print(f"\tmodel_id={individual['model_id']} HPs={hps}")
    if 'history' in individual:
        for k, v in individual['history'].items():
            v = ','.join(map(str, v))
            print(f"{k}:[{v}] ", end='')
        print()
    special_keys = ['model_id', 'hyperparameters', 'history']
    for k, v in individual.items():
        if k in special_keys:
            continue
        if type(v) is list and len(v) > 0 and all(type(v[i]) is float for i in range(len(v))):
            v = ', '.join(map(lambda x: f'{x:.4f}', v))
        print(f"{k}: {v}")


def print_population(population, name):
    print(name)
    for individiual in population:
        print_individual(individiual)


class MyAdamW(AdamW):
    r"""My modification of AdamW: remove state if the shape doesn't match
    (this way, the state for unchanged parts of the network is kept)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        super(MyAdamW, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def __setstate__(self, state):
        super(MyAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

            for p in group['params']:

                state = self.state[p]
                if len(state) == 0:
                    continue

                if state['exp_avg'].shape != p.shape:
                    # assert state['exp_avg'].shape == torch.Size([17]), f"{state['exp_avg'].shape=} {p.shape=}"
                    print('Deleted mismatching state without checking')
                    del self.state[p]

class SGDIgnoreMismatchingState(torch.optim.SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize: bool = False, foreach = None,
                 differentiable: bool = False):
        '''
        need to modify __setstate__ for the case if parameter exists but the shape doesn't match
        need to modify load_state_dict for the case if parameter doesn't exist.
        '''
        super(SGDIgnoreMismatchingState, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov,
                                                        maximize=maximize, foreach=foreach, differentiable=differentiable)

    def __setstate__(self, state):
        super(SGDIgnoreMismatchingState, self).__setstate__(state)
        for group in self.param_groups:
            # group.setdefault('nesterov', False)

            for p in group['params']:

                state = self.state[p]
                if len(state) == 0:
                    continue

                if state['momentum_buffer'].shape != p.shape:
                    print('Deleted mismatching state without checking')
                    del self.state[p]

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")

        # MY CHANGE: do not restore the state if the nubmer of parameters changed
        for i in range(len(groups)):
            if len(groups[i]['params']) != len(saved_groups[i]['params']):
                saved_groups[i]['params'] = copy.deepcopy(groups[i]['params'])
        # /MY CHANGE

        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
                if (key != "step"):
                    if param.is_floating_point():
                        value = value.to(param.dtype)
                    value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

def adjust_optimizer_settings(optimizer, lr, wd=None):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        if wd is not None:
            param_group['weight_decay'] = wd

    return optimizer

def optimizer_to(optim, device):
    if not (type(optim) is dict):
        all_values = [optim.state.values()]
    else:  # in gan, optim is a dict of optims
        all_values = [o.state.values() for o in optim.values()]
    for values in all_values:
        for param in values:
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

def seq_to_str(ar):
    return ','.join([str(t) for t in ar])

def try_load_checkpoint(folder, checkpoint_name):
    checkpoint_full_name = os.path.join(folder, checkpoint_name)

    if not os.path.exists(checkpoint_full_name):
        print(f'Checkpoint for model {checkpoint_name} not found')
        return None

    checkpoint = torch.load(checkpoint_full_name, map_location='cpu')

    return checkpoint

def rmtree_if_exists(path):
    if not os.path.exists(path):
        return
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def resize_images(source_dir, target_dir, size=(128, 128)):
    """
    Resizes all PNG images in the source directory and saves them to the target directory.

    :param source_dir: Path to the directory containing the original images.
    :param target_dir: Path to the directory where resized images will be saved.
    :param size: New size for the images as a tuple (width, height). Default is (128, 128).
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    for image_path in source_path.glob('*.png'):
        img = Image.open(image_path)
        img = img.resize(size)
        img.save(target_path / image_path.name)