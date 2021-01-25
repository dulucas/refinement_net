import cv2
import numpy as np

import torch
from torch.utils import data

from config import config
from utils.img_utils import random_scale_new, random_mirror_new, normalize, \
    generate_random_crop_pos_center, random_crop_pad_to_shape, random_scale_with_length


class TrainPre(object):
    def __init__(self, img_mean, img_std, target_size):
        self.img_mean = img_mean
        self.img_std = img_std
        self.target_size = target_size

    def __call__(self, img, pred, gt, fgprob):
        img, pred, gt, fgprob = random_mirror_new(img, pred, gt, fgprob)
        if config.train_scale_array is not None:
            img, pred, gt, fgprob, scale = random_scale_new(img, pred, gt, fgprob, config.train_scale_array)

        img = normalize(img, self.img_mean, self.img_std)
        pred[pred > 0] = 1
        gt[gt > 0] = 1

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos_center(img.shape[:2], crop_size, gt)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 0)
        p_pred, _ = random_crop_pad_to_shape(pred, crop_pos, crop_size, 0)
        p_fgprob, _ = random_crop_pad_to_shape(fgprob, crop_pos, crop_size, 0)

        p_img = p_img.transpose(2, 0, 1)
        p_pred = np.expand_dims(p_pred, 0)
        p_fgprob = np.expand_dims(p_fgprob, 0)
        p_img = np.concatenate([p_img, p_pred], axis=0)
        p_img = np.concatenate([p_img, p_fgprob], axis=0)

        p_gt = np.expand_dims(p_gt, 0)

        extra_dict = None

        return p_img, p_gt, extra_dict


def get_train_loader(engine, dataset):
    data_setting = {'data_train_source': config.data_root_path}
    train_preprocess = TrainPre(config.image_mean, config.image_std,
                                config.target_size)

    train_dataset = dataset(data_setting, "train", 0.3, train_preprocess,
                            config.niters_per_epoch * config.batch_size)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
