import cv2
import numpy as np

import torch
from torch.utils import data

from config import config
from utils.img_utils import random_bbox_jitter, random_crop_w_bbox, random_hflip, random_vflip, random_gamma, normalize


class TrainPre(object):
    def __init__(self, img_mean, img_std, target_size):
        self.img_mean = img_mean
        self.img_std = img_std
        self.target_size = target_size

    def __call__(self, img, mask, bbox):
        bbox_shift = random_bbox_jitter(bbox, img.shape[0], img.shape[1])
        img, mask, valid = random_crop_w_bbox(img, mask, bbox_shift, bbox)

        img = cv2.resize(img, (config.image_width, config.image_height))
        mask = cv2.resize(mask, (config.image_width, config.image_height), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid, (config.image_width, config.image_height), interpolation=cv2.INTER_NEAREST)

        img, mask, valid = random_hflip(img, mask, valid)
        img, mask, valid = random_vflip(img, mask, valid)
        img = random_gamma(img)
        img = normalize(img, self.img_mean, self.img_std)

        valid = np.expand_dims(valid, -1)
        img = np.concatenate([img, valid], axis=-1)
        img = img.transpose(2, 0, 1)

        extra_dict = None

        return img, mask, extra_dict


def get_train_loader(engine, dataset):
    data_setting = {'data_train_source': config.data_root_path}
    train_preprocess = TrainPre(config.image_mean, config.image_std,
                                config.target_size)

    train_dataset = dataset(data_setting, "train", 0.2, train_preprocess,
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
