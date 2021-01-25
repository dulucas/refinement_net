import cv2
import numpy as np

import torch
from torch.utils import data

from config import config
from utils.img_utils import random_scale, random_mirror, normalize, \
    generate_random_crop_pos_center, random_crop_pad_to_shape, random_scale_with_length

import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrainPre(object):
    def __init__(self, img_mean, img_std, target_size):
        self.img_mean = img_mean
        self.img_std = img_std
        self.target_size = target_size

    def __call__(self, img, gt):
        img, gt = random_mirror(img, gt)
        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array)

        img = normalize(img, self.img_mean, self.img_std)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos_center(img.shape[:2], crop_size, gt)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

        p_img = p_img.transpose(2, 0, 1)

        extra_dict = None

        return p_img, p_gt, extra_dict


def get_train_loader(engine, dataset):
    data_setting = {'data_train_source': config.data_root_path}

    train_transform = A.Compose(
	[
            A.RandomResizedCrop(always_apply=False, p=0.8, height=512, width=512, scale=(0.1, 2.0), ratio=(0.75, 1.3333333333333333)),
            A.ElasticTransform(always_apply=False, p=0.8, alpha=1.0, sigma=50.0, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
            A.Cutout(always_apply=False, p=1.0, num_holes=16, max_h_size=8, max_w_size=8),
            A.ChannelShuffle(always_apply=False, p=0.5),
            #A.ISONoise(always_apply=False, p=1.0, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
            A.Equalize(always_apply=False, p=1.0, mode='cv', by_channels=True),
            A.Flip(always_apply=False, p=0.5),
            #A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(-0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            #A.InvertImg(always_apply=False, p=0.2),
            #A.JpegCompression(always_apply=False, p=1.0, quality_lower=80, quality_upper=100),
            A.MultiplicativeNoise(always_apply=False, p=1.0, multiplier=(0.8999999761581421, 1.9399999380111694), per_channel=True, elementwise=True),
            A.RGBShift(always_apply=False, p=1.0, r_shift_limit=(-164, 149), g_shift_limit=(-208, 223), b_shift_limit=(-193, 193)),
            A.VerticalFlip(always_apply=False, p=0.5),
	    A.Resize(512, 512),
	    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
	    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
	    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
	    ToTensorV2(),
	]
    )

    train_dataset = dataset(data_setting, "train", train_transform,
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
