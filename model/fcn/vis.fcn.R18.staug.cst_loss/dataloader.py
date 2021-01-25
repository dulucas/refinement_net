import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils import data
import torch.nn.functional as F

from config import config
from utils.img_utils import random_hflip, random_vflip
from utils.img_utils import random_rotation_pair as random_rotation
from utils.img_utils import random_scale_pair as random_scale
from utils.img_utils import normalize, generate_random_crop_pos_center, random_crop_pad_to_shape, random_crop_pad_to_shape_flow

class TrainPre(object):
    def __init__(self, img_mean, img_std, target_size, opflow_model):
        self.img_mean = img_mean
        self.img_std = img_std
        self.target_size = target_size
        self.opflow_model = opflow_model

    def _generate_opflow(self, img0, img1, mask, num_iters=20):
        _,h,w = img0.shape
        image_prev = img0.unsqueeze(0)
        image_nex = img1.unsqueeze(0)

        with torch.no_grad():
            _, flow_forward = self.opflow_model(image_prev, image_nex, iters=num_iters, test_mode=True)
            _, flow_backward = self.opflow_model(image_nex, image_prev, iters=num_iters, test_mode=True)

        flow_forward = flow_forward.squeeze()
        flow_backward = flow_backward.squeeze()
        flow_forward = flow_forward.permute(1, 2, 0).cpu().numpy()
        flow_backward = flow_backward.permute(1, 2, 0).cpu().numpy()

        flow_forward[:,:,0] = flow_forward[:,:,0] / w * mask
        flow_forward[:,:,1] = flow_forward[:,:,1] / h * mask
        flow_backward[:,:,0] = flow_backward[:,:,0] / w * mask
        flow_backward[:,:,1] = flow_backward[:,:,1] / h * mask
        #print(flow_forward.min(), flow_forward.max())
        return flow_forward, flow_backward

    def __call__(self, img0, img1, gt0, gt1):
        img0, img1, gt0, gt1 = random_hflip(img0, img1, gt0, gt1)
        img0, img1, gt0, gt1 = random_vflip(img0, img1, gt0, gt1)

        #img0, img1, gt0, gt1, mask = random_rotation(img0, img1, gt0, gt1)
        mask = np.ones(gt0.shape)

        #img0_ = torch.from_numpy(np.ascontiguousarray(img0)).float().cuda().permute(2,0,1)
        #img1_ = torch.from_numpy(np.ascontiguousarray(img1)).float().cuda().permute(2,0,1)

        #flow_forward, flow_backward = self._generate_opflow(img0_, img1_, mask)
        flow0 = img0
        flow1 = img1

        if config.train_scale_array is not None:
            img0, img1, gt0, gt1, flow0, flow1, mask, scale = random_scale(img0, img1, gt0, gt1, flow0, flow1, mask, config.train_scale_array[0], config.train_scale_array[-1])

        img0 = normalize(img0, self.img_mean, self.img_std)
        img1 = normalize(img1, self.img_mean, self.img_std)

        crop_size = (config.image_height, config.image_width)
        gt = gt0 if np.random.rand() > .5 else gt1
        cls_list = np.unique(gt).tolist()
        cls_list = [i for i in cls_list if i]
        cls = np.random.choice(cls_list)
        gt = (gt == cls).astype(np.float32)
        crop_pos = generate_random_crop_pos_center(img0.shape[:2], crop_size, gt)

        p_img0, _ = random_crop_pad_to_shape(img0, crop_pos, crop_size, 0)
        p_img1, _ = random_crop_pad_to_shape(img1, crop_pos, crop_size, 0)
        p_gt0, _ = random_crop_pad_to_shape(gt0, crop_pos, crop_size, 255)
        p_gt1, _ = random_crop_pad_to_shape(gt1, crop_pos, crop_size, 255)
        #p_flow0, _ = random_crop_pad_to_shape_flow(flow0, crop_pos, crop_size, 0)
        #p_flow1, _ = random_crop_pad_to_shape_flow(flow1, crop_pos, crop_size, 0)
        p_mask, _ = random_crop_pad_to_shape(mask, crop_pos, crop_size, 0)
        p_img0[:,:,0][p_mask == 0] = 0
        p_img0[:,:,1][p_mask == 0] = 0
        p_img0[:,:,2][p_mask == 0] = 0
        p_img1[:,:,0][p_mask == 0] = 0
        p_img1[:,:,1][p_mask == 0] = 0
        p_img1[:,:,2][p_mask == 0] = 0

        p_img0_ = torch.from_numpy(np.ascontiguousarray((p_img0 * config.image_std + config.image_mean) * 255).astype(np.uint8)).float().cuda().permute(2,0,1)
        p_img1_ = torch.from_numpy(np.ascontiguousarray((p_img1 * config.image_std + config.image_mean) * 255).astype(np.uint8)).float().cuda().permute(2,0,1)
        flow_forward, flow_backward = self._generate_opflow(p_img0_, p_img1_, p_mask)
        p_flow0 = flow_forward
        p_flow1 = flow_backward

        if False:
            p_img0_warp = update_matrix(p_img0_, torch.from_numpy(p_flow0).float().cuda(), torch.from_numpy(p_mask).float().cuda())
            plt.figure()
            plt.imshow(((p_img0 * config.image_std + config.image_mean) * 255).astype(np.uint8))
            plt.figure()
            plt.imshow(((p_img1 * config.image_std + config.image_mean) * 255).astype(np.uint8))
            plt.figure()
            plt.imshow(((p_img0_warp.cpu().numpy().transpose(1,2,0))).astype(np.uint8))
            plt.figure()
            plt.imshow(p_flow0[:,:,0])
            plt.figure()
            plt.imshow(p_gt0)
            plt.show()

        p_img0 = p_img0.transpose(2, 0, 1)
        p_img1 = p_img1.transpose(2, 0, 1)

        return p_img0, p_img1, p_gt0, p_gt1, p_flow0, p_flow1, p_mask

def update_matrix(img, flow, mask):
    flow = flow.unsqueeze(0)
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(-1).unsqueeze(0)

    theta = torch.Tensor([[1, 0, 0], [0, 1, 0]])
    theta = theta.view(-1, 2, 3).cuda()

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    grid = grid + 2*flow
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=False)
    return img.squeeze()

def get_train_loader(engine, dataset, opflow_model):
    data_setting = {'data_train_source': config.data_root_path}

    train_preprocess = TrainPre(config.image_mean, config.image_std,
                                config.target_size, opflow_model)

    train_dataset = dataset(data_setting, "train", train_preprocess,
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
