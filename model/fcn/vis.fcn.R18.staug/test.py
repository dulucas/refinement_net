import os
import glob
import json
import cv2
import numpy as np
import pickle as pkl
from config import config
from network import FCN
from utils.img_utils import normalize
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

def color_image_w_masks(image, labelmap, color_palette):
    for index in np.unique(labelmap):
        if index == 0:
            continue
        mask = (labelmap == index).astype(np.float32)
        color = color_palette[index]
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)
        mask = mask * np.array(color).reshape((-1, 3)) + (1 - mask) * image
        image = mask.astype(np.uint8)
        #image = cv2.addWeighted(image, .5, mask, .5, 0)
    return image

color_palette = np.loadtxt('/home/duy/phd/UVC/libs/data/palette.txt', dtype=np.uint8).reshape(-1,3)

def eval(model, name, use_gt=False):
    video_name = config.data_root_path.split('/')[-2] #'8939db6354'
    images_name = glob.glob('/home/duy/phd/lucasdu/duy/train_all_frames/JPEGImages/' + video_name.strip() + '/*')
    images_name.sort()
    images_name = images_name[::5]

    video_rgbs = []

    for image_name in images_name:

        image = cv2.imread(image_name)
        image = cv2.resize(image, (1024, 512))
        if use_gt:
            gt = Image.open(image_name.replace('JPEGImages', 'Annotations').replace('jpg', 'png'))
        else:
            gt_path = image_name.replace('train_all_frames/JPEGImages', 'semseg_vis/train').replace('jpg', 'png')
            if not os.path.exists(gt_path):
                continue
            gt = Image.open(gt_path)
        gt = np.atleast_3d(gt)[...,0]
        gt = cv2.resize(gt, (512, 256), interpolation=cv2.INTER_NEAREST)
        image_ori = image.copy()
        image = normalize(image, config.image_mean, config.image_std)
        image = image.transpose(2, 0, 1)

        image_ = image.copy()
        image_ = torch.FloatTensor(image_)
        image_ = image_.unsqueeze(0).cuda()
        with torch.no_grad():
            pred = model(image_)
            pred = F.softmax(pred, dim=1)

        pred = pred.squeeze().cpu().numpy()
        labelmap = np.argmax(pred, axis=0)
        labelmap = labelmap.astype(np.uint8)
        labelmap = cv2.resize(labelmap, (512, 256))
        image_ori = cv2.resize(image_ori, (512, 256))
        black = (255 * np.ones((labelmap.shape[0], labelmap.shape[1], 3))).astype(np.uint8)
        colormap = color_image_w_masks(black, labelmap, color_palette)
        black = (255 * np.ones((labelmap.shape[0], labelmap.shape[1], 3))).astype(np.uint8)
        colormap_gt = color_image_w_masks(black, gt, color_palette)

        final_result = np.concatenate([image_ori, colormap_gt], axis=1)
        final_result = np.concatenate([final_result, colormap], axis=1)

        for i in range(6):
            video_rgbs.append(final_result)

    size = (int(video_rgbs[0].shape[1]), int(video_rgbs[0].shape[0]))
    if use_gt:
        video_name = video_name + '_gt'
    if not os.path.exists('./videos/' + video_name):
        os.makedirs('./videos/' + video_name)
    out_rgbs = cv2.VideoWriter('./videos/' + video_name + '/' + name + '_rgb.mp4', 0x7634706d, 20, size)

    for i in range(len(video_rgbs)):
        out_rgbs.write(video_rgbs[i])

    out_rgbs.release()
    #cv2.imshow('img', image_ori)
    #cv2.imshow('mask', colormap)
    #cv2.waitKey(5)

if __name__ == '__main__':
    model = FCN(#config.num_classes, criterion=None,
                7, criterion=None,
                pretrained_model=None,
                norm_layer=torch.nn.BatchNorm2d)
    model_dict = model.state_dict()

    pretrained_model_path = './log/snapshot/epoch-last.pth'
    tmp_dict = torch.load(pretrained_model_path)
    pretrained_dict = {}
    for k, v in tmp_dict['model'].items():
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            print(k)
    for k in model_dict.keys():
        if k not in pretrained_dict:
            print(k)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    model.eval()

    eval(model, 'test', False)
