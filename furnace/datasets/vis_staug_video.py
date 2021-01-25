#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : BaseDataset.py

import os
import time
import glob
import cv2
import torch
import pickle as pkl
import numpy as np
from PIL import Image
import torch.utils.data as data


class VIS_VIDEO(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, step=3, num=2):
        super(VIS_VIDEO, self).__init__()
        self._split_name = split_name
        self._file_names = self._load_data_file(setting['data_train_source'])
        self._file_length = len(self._file_names)
        self.preprocess = preprocess
        self.step = step
        self.num = num

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        name = self._file_names[index]

        imgs, gts = self._fetch_data(name)
        img0, img1, gt0, gt1, flow0, flow1, mask = self.preprocess(imgs[0], imgs[1], gts[0], gts[1])

        img0 = torch.from_numpy(np.ascontiguousarray(img0)).float()
        img1 = torch.from_numpy(np.ascontiguousarray(img1)).float()
        gt0 = torch.from_numpy(np.ascontiguousarray(gt0)).long()
        gt1 = torch.from_numpy(np.ascontiguousarray(gt1)).long()
        flow0 = torch.from_numpy(np.ascontiguousarray(flow0)).float()
        flow1 = torch.from_numpy(np.ascontiguousarray(flow1)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

        output_dict = dict(img0=img0, img1=img1, gt0=gt0, gt1=gt1, flow0=flow0, flow1=flow1, mask=mask,
                           n=len(self._file_names))

        return output_dict

    def _fetch_data(self, name):
        imgs = []
        gts = []

        frame_index = int(name.split('/')[-1].split('.')[0])
        frame_indexes = self._get_valid_frame_indexes(frame_index, name, self.step, self.num)

        for index in frame_indexes:

            gtname = name.replace(str(frame_index).zfill(5), str(index).zfill(5))
            imgname = gtname.replace('/home/duy/phd/lucasdu/duy/semseg_vis/train/', '/home/duy/phd/lucasdu/duy/train_all_frames/JPEGImages/').replace('png', 'jpg')

            img = cv2.imread(imgname)
            img = img[:, :, ::-1]
            gt = Image.open(gtname)
            gt = np.atleast_3d(gt)[...,0]
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            imgs.append(img)
            gts.append(gt)

        return imgs, gts

    def _get_valid_frame_indexes(self, index, name, step, num):
        path = name.split('/')[-2]
        frames = glob.glob('/home/duy/phd/lucasdu/duy/semseg_vis/train/' + path + '/*png')
        frames.sort()
        min_index = int(frames[0].split('/')[-1].split('.')[0])
        max_index = int(frames[-1].split('/')[-1].split('.')[0])

        start_index = min(max(min_index, index), max_index-(num-1)*step)
        end_index = min(index+(num-1)*step, max_index)

        indexes = []
        for i in range(start_index, end_index+1, step):
            indexes.append(i)
        return indexes

    def _load_data_file(self, data_source_path):
        data_file = glob.glob(data_source_path + '/*png')
        data_file.sort()
        return data_file


    def _get_file_names(self, imgs):
        file_names = list(set([i['id'] for i in imgs]))
        return file_names

    def get_length(self):
        return self.__len__()

if __name__ == "__main__":
    data_setting = {'data_train_source': '/home/duy/phd/lucasdu/duy/coco_train2017_refine_training/info.txt'}
    bd = VIS(data_setting, 'train', 0.3)
    print(len(bd._file_names))
