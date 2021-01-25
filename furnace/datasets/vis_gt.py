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


class VIS_GT(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None):
        super(VIS_GT, self).__init__()
        self._split_name = split_name
        self._file_names = self._load_data_file(setting['data_train_source'])
        self._file_length = len(self._file_names)
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        name = self._file_names[index]

        img, gt = self._fetch_data(name)
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, gt)

        if self._split_name == 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()

            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].float()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(img=img, gt=gt,
                           n=len(self._file_names))
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, name):
        gtname = name
        imgname = gtname.replace('/home/duy/phd/lucasdu/duy/train_all_frames/Annotations/', '/home/duy/phd/lucasdu/duy/train_all_frames/JPEGImages/').replace('png', 'jpg')

        img = cv2.imread(imgname)
        gt = Image.open(gtname)
        gt = np.atleast_3d(gt)[...,0]
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        return img, gt

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
