#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : BaseDataset.py

import os
import time
import cv2
import torch
import pickle as pkl
import numpy as np
import torch.utils.data as data


class COCO(data.Dataset):
    def __init__(self, setting, split_name, thre, preprocess=None,
                 file_length=None):
        super(COCO, self).__init__()
        self._split_name = split_name
        self._file_names = self._load_data_file(setting['data_train_source'], thre)
        self._file_length = len(self._file_names)
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        name = self._file_names[index]

        img, pred, gt, fgprob = self._fetch_data(name)
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, pred, gt, fgprob)

        if self._split_name is 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).float()

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
        names = name.split('\t')

        imgname = names[0]
        predname = names[1]
        gtname = names[2]
        fgname = imgname.replace('train2017', 'coco_train2017_refine_training/fg').replace('jpg', 'pkl')

        img = cv2.imread(imgname)
        fgprob = pkl.load(open(fgname, 'rb'))
        pred = cv2.imread(predname, 0)
        gt = cv2.imread(gtname, 0)

        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)

        return img, pred, gt, fgprob

    def _load_data_file(self, data_source_path, thre):
        data_file = open(data_source_path, 'r').readlines()
        data = []
        for line in data_file:
            line = line.strip()
            score = float(line.split('\t')[-1])
            if score > thre:
                data.append(line)
        return data


    def _get_file_names(self, imgs):
        file_names = list(set([i['id'] for i in imgs]))
        return file_names

    def get_length(self):
        return self.__len__()


if __name__ == "__main__":
    data_setting = {'data_train_source': '/home/duy/phd/lucasdu/duy/coco_train2017_refine_training/info.txt'}
    bd = COCO(data_setting, 'train', 0.3)
    print(len(bd._file_names))
