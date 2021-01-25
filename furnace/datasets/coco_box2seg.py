#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : BaseDataset.py

import os
import time
import json
import cv2
import torch
import pickle as pkl
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO as cocoapi

class COCO(data.Dataset):
    def __init__(self, setting, split_name, thre, preprocess=None,
                 file_length=None):
        super(COCO, self).__init__()
        self._split_name = split_name
        self.annos, self.coco, self.id2name = self._load_data_file(setting['data_train_source'], thre)
        self._file_length = len(self.annos)
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self.annos)

    def __getitem__(self, index):
        name = self.annos[index]

        img, mask, bbox = self._fetch_data(name)
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, mask, bbox)

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

        output_dict = dict(img=img, gt=gt)

        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, anno):
        index = anno['image_id']
        name = self.id2name[index]
        bbox = anno['bbox']

        img = cv2.imread(name)
        img = img[:,:,::-1]
        mask = self.coco.annToMask(anno)

        return img, mask, bbox

    def _load_data_file(self, data_source_path, thre):
        data_file = json.load(open(data_source_path, 'r'))
        coco = cocoapi(data_source_path)
        annos = data_file['annotations']
        annos = [i for i in annos if not i['iscrowd']]
        annos = [i for i in annos if i['bbox'][2]*i['bbox'][3] > 36*36]
        id2name = dict()
        for i in data_file['images']:
            id2name[i['id']] = '/home/duy/phd/lucasdu/duy/train2017/' + i['file_name']
        return annos, coco, id2name

    def get_length(self):
        return self.__len__()


if __name__ == "__main__":
    data_setting = {'data_train_source': '/home/duy/phd/lucasdu/duy/coco_train2017_refine_training/info.txt'}
    bd = COCO(data_setting, 'train', 0.3)
