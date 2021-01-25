import glob
import json
import cv2
import numpy as np
import pickle as pkl
from imantics import Polygons, Mask
from config import config
from network import PSPNet
from utils.img_utils import normalize
import matplotlib.pyplot as plt
from pycocotools.coco import COCO as cocoapi

import torch

softmax = torch.nn.Softmax(dim=1)

model = PSPNet(2, None)
model_dict = model.state_dict()

pretrained_model_path = './log/snapshot/epoch-5.pth'
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

f = json.load(open('/home/duy/phd/lucasdu/duy/annotations/instances_val2017.json', 'r'))
images = f['images']
annos = f['annotations']
coco = cocoapi('/home/duy/phd/lucasdu/duy/annotations/instances_val2017.json')

np.random.shuffle(annos)

id2name = dict()
for i in images:
    id2name[i['id']] = '/home/duy/phd/lucasdu/duy/val2017/' + i['file_name']

for i in annos:
    img = cv2.imread(id2name[i['image_id']])
    img = img[:,:,::-1]
    bbox = i['bbox']
    h,w = bbox[3],bbox[2]
    if h*w < 60*60:
        continue
    print(id2name[i['image_id']])
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
    bbox = [int(i) for i in bbox]
    bbox_ori = bbox

    ratio = 0.05
    x0,y0,x1,y1 = bbox[0],bbox[1],bbox[2],bbox[3]
    x0 += -ratio * w
    y0 += -ratio * h
    x1 += ratio * w
    y1 += ratio * h
    x0 = int(max(0, x0))
    y0 = int(max(0, y0))
    x1 = int(min(x1, img.shape[1]))
    y1 = int(min(y1, img.shape[0]))

    valid = np.zeros((img.shape[0], img.shape[1]))
    valid[bbox_ori[1]:bbox_ori[3], bbox_ori[0]:bbox_ori[2]] = 1
    img_crop = img[y0:y1, x0:x1, :]
    img_ori = img_crop.copy()
    img_crop = normalize(img_crop, config.image_mean, config.image_std)
    valid = valid[y0:y1, x0:x1]

    img_crop = cv2.resize(img_crop, (384, 384))
    valid = cv2.resize(valid, (384, 384), interpolation=cv2.INTER_NEAREST)
    valid = np.expand_dims(valid, -1)
    img_crop = np.concatenate([img_crop, valid], axis=-1)
    img_crop = img_crop.transpose(2,0,1)

    img_crop = torch.FloatTensor(img_crop).cuda()
    img_crop = img_crop.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_crop)
        pred = softmax(pred)

    pred = pred.squeeze().cpu().numpy()
    pred = pred[1]
    pred = cv2.resize(pred, (img_ori.shape[1], img_ori.shape[0]))
    plt.figure()
    plt.imshow(pred)
    plt.figure()
    plt.imshow(img_ori)
    plt.show()
