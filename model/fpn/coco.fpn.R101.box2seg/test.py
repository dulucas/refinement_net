import glob
import json
import cv2
import numpy as np
import pickle as pkl
from imantics import Polygons, Mask

from config import config
from semantic_seg import SemanticSegmentor
from fpn_config import get_cfg

from utils.img_utils import normalize
import matplotlib.pyplot as plt
from pycocotools.coco import COCO as cocoapi

import torch

softmax = torch.nn.Softmax(dim=1)

cfg = get_cfg()
cfg.merge_from_file('./fpn_config/semantic_R_50_FPN_1x.yaml')
model = SemanticSegmentor(cfg)
model.load_state_dict(torch.load('./log/snapshot/epoch-5.pth')['model'], strict=True)

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
        print(pred.shape)
        pred = softmax(pred)

    pred = pred.squeeze().cpu().numpy()
    pred = pred[1]
    pred = cv2.resize(pred, (img_ori.shape[1], img_ori.shape[0]))

    #cv2.imwrite('pred.jpg', (pred*255).astype(np.uint8))
    plt.figure()
    plt.imshow(pred)
    plt.figure()
    plt.imshow(img_ori)
    plt.show()
