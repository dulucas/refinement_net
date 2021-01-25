import glob
import json
import cv2
import numpy as np
import pickle as pkl
from imantics import Polygons, Mask
from config import config
from network import Unet
from utils.img_utils import normalize
import matplotlib.pyplot as plt

import torch

softmax = torch.nn.Softmax(dim=1)
sigmoid = torch.nn.Sigmoid()

model = Unet(pretrained_model=None)
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

videos = open('/home/duy/phd/tmp_half/high_recall_video_list_for_global_opt_large.txt', 'r').readlines()
video_name = np.random.choice(videos)
video_name = 'f1ba9e1a3e'
images_name = glob.glob('/home/duy/phd/lucasdu/duy/train_all_frames/JPEGImages/' + video_name.strip() + '/*')
image_name = np.random.choice(images_name)

frame_index = int(image_name.split('/')[-1].split('.')[0])
det_results = glob.glob('/home/duy/phd/lucasdu/duy/vos_inference/Youtube_VOS_new/' + video_name.strip() + '/instance_frame' + str(frame_index).zfill(4) + '_instance*')
det_results.sort()
det_results = det_results[:10]
probs_path = '/home/duy/phd/lucasdu/duy/vos_stuff_thing_better/' + image_name.split('/')[-2] + '/' + image_name.split('/')[-1].replace('jpg', 'pkl')

prob = pkl.load(open(probs_path, 'rb'))[0]
masks = []
for det in det_results:
    det = json.load(open(det, 'r'))
    score = det['score']
    h = det['h']
    w = det['w']
    segmentation = det['mask']
    mask = Polygons(segmentation).mask(w, h)
    mask = mask.array.astype(np.float32)
    masks.append(mask)

image = cv2.imread(image_name)
image = cv2.resize(image, (1024, 512))
image_ori = image.copy()
image = normalize(image, config.image_mean, config.image_std)
image = image.transpose(2, 0, 1)
prob = cv2.resize(prob, (image.shape[2], image.shape[1]))
prob = np.expand_dims(prob, axis=0)

for mask in masks:
    image_ = image.copy()
    mask[mask > 0] = 1
    mask = cv2.resize(mask, (1024, 512), interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(mask, axis=0)
    image_ = np.concatenate([image_, mask], axis=0)
    image_ = np.concatenate([image_, prob], axis=0)

    image_ = torch.FloatTensor(image_)
    image_ = image_.unsqueeze(0).cuda()
    with torch.no_grad():
        pred = model(image_)
        pred = sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()
    plt.figure()
    plt.imshow(pred)
    plt.figure()
    plt.imshow(mask.squeeze())
    plt.figure()
    plt.imshow(image_ori[:,:,::-1])
    plt.show()
