import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils import data
import torch.nn.functional as F

def update_matrix(img, flow):
    theta = torch.Tensor([[1, 0, 0], [0, 1, 0]])
    theta = theta.view(-1, 2, 3).cuda()

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    grid = grid + flow
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=False)
    return img

img = torch.zeros((1,1,100,100))
img[:,:,:10,:10] = 1
img = img.cuda()
flow = torch.ones((1,100,100,2))
flow *= -1
flow = flow.cuda()

while True:
    plt.imshow(img.squeeze().cpu().numpy())
    plt.show()
    img = update_matrix(img, flow)
