from __future__ import division
import math
import os.path as osp
import sys
import argparse
from tqdm import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from torch.multiprocessing import set_start_method

from config import config
from dataloader import get_train_loader
from network import FCN
from datasets import VIS_VIDEO
from utils.init_func import init_weight, group_weight
from engine.lr_policy import PolyLR
from engine.engine import Engine
from legacy.sync_bn import DataParallelModel, Reduce #, BatchNorm2d

from test import eval

from raft import RAFT

#try:
#    from apex.parallel import SyncBatchNorm
#except ImportError:
#    raise ImportError(
#        "Please install apex from https://www.github.com/nvidia/apex .")


def update_matrix(img, pred, flow, mask):
    pred = pred.unsqueeze(0)
    flow = flow.unsqueeze(0)
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(-1).unsqueeze(0)

    theta = torch.Tensor([[1, 0, 0], [0, 1, 0]])
    theta = theta.view(-1, 2, 3).cuda()

    grid = F.affine_grid(theta, pred.size(), align_corners=False)
    grid = grid + flow #* mask

    pred = F.grid_sample(pred, grid, mode='bilinear', align_corners=False)
    mask = mask.permute(0,3,1,2)
    mask = F.grid_sample(mask, grid, mode='nearest', align_corners=False)

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    grid = grid + flow
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=False)
    return img.squeeze(), pred.squeeze(), mask

def calculate_cons_loss(imgs0, imgs1, pred0, pred1, flow0, flow1, mask, criterion):
    n,h,w = mask.shape
    loss = 0
    for i in range(n):
        insimg0 = imgs0[i]
        insimg1 = imgs1[i]
        inspred0 = pred0[i]
        inspred1 = pred1[i]
        insflow0 = flow0[i]
        insflow1 = flow1[i]
        insmask = mask[i]

        insimg0_warp, inspred0_warp, insmask0 = update_matrix(insimg0, inspred0, insflow0, insmask.clone())
        insimg1_warp, inspred1_warp, insmask1 = update_matrix(insimg1, inspred1, insflow1, insmask.clone())

        loss += criterion(F.log_softmax(inspred0_warp, 0)*insmask0, F.log_softmax(inspred1, 0)*insmask0).sum() / insmask0.sum()
        loss += criterion(F.log_softmax(inspred1_warp, 0)*insmask1, F.log_softmax(inspred0, 0)*insmask1).sum() / insmask1.sum()
        loss += criterion(F.log_softmax(inspred1, 0)*insmask0, F.log_softmax(inspred0_warp, 0)*insmask0).sum() / insmask0.sum()
        loss += criterion(F.log_softmax(inspred0, 0)*insmask1, F.log_softmax(inspred1_warp, 0)*insmask1).sum() / insmask1.sum()

        if True:
            plt.figure()
            plt.imshow(((insimg0.squeeze().cpu().numpy().transpose(1,2,0) * config.image_std + config.image_mean) * 255).astype(np.uint8))
            plt.figure()
            plt.imshow(((insimg0_warp.squeeze().cpu().numpy().transpose(1,2,0) * config.image_std + config.image_mean) * 255).astype(np.uint8))
            plt.figure()
            plt.imshow(((insimg1.squeeze().cpu().numpy().transpose(1,2,0) * config.image_std + config.image_mean) * 255).astype(np.uint8))
            plt.figure()
            plt.imshow(((insimg1_warp.squeeze().cpu().numpy().transpose(1,2,0) * config.image_std + config.image_mean) * 255).astype(np.uint8))
            plt.figure()
            plt.imshow(insmask.squeeze().cpu().numpy())
            plt.figure()
            plt.imshow(insmask0.squeeze().cpu().numpy())
            plt.figure()
            plt.imshow(insflow0.squeeze()[:,:,0].cpu().numpy())
            plt.show()

        #loss += criterion(F.log_softmax(inspred0_warp, 0), F.log_softmax(inspred1, 0)).mean()
        #loss += criterion(F.log_softmax(inspred1_warp, 0), F.log_softmax(inspred0, 0)).mean()
        #loss += criterion(F.log_softmax(inspred1, 0), F.log_softmax(inspred0_warp, 0)).mean()
        #loss += criterion(F.log_softmax(inspred0, 0), F.log_softmax(inspred1_warp, 0)).mean()

    loss /= n

    return loss

def main():
    parser = argparse.ArgumentParser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        cudnn.benchmark = True
        if engine.distributed:
            torch.cuda.set_device(engine.local_rank)

        opflow_model = torch.nn.DataParallel(RAFT(config))
        opflow_model.load_state_dict(torch.load(config.opflow_model_path))
        opflow_model = opflow_model.module
        opflow_model.cuda()
        opflow_model.eval()

        config.use_gt = False
        if config.use_gt is True:
            vis = VIS_STAUG_GT
            config.num_classes = 5
            config.data_root_path = "/home/duy/phd/lucasdu/duy/train_all_frames/Annotations/" + config.data_root_path.split('/')[-2] + '/'
            config.num_train_imgs = len(glob.glob(config.data_root_path + '/*png'))
            config.niters_per_epoch = int(np.ceil(config.num_train_imgs // config.batch_size))
        else:
            vis = VIS_VIDEO

        # data loader
        train_loader, train_sampler = get_train_loader(engine, vis, opflow_model)

        # config network and criterion
        criterion = nn.CrossEntropyLoss(reduction='none',
                                        ignore_index=255)
        criterion_cons = nn.KLDivLoss(reduction='none', log_target=True)
        #criterion_cons = nn.L1Loss(reduction='none')
        #criterion_cons = nn.MSELoss(reduction='none')
        loss_type = 'normal_ce'
        #loss_type = 'pixel_level_ohem'

        if engine.distributed:
            BatchNorm2d = SyncBatchNorm
        else:
            BatchNorm2d = nn.BatchNorm2d
        model = FCN(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
        init_weight(model.business_layer, nn.init.kaiming_normal_,
                    BatchNorm2d, config.bn_eps, config.bn_momentum,
                    mode='fan_out', nonlinearity='relu')

        # group weight and config optimizer
        base_lr = config.lr
        if engine.distributed:
            base_lr = config.lr * engine.world_size

        params_list = []
        params_list = group_weight(params_list, model,
                                   BatchNorm2d, base_lr)

        optimizer = torch.optim.SGD(params_list,
                                    lr=base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)

        # config lr policy
        total_iteration = config.nepochs * config.niters_per_epoch
        lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

        if engine.distributed:
            if torch.cuda.is_available():
                model.cuda()
                model = DistributedDataParallel(model,
                                                device_ids=[engine.local_rank],
                                                output_device=engine.local_rank)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DataParallelModel(model, engine.devices)
            model.to(device)

        engine.register_state(dataloader=train_loader, model=model,
                              optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()

        optimizer.zero_grad()
        model.train()

        for epoch in range(engine.state.epoch, config.nepochs):
            if engine.distributed:
                train_sampler.set_epoch(epoch)
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                        bar_format=bar_format)
            dataloader = iter(train_loader)
            for idx in pbar:
                engine.update_iteration(epoch, idx)

                minibatch = dataloader.next()
                imgs0 = minibatch['img0']
                gts0 = minibatch['gt0']
                imgs1 = minibatch['img1']
                gts1 = minibatch['gt1']
                flow0 = minibatch['flow0']
                flow1 = minibatch['flow1']
                mask = minibatch['mask']

                imgs0 = imgs0.cuda(non_blocking=True)
                gts0 = gts0.cuda(non_blocking=True)
                imgs1 = imgs1.cuda(non_blocking=True)
                gts1 = gts1.cuda(non_blocking=True)
                flow0 = flow0.cuda(non_blocking=True)
                flow1 = flow1.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                loss0, aux_pred0, pred0 = model(imgs0, gts0)
                loss1, aux_pred1, pred1 = model(imgs1, gts1)

                # reduce the whole loss over multi-gpu
                if engine.distributed:
                    dist.all_reduce(loss, dist.ReduceOp.SUM)
                    loss = loss / engine.world_size
                else:
                    #loss = Reduce.apply(*loss) / len(loss)
                    if loss_type == 'image_level_ohem':
                        loss, _ = loss.topk(k=int(0.6 * imgs.size()[0]), dim=0)
                        loss = loss.mean()
                    elif loss_type == 'image_level_oeem':
                        loss, _ = loss.topk(k=int(0.6 * imgs.size()[0]), dim=0)
                        loss = loss.mean()
                    elif loss_type == 'pixel_level_ohem':
                        loss = torch.flatten(loss)
                        loss, _ = loss.topk(k=int(0.6 * loss.size()[0]))
                        loss = loss.mean()
                    elif loss_type == 'pixel_level_oeem':
                        loss = torch.flatten(loss)
                        loss, _ = (-1 * loss).topk(k=int(0.6 * loss.size()[0]))
                        loss = (-1 * loss).mean()
                    elif loss_type == 'normal_ce':
                        norm_loss = loss0.mean() + loss1.mean()
                        cons_loss = calculate_cons_loss(imgs0, imgs1, pred0, pred1, flow0, flow1, mask, criterion_cons)
                        #loss = norm_loss + epoch / config.nepochs * cons_loss
                        loss = norm_loss + .5 * cons_loss

                optimizer.zero_grad()
                current_idx = epoch * config.niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)

                for i in range(0, len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr

                loss.backward()
                optimizer.step()
                print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' norm_loss=%.2f' % norm_loss.item() \
                            + ' cons_loss=%.2f' % cons_loss.item() \
                            + ' total_loss=%.2f' % loss.item()

                pbar.set_description(print_str, refresh=False)

            eval(model, str(epoch), config.use_gt)

            if epoch % config.snapshot_iter == 0:
                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint(config.snapshot_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(config.snapshot_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    main()
