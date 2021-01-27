from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from semantic_seg import SemanticSegmentor
from backbone import NaiveSyncBatchNorm
from fpn_config import get_cfg

from dataloader import get_train_loader
from datasets.coco_box2seg import COCO

from utils.init_func import build_optimizer
from utils.pyt_utils import all_reduce_tensor
from engine.lr_policy import PolyLR
from engine.logger import get_logger
from engine.engine import Engine

from detectron2.checkpoint import DetectionCheckpointer

try:
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

logger = get_logger()

torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, COCO)

    # config network and criterion
    #criterion = nn.CrossEntropyLoss(reduction='mean',
    #                                ignore_index=255)

    if engine.distributed:
        logger.info('Use the Multi-Process-SyncBatchNorm')

    cfg = get_cfg()
    cfg.merge_from_file('./fpn_config/semantic_R_50_FPN_1x.yaml')
    model = SemanticSegmentor(cfg)

    #DetectionCheckpointer(model).resume_or_load('/home/duy/.cache/torch/checkpoints/R-101.pkl', resume=False)
    model.load_state_dict(torch.load(config.pretrained_model), strict=False)
    #model.load_state_dict(torch.load(config.pretrained_model))

    # group weight and config optimizer
    base_lr = config.lr

    params_list = build_optimizer(cfg, model)
    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum)
    #optimizer = torch.optim.Adam(params_list,
    #                            lr=base_lr)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if engine.distributed:
        model = DistributedDataParallel(model)

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
            imgs = minibatch['img']
            gts = minibatch['gt']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            loss = model(imgs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss,
                                                world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % reduce_loss.item()

            pbar.set_description(print_str, refresh=False)

        if (epoch >= config.nepochs - 20) or (
                epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
