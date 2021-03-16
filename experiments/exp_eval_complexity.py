import os

import frcnn_utils
from experiments.exp_utils import get_config_var, LoggerForSacred, Args
from init_frcnn_utils import init_dataloaders_1s_1t, init_val_dataloaders_mt, init_val_dataloaders_1t, \
    init_htcn_model, init_optimizer
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import re

vars = get_config_var()
from sacred import Experiment
ex = Experiment()
from sacred.observers import MongoObserver
if False:
    ex.observers.append(MongoObserver(url='mongodb://{}:{}@{}/admin?authMechanism=SCRAM-SHA-1'.format(vars["SACRED_USER"],
                                                                                                      vars["SACRED_PWD"],
                                                                                                      vars["SACRED_URL"]),
                                      db_name=vars["SACRED_DB"]))
    ex.captured_out_filter = lambda text: 'Output capturing turned off.'

from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn

from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, FocalLoss, EFocalLoss


from model.utils.parser_func import set_dataset_args

import traineval_net_HTCN
from typing import Any
import dtm_util
from thop import profile, clever_format
import init_frcnn_utils

from model.faster_rcnn.resnet import resnet as n_resnet
from model.faster_rcnn.resnet_saito import resnet as s_resnet
from model.faster_rcnn.resnet_HTCN import resnet as htcn_resnet
from model.faster_rcnn.vgg16 import vgg16 as n_vgg16
from model.faster_rcnn.vgg16_HTCN import vgg16 as htcn_vgg16
from model.faster_rcnn.vgg16_HTCN_mrpn import vgg16 as vgg16_mrpn

@ex.config
def exp_config():
    # Most of the hyper-parameters are already in the CFG here is in case we want to override

    # config file
    cfg_file = "cfgs/res101.yml"
    output_dir = "all_saves/htcn_gen_img_mask"
    dataset_source = "kitti_car_trainval"
    dataset_target = "cs_car"
    val_datasets = ["cs_car"]


    device = "cuda"
    net = "res101"
    optimizer = "sgd"
    num_workers = 0

    lr = 0.001
    batch_size = 1
    start_epoch = 1
    max_epochs = 7
    lr_decay_gamma = 0.1
    lr_decay_step = [5]

    mask_load_p = ""
    resume = False
    load_name = ""
    pretrained = True #Whether to use pytorch models (FAlse) or caffe model (True)
    model_type = 'htcn'

    eta = 0.1
    gamma = 3
    ef = False
    class_agnostic = False
    lc = True
    gc = True
    LA_ATT = True
    MID_ATT = True


    debug = True

@ex.capture
def exp_htcn_mixed(cfg_file, output_dir, dataset_source, dataset_target, val_datasets,
                    device, net, optimizer, num_workers,
                    lr, batch_size, start_epoch, max_epochs, lr_decay_gamma, lr_decay_step,
                    mask_load_p, resume, load_name, pretrained, model_type,
                    eta, gamma, ef, class_agnostic, lc, gc, LA_ATT, MID_ATT,
                    debug, _run):

    args = Args(dataset=dataset_source, dataset_t=dataset_target, cfg_file=cfg_file, net=net)
    args = set_dataset_args(args)
    is_bgr = False
    if net in ['res101', 'res50', 'res152', 'vgg16']:
        is_bgr = True


    logger = LoggerForSacred(None, ex, False)

    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    np.random.seed(cfg.RNG_SEED)

    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = True if device == 'cuda' else False
    device = torch.device(device)

    backbone_fn = htcn_resnet
    if 'res' in net:
        if model_type == 'normal':
            backbone_fn = n_resnet
        elif model_type == 'saitp':
            backbone_fn = s_resnet
    else:
        if model_type == 'normal':
            backbone_fn = n_vgg16
        elif model_type == 'htcn':
            backbone_fn = htcn_vgg16
        elif model_type == 'saitp':
            backbone_fn = None
    dataloader_s, dataloader_t, imdb, imdb_t = init_dataloaders_1s_1t(args, batch_size, num_workers, is_bgr, False)
    model = init_frcnn_utils.init_model_only(device, net, backbone_fn, imdb_t, '', class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    model.eval()

    im_data = torch.randn(1, 3, 600, 1200).to(device)
    im_info = torch.FloatTensor([[600, 900, 2]]).to(device)
    gt_boxes = torch.zeros((1, 1, 5)).to(device)
    num_boxes = torch.zeros([1]).to(device)
    macs, params = profile(model, inputs=(im_data,im_info, gt_boxes, num_boxes))
    macs, params = clever_format([macs, params], "%.3f")

    print("Model CFLOPS: {}".format(macs))
    print("Model Cparams: {}".format(params))


    random_mask =   nn.Sequential(nn.Conv2d(3, 256, 1, stride=1, padding=0, bias=False),
                         nn.ReLU(),
                         nn.Conv2d(256, 3, 1)).to(device)

    macs, params = profile(random_mask, inputs=(im_data,))
    macs, params = clever_format([macs, params], "%.3f")

    print("Mask CFLOPS: {}".format(macs))
    print("Mask Cparams: {}".format(params))

    iters_per_epoch = int(1000 / batch_size)
    data_iter_s = iter(dataloader_s)


    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)
        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)




        pass




@ex.main
def run_exp():
    return exp_htcn_mixed()

if __name__ == "__main__":

    # ex.run(config_updates={'cfg_file': 'cfgs/vgg16.yml',
    #                        'lr': 0.0001,
    #                        'lr_decay_step': [3],
    #                        'max_epochs': 7,
    #                        'net': 'vgg16',
    #                        'pretrained': True,
    #                        'mask_load_p': 'all_saves/htcn_train_mask_312/mask_target_cnn_312_50000.p',
    #                        "load_name": "./all_saves/htcn_single_312/target_cs_fg_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_5_total_step_50000.pth",
    #                        'dataset_source': 'cs',
    #                        'dataset_target': 'cs_rain',
    #                        'val_datasets': ['cs_fg', 'cs_rain']},
    #
    #        options={"--name": 'htcn_inc_mask_cs_2_cs_rain_vgg16_50k_312'})

    ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
                           'lr': 0.0001,
                           'lr_decay_step': [5],
                           'max_epochs': 7,
                           'net': 'res50',
                           'pretrained': True,
                           'dataset_source': 'voc_0712',
                           'dataset_target': 'clipart',
                           'val_datasets': ['clipart']
                           },

           options={"--name": 'htcn_eval_comp_res50'})

    # ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
    #                        'lr': 0.0001,
    #                        'lr_decay_step': [5],
    #                        'max_epochs': 7,
    #                        'net': 'res101',
    #                        'pretrained': True,
    #                        #'mask_load_p': 'all_saves/htcn_train_mask_208/mask_target_cnn.p',
    #                        #'mask_load_p': 'all_saves/htcn_train_mask_206/mask_target_cnn_70000.p',
    #                        'mask_load_p': "all_saves/htcn_train_mask_208/mask_target_cnn_208_40000.p",
    #                        # 'mask_load_p': 'all_saves/htcn_train_mask_269/mask_target_cnn_clip_water_2_comic_40000.p',
    #                        #'mask_load_p': 'all_saves/htcn_train_mask_269/mask_target_cnn_clip_water_2_comic_70000.p',
    #                        #'load_name': "./all_saves/htcn_inc_mask_269/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_7_total_step_70000.pth",
    #                        'load_name': "./all_saves/htcn_single_208/target_clipart_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_4_total_step_40000.pth",
    #                        #'load_name': "./all_saves/htcn_single_206/target_clipart_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_7_total_step_70000.pth",
    #                        #'load_name': './all_saves/htcn_single_208/target_clipart_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_10_total_step_100000.pth',
    #                        'dataset_source': 'voc_0712',
    #                        'dataset_target': 'watercolor',
    #                        'val_datasets': ['clipart', 'watercolor']},
    #
    #        options={"--name": 'htcn_inc_mask_clipart_2_watercolor_res101_40k_208'})
