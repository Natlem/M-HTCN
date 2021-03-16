import os

import dtm_util
import frcnn_utils
from experiments.exp_utils import get_config_var, LoggerForSacred, Args
from init_frcnn_utils import init_dataloaders_1s_1t, init_val_dataloaders_mt, init_val_dataloaders_1t, \
    init_htcn_model, init_optimizer, init_dataloaders_1s_mt

vars = get_config_var()
from sacred import Experiment
ex = Experiment()
from sacred.observers import MongoObserver
if True:
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
import inc_frcnn_utils
import traineval_net_HTCN
from typing import Any
import re

@ex.config
def exp_config():
    # Most of the hyper-parameters are already in the CFG here is in case we want to override

    # config file
    cfg_file = "cfgs/res101.yml"
    output_dir = "all_saves/htcn_train_mask"
    dataset_source = "kitti_car_trainval"
    dataset_targets = ["cs_car"]
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

    resume = False
    load_name = ""
    pretrained = True #Whether to use pytorch models (FAlse) or caffe model (True)

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
def exp_htcn_mixed(cfg_file, output_dir, dataset_source, dataset_targets, val_datasets,
                    device, net, optimizer, num_workers,
                    lr, batch_size, start_epoch, max_epochs, lr_decay_gamma, lr_decay_step,
                    resume, load_name, pretrained,
                    eta, gamma, ef, class_agnostic, lc, gc, LA_ATT, MID_ATT,
                    debug, _run):

    args = Args(dataset=dataset_source, dataset_t=dataset_targets, imdb_name_target=[], cfg_file=cfg_file, net=net)
    args = set_dataset_args(args)

    args_val = Args(dataset=dataset_source, dataset_t=val_datasets, imdb_name_target=[], cfg_file=cfg_file, net=net)
    args_val = set_dataset_args(args_val, test=True)

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

    load_id = re.findall("\d+", load_name)[0]
    output_dir = output_dir + "_{}".format(load_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataloader_s, dataloader_ts, imdb, imdb_t = init_dataloaders_1s_mt(args, batch_size, num_workers, is_bgr)
    val_dataloader_ts, val_imdb_ts = init_val_dataloaders_mt(args_val, 1, num_workers, is_bgr)

    session = 1
    fasterRCNN = init_htcn_model(LA_ATT, MID_ATT, class_agnostic, device, gc, imdb, lc, load_name, net, strict=False, pretrained=pretrained)


    mask = nn.Sequential(nn.Conv2d(3, 256, 1, stride=1, padding=0, bias=False),
                         nn.ReLU(),
                         nn.Conv2d(256, 3, 1))


    mask.to(device)
    optimizer = torch.optim.SGD(mask.parameters(), lr=0.01, momentum=0.9)


    if torch.cuda.device_count() > 1:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(10000 / batch_size)

    if ef:
        FL = EFocalLoss(class_num=2, gamma=gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=gamma)

    dtm_util.get_mask_for_target(args, FL, 0, dataloader_s, dataloader_ts[:-1], iters_per_epoch,
                                 fasterRCNN, mask, optimizer, device, logger)

    torch.save(mask, os.path.join(output_dir, 'mask_target_cnn_{}_{}.p'.format(load_id, re.findall("\d+", load_name)[-1])))

    # ap = current_exp_pg.eval_mask_for_target(args, FL, 0, val_dataloader_ts[0], dataloader_ts[:-1], iters_per_epoch,
    #                                                                fasterRCNN, mask, optimizer, device, logger)



    return 0


@ex.main
def run_exp():
    return exp_htcn_mixed()

if __name__ == "__main__":

    # ex.run(config_updates={'cfg_file': 'cfgs/vgg16.yml',
    #                        'lr': 0.01,
    #                        'lr_decay_step': [5],
    #                        'max_epochs': 12,
    #                        'net': 'vgg16',
    #                        'pretrained': True,
    #                        "load_name": "./all_saves/htcn_single_444/target_cs_fg_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_6_total_step_60000.pth",
    #                        'dataset_source': 'cs',
    #                        'dataset_targets': ['cs_rain'],
    #                        'val_datasets': ['cs_fg', 'cs_rain']},
    #
    #        options={"--name": 'htcn_mask_train_csfg_vgg_444_60k_pascal'})

    ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
                           'lr': 0.01,
                           'lr_decay_step': [5],
                           'max_epochs': 12,
                           'net': 'res50',
                           'pretrained': True,
                           "load_name":"./all_saves/htcn_inc_mask_472/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_3_total_step_30000.pth",
                           'dataset_source': 'voc_0712',
                           'dataset_targets': ['watercolor'],
                           'val_datasets': ['watercolor']},

           options={"--name": 'htcn_mask_train_watercolor_res50_472_30k'})

    # ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
    #                        'lr': 0.01,
    #                        'lr_decay_step': [5],
    #                        'max_epochs': 7,
    #                        'net': 'res50',
    #                        'pretrained': True,
    #                         "load_name": "./all_saves/htcn_inc_mask_340/target_clipart_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_6_total_step_60000.pth",
    #                        'dataset_source': 'voc_0712',
    #                        'dataset_targets': ['clipart'],
    #                        'val_datasets': ['clipart']},
    #
    #        options={"--name": 'htcn_mask_train_comic_clip_resnet50_340_60k'})


    # ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
    #                        'lr': 0.01,
    #                        'lr_decay_step': [5],
    #                        'max_epochs': 12,
    #                        'net': 'res50',
    #                        'pretrained': True,
    #                        'load_name': "./all_saves/htcn_inc_mask_289/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_5_total_step_50000.pth",
    #                        'dataset_source': 'voc_0712',
    #                        'dataset_targets': ['clipart'],
    #                        'val_datasets': ['clipart']},
    #
    #        options={"--name": 'htcn_mask_train_clip_water_2_comic_resnet50_289_50k'})
    #
    # ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
    #                        'lr': 0.01,
    #                        'lr_decay_step': [5],
    #                        'max_epochs': 12,
    #                        'net': 'res101',
    #                        'pretrained': True,
    #                        #'load_name': "./all_saves/htcn_inc_mask_289/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_5_total_step_50000.pth",
    #                        #'load_name': "./all_saves/htcn_inc_mask_269/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_7_total_step_70000.pth",
    #                        #'load_name': "./all_saves/htcn_single_206/target_clipart_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_4_total_step_40000.pth",
    #                        'load_name': "./all_saves/htcn_inc_mask_365/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_3_total_step_30000.pth",
    #                        'dataset_source': 'voc_0712',
    #                        'dataset_targets': ['clipart'],
    #                        'val_datasets': ['clipart']},
    #
    #        options={"--name": 'htcn_mask_train_clip_watercolor_resnet101_365_30k'})

    # ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
    #                        'lr': 0.0001,
    #                        'lr_decay_step': [5],
    #                        'max_epochs': 12,
    #                        'net': 'res50',
    #                        'pretrained': True,
    #                        'load_name': './all_saves/htcn_single_212/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_7_total_step_70000.pth',
    #                        'dataset_source': 'voc_0712',
    #                        'dataset_target': 'comic',
    #                        'val_datasets': ['clipart', 'watercolor', 'comic']},
    #
    #        options={"--name": 'htcn_clipart_watercolor_2_comic_res50_0.0001_lr'})