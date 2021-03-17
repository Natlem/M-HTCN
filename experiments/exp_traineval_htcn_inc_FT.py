import os

import frcnn_utils
from experiments.exp_utils import get_config_var, LoggerForSacred, Args
from init_frcnn_utils import init_dataloaders_1s_1t, init_val_dataloaders_mt, init_val_dataloaders_1t, \
    init_htcn_model, init_optimizer

from sacred import Experiment
ex = Experiment()
from sacred.observers import MongoObserver
enable_mongo_observer = False
if enable_mongo_observer:
    vars = get_config_var()
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

@ex.config
def exp_config():
    # Most of the hyper-parameters are already in the CFG here is in case we want to override

    # config file
    cfg_file = "cfgs/res101.yml"
    output_dir = "all_saves/htcn_single"
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

    resume = False
    load_name = ""
    pretrained = True #Whether to use pytorch models (FAlse) or caffe model (True)

    eta = 1
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
                    resume, load_name, pretrained,
                    eta, gamma, ef, class_agnostic, lc, gc, LA_ATT, MID_ATT,
                    debug, _run):

    args = Args(dataset=dataset_source, dataset_t=dataset_target, cfg_file=cfg_file, net=net)
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

    output_dir = output_dir + "_{}".format(_run._id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataloader_s, dataloader_t, imdb, imdb_t = init_dataloaders_1s_1t(args, batch_size, num_workers, is_bgr)
    val_dataloader_ts, val_imdb_ts = init_val_dataloaders_mt(args_val, 1, num_workers, is_bgr)

    session = 1
    fasterRCNN = init_htcn_model(LA_ATT, MID_ATT, class_agnostic, device, gc, imdb, lc, load_name, net, pretrained=pretrained)
    lr, optimizer, session, start_epoch = init_optimizer(lr, fasterRCNN, optimizer, resume, load_name, session, start_epoch, is_all_params=True)


    if torch.cuda.device_count() > 1:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(10000 / batch_size)

    if ef:
        FL = EFocalLoss(class_num=2, gamma=gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=gamma)

    total_step = 0

    for epoch in range(start_epoch, max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()

        if epoch - 1 in lr_decay_step:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        total_step = frcnn_utils.train_htcn_one_epoch(args, FL, total_step, dataloader_s, dataloader_t, iters_per_epoch, fasterRCNN, optimizer, device, eta, logger)
        save_name = os.path.join(output_dir,
                                 'target_{}_eta_{}_local_{}_global_{}_gamma_{}_session_{}_epoch_{}_total_step_{}.pth'.format(
                                     args.dataset_t, args.eta,
                                     lc, gc, gamma,
                                     session, epoch,
                                     total_step))
        save_checkpoint({
            'session': session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if torch.cuda.device_count() > 1 else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': class_agnostic,
        }, save_name)
    return 0


@ex.main
def run_exp():
    return exp_htcn_mixed()

if __name__ == "__main__":

    ex.run(config_updates={'cfg_file': 'cfgs/vgg16.yml',
                           'lr': 0.0001,
                           'lr_decay_step': [5],
                           'max_epochs': 7,
                           'net': 'vgg16',
                           'pretrained': True,
                           'eta': 1,
                           'load_name':  "path_of_previous_trained_detection_model",
                           'dataset_source': 'cs',
                           'dataset_target': 'cs_rain',
                           'val_datasets': ['cs_fg', 'cs_rain']},

           options={"--name": 'htcn_inc_FT_cs_csfg_2_cs_rain_vgg16'})

