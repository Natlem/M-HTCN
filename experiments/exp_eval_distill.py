import os

import frcnn_utils
import distill_frcnn_utils
from experiments.exp_utils import get_config_var, LoggerForSacred, Args
import init_frcnn_utils

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

from model.faster_rcnn.resnet import resnet as n_resnet
from model.faster_rcnn.resnet_saito import resnet as s_resnet
from model.faster_rcnn.resnet_HTCN import resnet as htcn_resnet
from model.faster_rcnn.vgg16 import vgg16 as n_vgg16
from model.faster_rcnn.vgg16_HTCN import vgg16 as htcn_vgg16
from model.faster_rcnn.vgg16_HTCN_mrpn import vgg16 as vgg16_mrpn

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
    teacher_pth = ''
    student_pth = ''

    lr = 0.001
    batch_size = 1
    start_epoch = 1
    max_epochs = 7
    lr_decay_gamma = 0.1
    lr_decay_step = [5]

    resume = False
    load_name = ""

    imitation_loss_weight = 0.01
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
                    device, net, optimizer, num_workers, teacher_pth, student_pth,
                    lr, batch_size, start_epoch, max_epochs, lr_decay_gamma, lr_decay_step,
                    resume, load_name,
                    imitation_loss_weight, eta, gamma, ef, class_agnostic, lc, gc, LA_ATT, MID_ATT,
                    debug, _run):

    args_val = Args(dataset=dataset_source, dataset_t=val_datasets, imdb_name_target=[], cfg_file=cfg_file, net=net)
    args_val = set_dataset_args(args_val, test=True)


    logger = LoggerForSacred(None, ex, False)

    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if args_val.set_cfgs is not None:
        cfg_from_list(args_val.set_cfgs)

    np.random.seed(cfg.RNG_SEED)

    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = True if device == 'cuda' else False
    device = torch.device(device)

    val_dataloader_ts, val_imdb_ts = init_frcnn_utils.init_val_dataloaders_mt(args_val, 1, num_workers)

    session = 1
    teacher = init_frcnn_utils.init_model_only(device, "res101", htcn_resnet, val_imdb_ts[0], teacher_pth, class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    fasterRCNN = init_frcnn_utils.init_model_only(device, "res50", htcn_resnet, val_imdb_ts[0], student_pth, class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    fasterRCNN_2 = init_frcnn_utils.init_model_only(device, "res50", htcn_resnet, val_imdb_ts[0], student_pth, class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)

    fasterRCNN.RCNN_rpn = teacher.RCNN_rpn




    if torch.cuda.device_count() > 1:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    total_step = 0
    best_ap = 0.
    if isinstance(val_datasets, list):
        avg_ap = 0
        for i, val_dataloader_t in enumerate(val_dataloader_ts):
            map = frcnn_utils.eval_one_dataloader(output_dir, val_dataloader_t, fasterRCNN, device, val_imdb_ts[i])
            logger.log_scalar("student with teacher rpn map on {}".format(val_datasets[i]), map, 0)
            map = frcnn_utils.eval_one_dataloader(output_dir, val_dataloader_t, teacher, device, val_imdb_ts[i])
            logger.log_scalar("teacher original map on {}".format(val_datasets[i]), map, 0)
            teacher.RCNN_rpn = fasterRCNN_2.RCNN_rpn
            map = frcnn_utils.eval_one_dataloader(output_dir, val_dataloader_t, teacher, device, val_imdb_ts[i])
            logger.log_scalar("teacher with stu rpn map on {}".format(val_datasets[i]), map, 0)




@ex.main
def run_exp():
    return exp_htcn_mixed()

if __name__ == "__main__":

    ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
                           'net': 'res50',
                           'lr': 0.001,
                           'lr_decay_step': [5],
                           'dataset_source': 'voc_0712',
                           'dataset_target': 'watercolor',
                           'val_datasets': ['watercolor'],
                           'teacher_pth': './all_saves/htcn_single_135/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_6_total_step_60000.pth',
                           'student_pth': './all_saves/htcn_single_141/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_4_total_step_40000.pth'},

           options={"--name": 'htcn_voc_2_watercolor_res50_eval_with_teacher_rpn'})