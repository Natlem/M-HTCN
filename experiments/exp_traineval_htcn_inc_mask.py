import os

import frcnn_utils
from experiments.exp_utils import get_config_var, LoggerForSacred, Args
from init_frcnn_utils import init_dataloaders_1s_1t, init_val_dataloaders_mt, init_val_dataloaders_1t, \
    init_htcn_model, init_optimizer

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

import traineval_net_HTCN
from typing import Any
import dtm_util

#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True

@ex.config
def exp_config():
    # Most of the hyper-parameters are already in the CFG here is in case we want to override

    # config file
    cfg_file = "cfgs/res101.yml"
    output_dir = "all_saves/htcn_inc_mask"
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

    eta = 1
    gamma = 3
    ef = False
    class_agnostic = False
    lc = True
    gc = True
    LA_ATT = True
    MID_ATT = True
    alpha = 1


    debug = True

@ex.capture
def exp_htcn_mixed(cfg_file, output_dir, dataset_source, dataset_target, val_datasets,
                    device, net, optimizer, num_workers,
                    lr, batch_size, start_epoch, max_epochs, lr_decay_gamma, lr_decay_step,
                    mask_load_p, resume, load_name, pretrained,
                    eta, gamma, ef, class_agnostic, lc, gc, LA_ATT, MID_ATT, alpha,
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
    fasterRCNN = init_htcn_model(LA_ATT, MID_ATT, class_agnostic, device, gc, imdb, lc, load_name, net, strict=False, pretrained=pretrained)
    lr, optimizer, session, start_epoch = init_optimizer(lr, fasterRCNN, optimizer, resume, load_name, session, start_epoch, is_all_params=True)
    #fasterRCNN.init_new_adv(device)
    mask = torch.load(mask_load_p)
    mask.to(device)
    mask_dict = {}
    mask_dict['clipart_mask'] = mask

    if torch.cuda.device_count() > 1:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(10000 / batch_size)

    if ef:
        FL = EFocalLoss(class_num=2, gamma=gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=gamma)

    total_step = 0
    best_ap = 0.

    if isinstance(val_datasets, list):
        avg_ap = 0
        for i, val_dataloader_t in enumerate(val_dataloader_ts):
            map = frcnn_utils.eval_one_dataloader(output_dir, val_dataloader_t, fasterRCNN, device, val_imdb_ts[i])
            logger.log_scalar("map on {}".format(val_datasets[i]), map, 0)
            avg_ap += map / len(val_dataloader_ts)
        logger.log_scalar("avg map on", avg_ap, total_step)
        if avg_ap > best_ap:
            best_ap = avg_ap

    for epoch in range(start_epoch, max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()

        if epoch - 1 in lr_decay_step:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        total_step = dtm_util.train_htcn_one_epoch_ida_with_dtm(args, FL, total_step, dataloader_s, mask_dict, dataloader_t, iters_per_epoch, fasterRCNN, optimizer, device, eta, alpha, logger)

        if isinstance(val_datasets, list):
            avg_ap = 0
            for i, val_dataloader_t in enumerate(val_dataloader_ts):
                map = frcnn_utils.eval_one_dataloader(output_dir, val_dataloader_t, fasterRCNN, device, val_imdb_ts[i])
                logger.log_scalar("map on {}".format(val_datasets[i]), map, total_step)
                avg_ap += map / len(val_dataloader_ts)
            logger.log_scalar("avg map on", avg_ap, total_step)
            if avg_ap > best_ap:
                best_ap = avg_ap
                save_best_name = "{}_best_map.p_ds_{}_2_dt_{}_on_{}".format(_run._id, dataset_source, dataset_target, net)
                if torch.cuda.device_count() > 1:
                    torch.save(fasterRCNN.module, os.path.join())

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
    return best_ap.item()


@ex.main
def run_exp():
    return exp_htcn_mixed()

if __name__ == "__main__":

    ex.run(config_updates={'cfg_file': 'cfgs/vgg16.yml',
                           'lr': 0.0001,
                           'lr_decay_step': [2],
                           'max_epochs': 7,
                           'net': 'vgg16',
                           'pretrained': True,
                           'eta': 1,
                           'alpha': 0.5,
                           'mask_load_p': 'all_saves/htcn_train_mask_444/mask_target_cnn_444_60000.p',
                           "load_name": "./all_saves/htcn_single_444/target_cs_fg_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_6_total_step_60000.pth",
                           'dataset_source': 'cs',
                           'dataset_target': 'cs_rain',
                           'val_datasets': ['cs_fg', 'cs_rain']},

           options={"--name": 'htcn_inc_mask_cs_2_fg_vgg16_60k_444_mask_da_0.5'})

    # ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
    #                        'lr': 0.0001,
    #                        'lr_decay_step': [5],
    #                        'max_epochs': 7,
    #                        'net': 'res50',
    #                        'pretrained': True,
    #                        'alpha': 0.1,
    #                        'mask_load_p': "all_saves/htcn_train_mask_472/mask_target_cnn_472_30000.p",
    #                        'load_name': "./all_saves/htcn_inc_mask_472/target_watercolor_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_3_total_step_30000.pth",
    #                        'dataset_source': 'voc_0712',
    #                        'dataset_target': 'comic',
    #                        'val_datasets': ['clipart', 'watercolor', 'comic']},
    #
    #        options={"--name": 'htcn_abl_inc_mask_clipart_watercolor_2_comic_res50_30k_472_alpha_0.1'})

    # ex.run(config_updates={'cfg_file': 'cfgs/res50.yml',
    #                        'lr': 0.0001,
    #                        'lr_decay_step': [5],
    #                        'max_epochs': 7,
    #                        'net': 'res50',
    #                        'pretrained': True,
    #                        'mask_load_p': "all_saves/htcn_train_mask_384/mask_target_cnn_384_10000.p",
    #                        'load_name': "./all_saves/htcn_inc_mask_384/target_comic_eta_0.1_local_True_global_True_gamma_3_session_1_epoch_1_total_step_10000.pth",
    #                        'dataset_source': 'voc_0712',
    #                        'dataset_target': 'clipart',
    #                        'val_datasets': ['clipart','watercolor','comic']},
    #
    #        options={"--name": 'htcn_abl_inc_mask_watercolor_comic_2_clipart_res50_10k_384'})

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
