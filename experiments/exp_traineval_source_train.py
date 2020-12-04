import os

import frcnn_utils
import init_frcnn_utils
from experiments.exp_utils import get_config_var, LoggerForSacred, Args
from init_frcnn_utils import init_dataloaders_1s_1t, init_val_dataloaders_mt, init_val_dataloaders_1t, \
    init_model_optimizer

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

@ex.config
def exp_config():
    # Most of the hyper-parameters are already in the CFG here is in case we want to override

    # config file
    cfg_file = "cfgs/vgg16.yml"
    output_dir = "all_saves/frcnn_source_train"
    # dataset_source = "voc_0712"
    # val_datasets = ["comic", "clipart", "watercolor"]
    dataset_source = "wildtrack_C1"
    val_datasets = ["wildtrack_C2", "wildtrack_C3", "wildtrack_C4", "wildtrack_C5", "wildtrack_C6", "wildtrack_C7"]


    device = "cuda"
    net = "vgg16"
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

    class_agnostic = False


    debug = True

@ex.capture
def exp_htcn_mixed(cfg_file, output_dir, dataset_source, val_datasets,
                    device, net, optimizer, num_workers,
                    lr, batch_size, start_epoch, max_epochs, lr_decay_gamma, lr_decay_step,
                    resume, load_name, class_agnostic,
                    debug, _run):

    args = Args(dataset=dataset_source, dataset_t="", cfg_file=cfg_file, net=net)
    args = set_dataset_args(args)

    args_val = Args(dataset=dataset_source, dataset_t=val_datasets, imdb_name_target=[], cfg_file=cfg_file, net=net)
    args_val = set_dataset_args(args_val, test=True)


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

    dataloader_s, _, imdb = init_dataloaders_1s_1t(args, batch_size, num_workers)
    val_dataloader_ts, val_imdb_ts = init_val_dataloaders_mt(args_val, 1, num_workers)

    session = 1
    fasterRCNN, lr, optimizer, session, start_epoch, _ = init_frcnn_utils.init_non_damodel_optimizer(lr, class_agnostic, device,
                                                                                                     imdb, load_name, net, optimizer, resume,
                                                                                                     session, start_epoch)

    if torch.cuda.device_count() > 1:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(10000 / batch_size)

    total_step = 0
    best_ap = 0.
    for epoch in range(start_epoch, max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()

        if epoch - 1 in lr_decay_step:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        total_step = frcnn_utils.train_no_da_frcnn_one_epoch(args, total_step, dataloader_s, iters_per_epoch, fasterRCNN, optimizer, device, logger)
        if isinstance(val_datasets, list):
            avg_ap = 0
            for i, val_dataloader_t in enumerate(val_dataloader_ts):
                map = frcnn_utils.eval_one_dataloader(output_dir, val_dataloader_t, fasterRCNN, device, val_imdb_ts[i])
                logger.log_scalar("map on {}".format(val_datasets[i]), map, total_step)
                avg_ap += map / len(val_dataloader_ts)
            logger.log_scalar("avg map on", avg_ap, total_step)
            if avg_ap > best_ap:
                best_ap = avg_ap
        

        save_name = os.path.join(output_dir,
                                 'source_train_{}_session_{}_epoch_{}_total_step_{}.pth'.format(dataset_source,
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
    return best_ap


@ex.main
def run_exp():
    return exp_htcn_mixed()

if __name__ == "__main__":

    ex.run(config_updates={'cfg_file': 'cfgs/vgg16.yml'},
           options={"--name": 'htcn_source_train_wildtrack_c1_2_rest_vgg16'})