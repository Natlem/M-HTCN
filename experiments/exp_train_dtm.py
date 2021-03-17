import os

import dtm_util
import frcnn_utils
from experiments.exp_utils import get_config_var, LoggerForSacred, Args
from init_frcnn_utils import init_dataloaders_1s_1t, init_val_dataloaders_mt, init_val_dataloaders_1t, \
    init_htcn_model, init_optimizer, init_dataloaders_1s_mt


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
import inc_frcnn_utils
import traineval_net_HTCN
from typing import Any
import re

@ex.config
def exp_config():
    # Most of the hyper-parameters are already in the CFG here is in case we want to override

    # config file
    cfg_file = "cfgs/vgg16.yml"
    output_dir = "all_saves/htcn_train_mask"
    dataset_source = "cs"


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
def exp_htcn_mixed(cfg_file, output_dir, dataset_source,
                    device, net, optimizer, num_workers,
                    lr, batch_size, start_epoch, max_epochs, lr_decay_gamma, lr_decay_step,
                    resume, load_name, pretrained,
                    eta, gamma, ef, class_agnostic, lc, gc, LA_ATT, MID_ATT,
                    debug, _run):

    args = Args(dataset=dataset_source, dataset_t=[], imdb_name_target=[], cfg_file=cfg_file, net=net)
    args = set_dataset_args(args)

    is_bgr = False
    if net in ['res101', 'res50', 'res152', 'vgg16']:
        is_bgr = True


    logger = LoggerForSacred(None, ex)

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

    dataloader_s, _, imdb, _ = init_dataloaders_1s_mt(args, batch_size, num_workers, is_bgr)

    session = 1
    fasterRCNN = init_htcn_model(LA_ATT, MID_ATT, class_agnostic, device, gc, imdb, lc, load_name, net, strict=False, pretrained=pretrained)


    dtm = nn.Sequential(nn.Conv2d(3, 256, 1, stride=1, padding=0, bias=False),
                         nn.ReLU(),
                         nn.Conv2d(256, 3, 1))


    dtm.to(device)
    optimizer = torch.optim.SGD(dtm.parameters(), lr=lr, momentum=0.9)


    if torch.cuda.device_count() > 1:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(10000 / batch_size)

    if ef:
        FL = EFocalLoss(class_num=2, gamma=gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=gamma)

    dtm_util.get_mask_for_target(args, FL, 0, dataloader_s, iters_per_epoch,
                                 fasterRCNN, dtm, optimizer, device, logger)

    find_id = re.findall("\d+", load_name)
    if len(find_id) == 0:
        find_id = 0
    else:
        find_id = re.findall("\d+", load_name)[-1]
    torch.save(dtm, os.path.join(output_dir, 'dtm_target_cnn_{}_{}.p'.format(load_id, find_id)))

    return 0


@ex.main
def run_exp():
    return exp_htcn_mixed()

if __name__ == "__main__":


    ex.run(config_updates={'cfg_file': 'cfgs/vgg16.yml',
                           'lr': 0.01,
                           'lr_decay_step': [5],
                           'max_epochs': 12,
                           'net': 'vgg16',
                           'pretrained': True,
                           "load_name": "path_of_previous_trained_detection_model",
                           'dataset_source': 'cs'
                           },

           options={"--name": 'htcn_train_dtm'})


