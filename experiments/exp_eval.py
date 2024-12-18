import frcnn_utils
from experiments.exp_utils import get_config_var, LoggerForSacred, Args
import init_frcnn_utils


from sacred import Experiment
ex = Experiment()
from sacred.observers import MongoObserver
enable_mongo_observer = False
if enable_mongo_observer:
    vars = get_config_var(is_eval=True)
    ex.observers.append(MongoObserver(url='mongodb://{}:{}@{}/admin?authMechanism=SCRAM-SHA-1'.format(vars["SACRED_USER"],
                                                                                                      vars["SACRED_PWD"],
                                                                                                      vars["SACRED_URL"]),
                                      db_name=vars["SACRED_DB"]))
    ex.captured_out_filter = lambda text: 'Output capturing turned off.'

import numpy as np

import torch

from model.utils.config import cfg, cfg_from_file, cfg_from_list

from model.faster_rcnn.resnet import resnet as n_resnet
from model.faster_rcnn.resnet_saito import resnet as s_resnet
from model.faster_rcnn.resnet_HTCN import resnet as htcn_resnet
from model.faster_rcnn.vgg16 import vgg16 as n_vgg16
from model.faster_rcnn.vgg16_HTCN import vgg16 as htcn_vgg16

from model.utils.parser_func import set_dataset_args


@ex.config
def exp_config():
    # Most of the hyper-parameters are already in the CFG here is in case we want to override

    # config file
    cfg_file = "cfgs/res101.yml"
    output_dir = "all_saves/htcn_single"
    dataset_source = "kitti_car_trainval"
    dataset_target = "cs_car"
    val_datasets = ["cs_car"]

    model_type = 'htcn' # can be 'htcn', 'normal', 'saito'
    device = "cuda"
    net = "res101"
    optimizer = "sgd"
    num_workers = 0
    model_pth = ''

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
                    model_type, device, net, optimizer, num_workers, model_pth, class_agnostic, lc, gc, LA_ATT, MID_ATT,
                    debug, _run):

    args_val = Args(dataset=dataset_source, dataset_t=val_datasets, imdb_name_target=[], cfg_file=cfg_file, net=net)
    args_val = set_dataset_args(args_val, test=True)


    logger = LoggerForSacred(None, ex, True)

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


    model = init_frcnn_utils.init_model_only(device, net, backbone_fn, val_imdb_ts[0], model_pth, class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)



    total_step = 0
    best_ap = 0.
    avg_ap = 0.
    avg_ap_per_class = {}
    if isinstance(val_datasets, list):
        for i, val_dataloader_t in enumerate(val_dataloader_ts):
            map, ap_per_class = frcnn_utils.eval_one_dataloader(output_dir, val_dataloader_t, model, device, val_imdb_ts[i], return_ap_class=True)
            logger.log_scalar(" map on {}".format(val_datasets[i]), map, 0)
            for cls, ap in ap_per_class.items():
                if cls in avg_ap_per_class:
                    avg_ap_per_class[cls] += ap
                else:
                    avg_ap_per_class[cls] = ap
            avg_ap += map
        avg_ap /= len(val_dataloader_ts)
        for cls, ap in avg_ap_per_class.items():
            ap /= len(val_dataloader_ts)
            logger.log_scalar(" map of class {}".format(cls), ap, 0)
    logger.log_scalar("avp map",avg_ap, 0)

    return avg_ap.item()




@ex.main
def run_exp():
    return exp_htcn_mixed()

if __name__ == "__main__":

    ex.run(config_updates={'cfg_file': 'cfgs/vgg16.yml',
                           'net': 'vgg16',
                           'model_type': 'htcn',
                           'val_datasets': ['cs_fg', 'cs_rain'],
                           'model_pth': "path_to_model"
                           },

           options={"--name": 'htcn_eval'})