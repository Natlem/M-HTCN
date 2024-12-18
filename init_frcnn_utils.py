import torch
from model.faster_rcnn.resnet import resnet as n_resnet
from model.faster_rcnn.resnet_saito import resnet as s_resnet
from model.faster_rcnn.resnet_HTCN import resnet
from model.faster_rcnn.resnet_HTCN_multi import resnet as m_resnet
from model.faster_rcnn.vgg16 import vgg16 as n_vgg16
from model.faster_rcnn.vgg16_HTCN import vgg16
from model.faster_rcnn.vgg16_saito import vgg16 as s_vgg16
from model.faster_rcnn.vgg16_HTCN_mrpn import vgg16 as vgg16_mrpn
from model.utils.config import cfg
from model.utils.net_utils import sampler
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb
import os

def init_dataloaders_1s_1t(args, batch_size, num_workers, is_bgr=True, is_train=True):
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    sampler_batch = sampler(train_size, batch_size)
    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                               imdb.num_classes, training=is_train, is_bgr=is_bgr)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=batch_size,
                                               sampler=sampler_batch, num_workers=num_workers)
    # target dataset
    dataloader_t = None
    imdb_t = None
    if args.dataset_t != "":
        imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
        train_size_t = len(roidb_t)
        sampler_batch_t = sampler(train_size_t, batch_size)

        dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, batch_size, \
                                   imdb.num_classes, training=is_train, is_bgr=is_bgr)
        dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size,
                                                   sampler=sampler_batch_t, num_workers=num_workers)
    return dataloader_s, dataloader_t, imdb, imdb_t


def init_dataloaders_1s_mt(args, batch_size, num_workers, is_bgr=True, is_shuffle=True):
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    sampler_batch = sampler(train_size, batch_size)
    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                               imdb.num_classes, training=True, is_bgr=is_bgr)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=batch_size,
                                               sampler=sampler_batch, num_workers=num_workers)

    m_dataloader_t = []
    m_imdb_t = []
    # target dataset
    for i_n_t in args.imdb_name_target:
        imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(i_n_t)
        m_imdb_t.append(imdb_t)
        train_size_t = len(roidb_t)

        sampler_batch_t = sampler(train_size_t, batch_size)

        dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, batch_size, \
                                   imdb.num_classes, training=True, is_bgr=is_bgr)
        dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size,
                                                   sampler=sampler_batch_t, num_workers=num_workers)
        m_dataloader_t.append(dataloader_t)
    return dataloader_s, m_dataloader_t, imdb, m_imdb_t


def init_dataloaders_1s_mixed_mt(args, batch_size, num_workers):
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    sampler_batch = sampler(train_size, batch_size)
    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                               imdb.num_classes, training=True)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=batch_size,
                                               sampler=sampler_batch, num_workers=num_workers)

    m_datasets = []
    m_imdb_t = []
    total_train_size = 0
    # target dataset
    for i_n_t in args.imdb_name_target:
        imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(i_n_t)
        m_imdb_t.append(imdb_t)
        total_train_size += len(roidb_t)



        dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, batch_size, \
                                   imdb.num_classes, training=True)

        m_datasets.append(dataset_t)

    sampler_batch_t = sampler(total_train_size, batch_size)
    concat_dataset = torch.utils.data.ConcatDataset(m_datasets)
    dataloader_t = torch.utils.data.DataLoader(concat_dataset, batch_size=batch_size,
                                               sampler=sampler_batch_t, num_workers=num_workers)

    return dataloader_s, dataloader_t, imdb


def init_val_dataloaders_mt(args, batch_size, num_workers,is_bgr=True):
    m_dataloader_t = []
    m_imdb_t = []
    # target dataset
    for i_n_t in args.imdb_name_target:
        imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(i_n_t, False)
        m_imdb_t.append(imdb_t)

        dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, batch_size, \
                                   imdb_t.num_classes, training=False, normalize=True, is_bgr=is_bgr)
        dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        m_dataloader_t.append(dataloader_t)
    return m_dataloader_t, m_imdb_t


def init_val_dataloaders_1t(args, batch_size, num_workers):

    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target, False)

    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, batch_size, \
                               imdb_t.num_classes, training=False, normalize=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader_t, imdb_t


def init_model_only(device, net, backbone_fn, imdb, load_path, **kwargs):

    if net == 'vgg16':
        fasterRCNN = backbone_fn(imdb.classes, pretrained=True, **kwargs)
    elif net == 'res101':
        fasterRCNN = backbone_fn(imdb.classes, 101, pretrained=True, **kwargs)
    elif net == 'res50':
        fasterRCNN = backbone_fn(imdb.classes, 50, pretrained=True, **kwargs)
    elif net == 'res152':
        fasterRCNN = backbone_fn(imdb.classes, 152, pretrained=True, **kwargs)
    else:
        raise NotImplementedError("Not implemented for other architecture")
    fasterRCNN.create_architecture()
    fasterRCNN.to(device)
    if load_path != '' and os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        fasterRCNN.load_state_dict(checkpoint['model'], strict=True)
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
    elif load_path != '' and not os.path.exists(load_path):
        raise FileNotFoundError('Load path does not exists')

    print("loaded checkpoint %s" % (load_path))
    return fasterRCNN

def init_htcn_model_optimizer(alr, LA_ATT, MID_ATT, class_agnostic, device, gc, imdb, lc, load_name, net, optimizer, resume,
                              session, start_epoch, pretrained=True, target_num=1, with_aop=False, is_all_params=False):

    optimizer_wd = None
    if net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=pretrained, class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT, target_num=target_num)
    elif net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res18':
        fasterRCNN = resnet(imdb.classes, 18, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res34':
        fasterRCNN = resnet(imdb.classes, 34, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)

    else:
        raise NotImplementedError("Not implemented for other architecture")
    #fasterRCNN.model_path = './pretrained/resnet101_caffe_voc0712.pth'
    fasterRCNN.create_architecture()
    lr = cfg.TRAIN.LEARNING_RATE
    lr = alr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad or is_all_params:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if with_aop:
        params_wd = []
        for key, value in dict(fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params_wd += [{'params': [value], 'lr': 0.00005 * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params_wd += [{'params': [value], 'lr': 0.00005, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        if addition_paramters:
            params_wd += addition_paramters


    if optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
        if with_aop:
            optimizer_wd = torch.optim.Adam(params_wd)

    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        if with_aop:
            optimizer_wd = torch.optim.SGD(params_wd, momentum=cfg.TRAIN.MOMENTUM)


    fasterRCNN.to(device)
    if resume:
        checkpoint = torch.load(load_name)
        session = checkpoint['session']
        start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
    return fasterRCNN, lr, optimizer, session, start_epoch, optimizer_wd

def init_htcn_model(LA_ATT, MID_ATT, class_agnostic, device, gc, imdb, lc, load_name, net, pretrained=True, strict=True, target_num=1):

    if net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=pretrained, class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT, target_num=target_num)
    elif net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res18':
        fasterRCNN = resnet(imdb.classes, 18, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res34':
        fasterRCNN = resnet(imdb.classes, 34, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)

    else:
        raise NotImplementedError("Not implemented for other architecture")
    fasterRCNN.create_architecture()
    fasterRCNN.to(device)
    if load_name != "":
        checkpoint = torch.load(load_name)
        fasterRCNN.load_state_dict(checkpoint['model'], strict=strict)
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print('Loading pretrained weight from {}'.format(load_name))
    return fasterRCNN

def init_htcn_multi_adv_model(LA_ATT, MID_ATT, class_agnostic, device, gc, imdb, lc, load_name, net, pretrained=True, strict=True, target_num=1):

    if net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=pretrained, class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT, target_num=target_num)
    elif net == 'res101':
        fasterRCNN = m_resnet(imdb.classes, 101, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT, target_num=target_num)
    elif net == 'res50':
        fasterRCNN = m_resnet(imdb.classes, 50, pretrained=pretrained, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT, target_num=target_num)
    else:
        raise NotImplementedError("Not implemented for other architecture")
    fasterRCNN.create_architecture()
    fasterRCNN.to(device)
    if load_name != "":
        checkpoint = torch.load(load_name)
        fasterRCNN.load_state_dict(checkpoint['model'], strict=strict)
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print('Loading pretrained weight from {}'.format(load_name))
    return fasterRCNN

def init_optimizer(alr, fasterRCNN, optimizer, resume, load_name, session, start_epoch, is_all_params=False):
    lr = cfg.TRAIN.LEARNING_RATE
    lr = alr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad or is_all_params:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    if optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if resume:
        checkpoint = torch.load(load_name)
        session = checkpoint['session']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    return lr, optimizer, session, start_epoch


def init_htcn_model_optimizer_with_od(alr, LA_ATT, MID_ATT, class_agnostic, device, gc, imdb, lc, load_name, net, optimizer, resume,
                              session, start_epoch, teacher, distiller_fn, is_all_params=False):

    optimizer_wd = None
    if net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT, target_num=1)
    elif net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    elif net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)

    else:
        raise NotImplementedError("Not implemented for other architecture")
    fasterRCNN.create_architecture()
    distill = distiller_fn(teacher, fasterRCNN)


    lr = cfg.TRAIN.LEARNING_RATE
    lr = alr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad or is_all_params:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    params += [{'params': distill.get_parameters(), 'lr': lr , \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY}]


    if optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)


    fasterRCNN.to(device)
    distill.to(device)
    if resume:
        checkpoint = torch.load(load_name)
        session = checkpoint['session']
        start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
    return fasterRCNN, lr, optimizer, session, start_epoch, distill


def init_saito_model_optimizer(alr, class_agnostic, device, gc, imdb, lc, load_name, net, optimizer, resume,
                         session, start_epoch, target_num=1, is_all_params=False):


    if net == 'vgg16':
        fasterRCNN = s_vgg16(imdb.classes, pretrained=True, class_agnostic=class_agnostic, lc=lc,
                           gc=gc)
    elif net == 'res101':
        fasterRCNN = s_resnet(imdb.classes, 101, pretrained=True, class_agnostic=class_agnostic,
                            lc=lc, gc=gc)
    elif net == 'res50':
        fasterRCNN = s_resnet(imdb.classes, 50, pretrained=True, class_agnostic=class_agnostic,
                            lc=lc, gc=gc)
    elif net == 'res152':
        fasterRCNN = s_resnet(imdb.classes, 152, pretrained=True, class_agnostic=class_agnostic,
                            lc=lc, gc=gc)

    else:
        raise NotImplementedError("Not implemented for other architecture")
    fasterRCNN.create_architecture()
    lr = cfg.TRAIN.LEARNING_RATE
    lr = alr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad or is_all_params:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    fasterRCNN.to(device)
    if resume:
        checkpoint = torch.load(load_name)
        session = checkpoint['session']
        start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
    return fasterRCNN, lr, optimizer, session, start_epoch, None

def init_non_damodel_optimizer(alr, class_agnostic, device, imdb, load_name, net, optimizer, resume,
                         session, start_epoch, is_all_params=False):

    optimizer_wd = None
    if net == 'vgg16':
        fasterRCNN = n_vgg16(imdb.classes, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res101':
        fasterRCNN = n_resnet(imdb.classes, 101, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res50':
        fasterRCNN = n_resnet(imdb.classes, 50, pretrained=True, class_agnostic=class_agnostic)
    else:
        raise NotImplementedError("Not implemented for other architecture")
    fasterRCNN.create_architecture()
    lr = cfg.TRAIN.LEARNING_RATE
    lr = alr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad or is_all_params:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)


    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    fasterRCNN.to(device)
    if resume:
        checkpoint = torch.load(load_name)
        session = checkpoint['session']
        start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
    return fasterRCNN, lr, optimizer, session, start_epoch, None


def init_mrpn_model_optimizer(alr, LA_ATT, MID_ATT, class_agnostic, device, gc, imdb, lc, load_name, net, optimizer, resume,
                         session, start_epoch, target_num, is_mtda):

    optimizer_wd = None
    if net == 'vgg16':
        fasterRCNN = vgg16_mrpn(imdb.classes, pretrained=True, class_agnostic=class_agnostic, lc=lc,
                           gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT, is_mtda=is_mtda, target_num=target_num)
    elif net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=class_agnostic,
                            lc=lc, gc=gc, la_attention=LA_ATT, mid_attention=MID_ATT)
    else:
        raise NotImplementedError("Not implemented for other architecture")
    fasterRCNN.create_architecture()
    lr = cfg.TRAIN.LEARNING_RATE
    lr = alr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)


    fasterRCNN.to(device)
    if resume:
        checkpoint = torch.load(load_name)
        session = checkpoint['session']
        start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
    return fasterRCNN, lr, optimizer, session, start_epoch, None