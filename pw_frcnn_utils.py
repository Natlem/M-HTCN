import time

import torch

from model.utils.config import cfg, get_output_dir
from model.utils.net_utils import clip_gradient
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from torch.nn import functional as F
from model.faster_rcnn import mmd as mmd
from model.faster_rcnn.faster_rcnn_HTCN import _fasterRCNN as frcnn_htcn
from model.faster_rcnn.faster_rcnn import _fasterRCNN as frcnn_no_da
from model.faster_rcnn.faster_rcnn_HTCN_mrpn import _fasterRCNN as frcnn_htcn_mprn

import numpy as np
import os

def train_htcn_one_epoch_mt_seq_binary(args, FL, total_step,
                                       dataloader_s, m_dataloader_t, iters_per_epoch,
                                       fasterRCNN, optimizer, device, logger=None):
    count_step = 0
    loss_temp_last = 1
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0

    data_iter_s = iter(dataloader_s)

    m_data_iters_t = []
    for dataloader_t in m_dataloader_t:
        m_data_iters_t.append(iter(dataloader_t))

    count_step = 0
    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)
        # eta = 1.0

        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

        count_step += 1
        total_step += 1

        loss_temp += loss.item()
        loss_rpn_cls_temp += rpn_loss_cls.mean().item()
        loss_rpn_box_temp += rpn_loss_box.mean().item()
        loss_rcnn_cls_temp += RCNN_loss_cls.mean().item()
        loss_rcnn_box_temp += RCNN_loss_bbox.mean().item()

        ######################### da loss 1 #####################################
        # domain label
        domain_s = torch.zeros(out_d.size(0)).long().to(device)
        # global alignment loss
        dloss_s = 0.5 * FL(out_d, domain_s)

        ######################### da loss 2 #####################################
        # domain label
        domain_s_mid = torch.zeros(out_d_mid.size(0)).long().to(device)
        ##### mid alignment loss
        dloss_s_mid = 0.5 * F.cross_entropy(out_d_mid, domain_s_mid)

        ######################### da loss 3 #####################################
        # local alignment loss
        dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

        ######################### da loss 4 #####################################
        # instance alignment loss
        domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device)
        dloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
        ##############################################################

        for i, data_iter_t in enumerate(m_data_iters_t):
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(m_dataloader_t[i])
                data_t = next(data_iter_t)

            im_data = data_t[0].to(device)
            im_info = data_t[1].to(device)

            gt_boxes = torch.zeros((1, 1, 5)).to(device)
            num_boxes = torch.zeros([1]).to(device)

            out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)
            out_d_ins_softmax = F.softmax(out_d_ins, 1)

            ######################### da loss 1 #####################################
            # domain label
            domain_t = torch.ones(out_d.size(0)).long().to(device)
            dloss_t = 0.5 * FL(out_d, domain_t)

            ######################### da loss 2 #####################################
            # domain label
            domain_t_mid = torch.ones(out_d_mid.size(0)).long().to(device)
            ##### mid alignment loss
            dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

            ######################### da loss 4 #####################################
            # instance alignment loss
            domain_gt_ins = torch.ones(out_d_ins.size(0)).long().to(device)
            dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##############################################################

            ######################### mtda loss 4 #####################################

            dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
            loss += dloss_t_t
        loss += (dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5)

        optimizer.zero_grad()
        loss.backward()
        if args.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()

        if step % args.disp_interval == 0:
            end = time.time()

            loss_temp /= count_step
            loss_rpn_cls_temp /= count_step
            loss_rpn_box_temp /= count_step
            loss_rcnn_cls_temp /= count_step
            loss_rcnn_box_temp /= count_step

            if torch.cuda.device_count() > 2:
                loss_rpn_cls = rpn_loss_cls.mean().item()
                loss_rpn_box = rpn_loss_box.mean().item()
                loss_rcnn_cls = RCNN_loss_cls.mean().item()
                loss_rcnn_box = RCNN_loss_bbox.mean().item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt
            else:
                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()
                dloss_s = dloss_s.item()
                dloss_t = dloss_t.item()
                dloss_s_p = dloss_s_p.item()
                dloss_t_p = dloss_t_p.item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

            if logger:
                logger.log_scalar("loss", loss_temp, total_step)
                logger.log_scalar("loss_rpn_cls", loss_rpn_cls_temp, total_step)
                logger.log_scalar("loss_rpn_box", loss_rpn_box_temp, total_step)
                logger.log_scalar("loss_rcnn_cls", loss_rcnn_cls_temp, total_step)
                logger.log_scalar("loss_rcnn_box", loss_rcnn_box_temp, total_step)

            count_step = 0
            loss_temp_last = loss_temp
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0
    return total_step

def train_htcn_one_epoch_mt_seq_binary_fast_pw(args, FL, total_step,
                         dataloader_s, m_dataloader_t, iters_per_epoch,
                         fasterRCNN, optimizer, device, logger=None):
    count_step = 0
    loss_temp_last = 1
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0

    data_iter_s = iter(dataloader_s)

    m_data_iters_t = []
    for dataloader_t in m_dataloader_t:
        m_data_iters_t.append(iter(dataloader_t))

    count_step = 0
    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)
        # eta = 1.0

        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()



        count_step += 1
        total_step += 1

        loss_temp += loss.item()
        loss_rpn_cls_temp += rpn_loss_cls.mean().item()
        loss_rpn_box_temp += rpn_loss_box.mean().item()
        loss_rcnn_cls_temp += RCNN_loss_cls.mean().item()
        loss_rcnn_box_temp += RCNN_loss_bbox.mean().item()

        ######################### da loss 1 #####################################
        # domain label
        domain_s = torch.zeros(out_d.size(0)).long().to(device)
        # global alignment loss
        dloss_s = 0.5 * FL(out_d, domain_s)

        ######################### da loss 2 #####################################
        # domain label
        domain_s_mid = torch.zeros(out_d_mid.size(0)).long().to(device)
        ##### mid alignment loss
        dloss_s_mid = 0.5 * F.cross_entropy(out_d_mid, domain_s_mid)

        ######################### da loss 3 #####################################
        # local alignment loss
        dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

        ######################### da loss 4 #####################################
        # instance alignment loss
        domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device)
        dloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
        ##############################################################

        for i, data_iter_t in enumerate(m_data_iters_t):
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(m_dataloader_t[i])
                data_t = next(data_iter_t)

            im_data = data_t[0].to(device)
            im_info = data_t[1].to(device)

            gt_boxes = torch.zeros((1, 1, 5)).to(device)
            num_boxes = torch.zeros([1]).to(device)

            out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)
            out_d_ins_softmax = F.softmax(out_d_ins, 1)

            ######################### da loss 1 #####################################
            # domain label
            domain_t = torch.ones(out_d.size(0)).long().to(device)
            dloss_t = 0.5 * FL(out_d, domain_t)

            ######################### da loss 2 #####################################
            # domain label
            domain_t_mid = torch.ones(out_d_mid.size(0)).long().to(device)
            ##### mid alignment loss
            dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

            ######################### da loss 4 #####################################
            # instance alignment loss
            domain_gt_ins = torch.ones(out_d_ins.size(0)).long().to(device)
            dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##############################################################

            ######################### mtda loss 4 #####################################

            dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
            loss += (dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5) + dloss_t_t

        optimizer.zero_grad()
        loss.backward()
        if args.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()

        if step % args.disp_interval == 0:
            end = time.time()

            loss_temp /= count_step
            loss_rpn_cls_temp /= count_step
            loss_rpn_box_temp /= count_step
            loss_rcnn_cls_temp /= count_step
            loss_rcnn_box_temp /= count_step

            if torch.cuda.device_count() > 2:
                loss_rpn_cls = rpn_loss_cls.mean().item()
                loss_rpn_box = rpn_loss_box.mean().item()
                loss_rcnn_cls = RCNN_loss_cls.mean().item()
                loss_rcnn_box = RCNN_loss_bbox.mean().item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt
            else:
                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()
                dloss_s = dloss_s.item()
                dloss_t = dloss_t.item()
                dloss_s_p = dloss_s_p.item()
                dloss_t_p = dloss_t_p.item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

            if logger:
                logger.log_scalar("loss", loss_temp, total_step)
                logger.log_scalar("loss_rpn_cls", loss_rpn_cls_temp, total_step)
                logger.log_scalar("loss_rpn_box", loss_rpn_box_temp, total_step)
                logger.log_scalar("loss_rcnn_cls", loss_rcnn_cls_temp, total_step)
                logger.log_scalar("loss_rcnn_box", loss_rcnn_box_temp, total_step)

            count_step = 0
            loss_temp_last = loss_temp
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0
    return total_step

def train_htcn_one_epoch_mt_seq_binary_pw(args, FL, total_step,
                                          dataloader_s, m_dataloader_t, iters_per_epoch,
                                          fasterRCNN, optimizer, device, logger=None):
    count_step = 0
    loss_temp_last = 1

    data_iter_s = iter(dataloader_s)

    m_data_iters_t = []
    dloss_dict = {}

    for i, dataloader_t in enumerate(m_dataloader_t):
        m_data_iters_t.append(iter(dataloader_t))
        dloss_dict[args.dataset_t[i]] = {}
        dloss_dict[args.dataset_t[i]]['dloss_t'] = 0
        dloss_dict[args.dataset_t[i]]['dloss_t_p'] = 0
        dloss_dict[args.dataset_t[i]]['dloss_t_mid'] = 0
        dloss_dict[args.dataset_t[i]]['dloss_t_ins'] = 0
        dloss_dict[args.dataset_t[i]]['loss_temp'] = 0
        dloss_dict[args.dataset_t[i]]['loss_rpn_cls_temp'] = 0
        dloss_dict[args.dataset_t[i]]['loss_rpn_box_temp'] = 0
        dloss_dict[args.dataset_t[i]]['loss_rcnn_cls_temp'] = 0
        dloss_dict[args.dataset_t[i]]['loss_rcnn_box_temp'] = 0

    count_step = 0
    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)
        # eta = 1.0
        count_step += 1
        total_step += 1

        for i, data_iter_t in enumerate(m_data_iters_t):
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(m_dataloader_t[i])
                data_t = next(data_iter_t)

            im_data = data_s[0].to(device)
            im_info = data_s[1].to(device)
            gt_boxes = data_s[2].to(device)
            num_boxes = data_s[3].to(device)

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()


            dloss_dict[args.dataset_t[i]]['loss_temp'] += loss.item()
            dloss_dict[args.dataset_t[i]]['loss_rpn_cls_temp'] += rpn_loss_cls.mean().item()
            dloss_dict[args.dataset_t[i]]['loss_rpn_box_temp'] += rpn_loss_box.mean().item()
            dloss_dict[args.dataset_t[i]]['loss_rcnn_cls_temp'] += RCNN_loss_cls.mean().item()
            dloss_dict[args.dataset_t[i]]['loss_rcnn_box_temp'] += RCNN_loss_bbox.mean().item()


            ######################### da loss 1 #####################################
            # domain label
            domain_s = torch.zeros(out_d.size(0)).long().to(device)
            # global alignment loss
            dloss_s = 0.5 * FL(out_d, domain_s)

            ######################### da loss 2 #####################################
            # domain label
            domain_s_mid = torch.zeros(out_d_mid.size(0)).long().to(device)
            ##### mid alignment loss
            dloss_s_mid = 0.5 * F.cross_entropy(out_d_mid, domain_s_mid)

            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

            ######################### da loss 4 #####################################
            # instance alignment loss
            domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device)
            dloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##############################################################

            im_data = data_t[0].to(device)
            im_info = data_t[1].to(device)

            gt_boxes = torch.zeros((1, 1, 5)).to(device)
            num_boxes = torch.zeros([1]).to(device)

            try:
                out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)
                out_d_ins_softmax = F.softmax(out_d_ins, 1)
            except ValueError as ve:
                print(str(ve))
                print(data_t[4])

            ######################### da loss 1 #####################################
            # domain label
            domain_t = torch.ones(out_d.size(0)).long().to(device)
            dloss_t = 0.5 * FL(out_d, domain_t)

            ######################### da loss 2 #####################################
            # domain label
            domain_t_mid = torch.ones(out_d_mid.size(0)).long().to(device)
            ##### mid alignment loss
            dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

            ######################### da loss 4 #####################################
            # instance alignment loss
            domain_gt_ins = torch.ones(out_d_ins.size(0)).long().to(device)
            dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##############################################################

            ######################### mtda loss 4 #####################################

            dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
            dloss_dict[args.dataset_t[i]]['dloss_t'] += dloss_t.item()
            dloss_dict[args.dataset_t[i]]['dloss_t_p'] += dloss_t_p.item() * 0.15
            dloss_dict[args.dataset_t[i]]['dloss_t_mid'] += dloss_t_mid.item() * 0.15
            dloss_dict[args.dataset_t[i]]['dloss_t_ins'] += dloss_t_ins.item() * 0.15

            loss += dloss_t_t
            loss += (dloss_s + dloss_s_p  + dloss_s_mid * 0.15 + dloss_s_ins * 0.5)

            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

        if step % args.disp_interval == 0:
            end = time.time()

            if logger:
                for k, v in dloss_dict.items():
                    for loss_n, loss_v in v.items():
                        logger.log_scalar("{}_{}".format(k, loss_n), loss_v / count_step, total_step)
                        v[loss_n] = 0

            count_step = 0
    return total_step
