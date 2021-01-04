import time

import torch

from model.utils.config import cfg, get_output_dir
from model.utils.net_utils import clip_gradient
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from torch.nn import functional as F
from model.faster_rcnn.wd import gradient_penalty
from model.faster_rcnn import mmd as mmd
from model.faster_rcnn.faster_rcnn_HTCN import _fasterRCNN as frcnn_htcn
from model.faster_rcnn.faster_rcnn import _fasterRCNN as frcnn_no_da
from model.faster_rcnn.faster_rcnn_HTCN_mrpn import _fasterRCNN as frcnn_htcn_mprn

import numpy as np
import os


def train_htcn_one_epoch_mt_seq_binary_fast_pw_bary_no_grad_targets(args, FL, total_step,
                         dataloader_s, m_dataloader_t, iters_per_epoch,
                         fasterRCNN, optimizer, device, logger=None):
    count_step = 0
    loss_temp_last = 1
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    bary_temp = 0

    data_iter_s = iter(dataloader_s)
    dloss_dict = {}
    m_data_iters_t = []

    for i, dataloader_t in enumerate(m_dataloader_t):
        m_data_iters_t.append(iter(dataloader_t))
        dloss_dict[args.dataset_t[i]] = {}
        dloss_dict[args.dataset_t[i]]['dloss_t'] = 0
        dloss_dict[args.dataset_t[i]]['dloss_t_p'] = 0
        dloss_dict[args.dataset_t[i]]['dloss_t_mid'] = 0
        dloss_dict[args.dataset_t[i]]['dloss_t_ins'] = 0

    count_step = 0
    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)
        # eta = 1.0

        feat_list = []
        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins, _, _, s_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, with_feat=True)
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

        feat_list.append(s_feat.mean([2, 3]))

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

            out_d_pixel, out_d, out_d_mid, out_d_ins, _, _, bf_1t = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True, with_feat=True)
            feat_list.append(bf_1t.detach().mean([2, 3]).detach())

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
            dloss_dict[args.dataset_t[i]]['dloss_t'] += dloss_t
            dloss_dict[args.dataset_t[i]]['dloss_t_p'] += dloss_t_p * 0.15
            dloss_dict[args.dataset_t[i]]['dloss_t_mid'] += dloss_t_mid * 0.15
            dloss_dict[args.dataset_t[i]]['dloss_t_ins'] += dloss_t_ins * 0.15


        bary_loss = mmd.mmd_loss_bary(feat_list, 0)
        bary_temp += 0.005 * bary_loss.item()
        loss += 0.005 * bary_loss

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
            bary_temp /= count_step

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
                logger.log_scalar("bary_loss", bary_temp, total_step)
                for k, v in dloss_dict.items():
                    for loss_n, loss_v in v.items():
                        logger.log_scalar("{}_{}".format(k, loss_n), loss_v.item() / count_step, total_step)
                        v[loss_n] = 0

            count_step = 0
            loss_temp_last = loss_temp
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0
            bary_temp = 0
    return total_step


def train_htcn_one_epoch_mt_seq_binary_fast_pw_bary_no_grad_teachers(args, FL, total_step,
                                                                    dataloader_s, m_dataloader_t, iters_per_epoch,
                                                                    fasterRCNN, teachers, optimizer, device, logger=None):
    count_step = 0
    loss_temp_last = 1
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    bary_temp = 0

    data_iter_s = iter(dataloader_s)
    dloss_dict = {}
    m_data_iters_t = []

    for i, dataloader_t in enumerate(m_dataloader_t):
        m_data_iters_t.append(iter(dataloader_t))
        dloss_dict[args.dataset_t[i]] = {}
        dloss_dict[args.dataset_t[i]]['dloss_t'] = 0
        dloss_dict[args.dataset_t[i]]['dloss_t_p'] = 0
        dloss_dict[args.dataset_t[i]]['dloss_t_mid'] = 0
        dloss_dict[args.dataset_t[i]]['dloss_t_ins'] = 0

    count_step = 0
    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)
        # eta = 1.0

        s_feat_list = []
        t_feat_list = []
        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins, _, _, s_feat, s_s_mask_batch = fasterRCNN(im_data, im_info, gt_boxes,
                                                                                        num_boxes, with_feat=True, is_sup=True)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

        s_feat_list.append(s_feat.mean([2, 3]))

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

            im_data_t = data_t[0].to(device)
            im_info_t = data_t[1].to(device)
            gt_boxes_t = torch.zeros((1, 1, 5)).to(device)
            num_boxes_t = torch.zeros([1]).to(device)

            out_d_pixel, out_d, out_d_mid, out_d_ins, _, _, s_bf_1t, s_t_mask_batches = fasterRCNN(im_data_t, im_info_t, gt_boxes_t, num_boxes_t,
                                                                               target=True, with_feat=True, is_sup=True)

            with torch.no_grad():
                _, _, _, _, _, _, t_bf_1s, t_mask = teachers[i](im_data, im_info, gt_boxes, num_boxes,
                                                                                   target=True, with_feat=True, is_sup=True)


            s_feat_list.append(bf_1s.detach().mean([2, 3]).detach())
            t_feat_list.append(bf_1t.detach().mean([2, 3]).detach())



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
            dloss_dict[args.dataset_t[i]]['dloss_t'] += dloss_t
            dloss_dict[args.dataset_t[i]]['dloss_t_p'] += dloss_t_p * 0.15
            dloss_dict[args.dataset_t[i]]['dloss_t_mid'] += dloss_t_mid * 0.15
            dloss_dict[args.dataset_t[i]]['dloss_t_ins'] += dloss_t_ins * 0.15

        bary_loss = mmd.mmd_loss_bary(s_feat_list, 0)
        bary_temp += 0.005 * bary_loss.item()
        loss += 0.005 * bary_loss

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
            bary_temp /= count_step

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
                logger.log_scalar("bary_loss", bary_temp, total_step)

            count_step = 0
            loss_temp_last = loss_temp
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0
            bary_temp = 0
    return total_step