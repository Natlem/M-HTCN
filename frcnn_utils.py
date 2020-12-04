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
import pickle

def train_no_da_frcnn_one_epoch(args, total_step,
                         dataloader_s, iters_per_epoch,
                         fasterRCNN, optimizer, device, logger=None):

    loss_temp_last = 1
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0

    data_iter_s = iter(dataloader_s)
    count_step = 0
    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)

        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label= fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

        count_step += 1
        total_step += 1

        loss_temp += loss.item()
        loss_rpn_cls_temp += rpn_loss_cls.mean().item()
        loss_rpn_box_temp += rpn_loss_box.mean().item()
        loss_rcnn_cls_temp += RCNN_loss_cls.mean().item()
        loss_rcnn_box_temp += RCNN_loss_bbox.mean().item()


        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("s_name: {}".format(data_s[4]))
            print("s_gt_boxes:{}".format(data_s[2]))
            print("s_im_info:{}".format(data_s[1]))
            print("s_num_bixes:{}".format(data_s[3]))
            raise

        optimizer.zero_grad()
        loss.backward()
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

def train_htcn_one_epoch(args, FL, total_step,
                         dataloader_s, dataloader_t, iters_per_epoch,
                         fasterRCNN, optimizer, device, logger=None):
    count_step = 0
    loss_temp_last = 1
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0

    data_iter_s = iter(dataloader_s)
    data_iter_t = iter(dataloader_t)
    count_step = 0
    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)
        try:
            data_t = next(data_iter_t)
        except:
            data_iter_t = iter(dataloader_t)
            data_t = next(data_iter_t)
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

        if args.dataset == 'sim':
            loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p + dloss_s_mid * 0.15 + dloss_t_mid * 0.15 + dloss_s_ins * 0.5 + dloss_t_ins * 0.5) * args.eta
        else:
            loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p + dloss_s_mid * 0.15 + dloss_t_mid * 0.15 + dloss_s_ins * 0.5 + dloss_t_ins * 0.5)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("s_name: {}".format(data_s[4]))
            print("s_gt_boxes:{}".format(data_s[2]))
            print("s_im_info:{}".format(data_s[1]))
            print("s_num_bixes:{}".format(data_s[3]))

            print("t_name: {}".format(data_t[4]))
            print("t_gt_boxes:{}".format(data_t[2]))
            print("t_im_info:{}".format(data_t[1]))
            print("t_num_bixes:{}".format(data_t[3]))
            raise

        optimizer.zero_grad()
        loss.backward()
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

def train_htcn_one_epoch_mixed_mtda(args, FL, total_step,
                         dataloader_s, dataloader_t, iters_per_epoch,
                         fasterRCNN, optimizer, device, logger=None):
    count_step = 0
    loss_temp_last = 1
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0

    data_iter_s = iter(dataloader_s)
    data_iter_t = iter(dataloader_t)
    count_step = 0
    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)
        try:
            data_t = next(data_iter_t)
        except:
            data_iter_t = iter(dataloader_t)
            data_t = next(data_iter_t)
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

        im_data = data_t[0].to(device)
        im_info = data_t[1].to(device)

        gt_boxes = torch.zeros((1, 1, 5)).to(device)
        num_boxes = torch.zeros([1]).to(device)
        target_num = data_t[4].to(device)


        out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)
        out_d_ins_softmax = F.softmax(out_d_ins, 1)

        ######################### da loss 1 #####################################
        # domain label
        domain_t = torch.ones(out_d.size(0)).long().to(device) * target_num
        dloss_t = 0.5 * FL(out_d, domain_t)

        ######################### da loss 2 #####################################
        # domain label
        domain_t_mid = torch.ones(out_d_mid.size(0)).long().to(device) * target_num
        ##### mid alignment loss
        dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

        ######################### da loss 3 #####################################
        # local alignment loss
        dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

        ######################### da loss 4 #####################################
        # instance alignment loss
        domain_gt_ins = torch.ones(out_d_ins.size(0)).long().to(device) * target_num
        dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
        ##############################################################

        if args.dataset == 'sim':
            loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p + dloss_s_mid * 0.15 + dloss_t_mid * 0.15 + dloss_s_ins * 0.5 + dloss_t_ins * 0.5) * args.eta
        else:
            loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p + dloss_s_mid * 0.15 + dloss_t_mid * 0.15 + dloss_s_ins * 0.5 + dloss_t_ins * 0.5)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("s_name: {}".format(data_s[4]))
            print("s_gt_boxes:{}".format(data_s[2]))
            print("s_im_info:{}".format(data_s[1]))
            print("s_num_bixes:{}".format(data_s[3]))

            print("t_name: {}".format(data_t[4]))
            print("t_gt_boxes:{}".format(data_t[2]))
            print("t_im_info:{}".format(data_t[1]))
            print("t_num_bixes:{}".format(data_t[3]))
            raise

        optimizer.zero_grad()
        loss.backward()
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


def train_htcn_one_epoch_multi_targets_seq_mce_binary(args, FL, total_step,
                         dataloader_s, m_dataloader_t, iters_per_epoch,
                         fasterRCNN, optimizer, device, is_mtda=False, logger=None):
    count_step = 0
    loss_temp_last = 1
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0

    dloss_dict = {}

    da_between_targets_temp = 0
    bary_temp = 0

    data_iter_s = iter(dataloader_s)
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
        feat_list = []

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

        dloss_t_t = 0
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
            feat_list.append(bf_1t.detach().mean([2, 3]))
            out_d_ins_softmax = F.softmax(out_d_ins, 1)

            ######################### da loss 1 #####################################
            # domain label
            domain_t = torch.ones(out_d.size(0)).long().to(device)
            if is_mtda:
                domain_t *= (i + 1)
            dloss_t = 0.5 * FL(out_d, domain_t)
            ######################### da loss between target only #####################################
            if is_mtda:
                domain_to = torch.ones(out_d.size(0)).long().to(device) * (i)
                dloss_to = 0.5 * FL(out_d, domain_to)

            ######################### da loss 2 #####################################
            # domain label
            domain_t_mid = torch.ones(out_d_mid.size(0)).long().to(device)
            if is_mtda:
                domain_t_mid *= (i + 1)
            ##### mid alignment loss
            dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

            ######################### da loss 4 #####################################
            # instance alignment loss
            domain_gt_ins = torch.ones(out_d_ins.size(0)).long().to(device)
            if is_mtda:
                domain_gt_ins *= (i + 1)
            dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##############################################################

            ######################### mtda loss 4 #####################################
            dloss_t_t += dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5

            dloss_dict[args.dataset_t[i]]['dloss_t'] += dloss_t
            dloss_dict[args.dataset_t[i]]['dloss_t_p'] += dloss_t_p * 0.15
            dloss_dict[args.dataset_t[i]]['dloss_t_mid'] += dloss_t_mid * 0.15
            dloss_dict[args.dataset_t[i]]['dloss_t_ins'] += dloss_t_ins * 0.15

        if args.dataset == 'sim':
            loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p + dloss_s_mid * 0.15 + dloss_t_mid * 0.15 + dloss_s_ins * 0.5 + dloss_t_ins * 0.5) * args.eta
        else:
            loss += (dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5) + dloss_t_t



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
                dloss_s_p = dloss_s_p.item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt


            if logger:
                logger.log_scalar("loss", loss_temp, total_step)
                logger.log_scalar("loss_rpn_cls", loss_rpn_cls_temp, total_step)
                logger.log_scalar("loss_rpn_box", loss_rpn_box_temp, total_step)
                logger.log_scalar("loss_rcnn_cls", loss_rcnn_cls_temp, total_step)
                logger.log_scalar("loss_rcnn_box", loss_rcnn_box_temp, total_step)
                for k, v in dloss_dict.items():
                    for loss_n, loss_v in v.items():
                        logger.log_scalar("{}_{}".format(k, loss_n), loss_v.item() / count_step, total_step)
                        v[loss_n] = 0
                logger.log_scalar("bary_loss", bary_temp, total_step)

            count_step = 0
            loss_temp_last = loss_temp
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0
    return total_step




def eval_one_dataloader(save_dir_test_out, dataloader_t, fasterRCNN, device, imdb,
                        class_agnostic=False, thresh=0.0, max_per_image=100):

    save_name = save_dir_test_out + '_test_in_'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    data_iter = iter(dataloader_t)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):

        data = next(data_iter)

        im_data = data[0].to(device)
        im_info = data[1].to(device)
        gt_boxes = data[2].to(device)
        num_boxes = data[3].to(device)

        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, d_pred, _, _, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        # d_pred = d_pred.data
        # path = data[4]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()  # [1, 300, 2] -> [300, 2]
        pred_boxes = pred_boxes.squeeze()  # [1, 300, 8] -> [300, 8]
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh, as_tuple=False).view(-1)  # [300]
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]  # [300]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]  # [300, 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)  # [300, 5]
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                # keep = nms(cls_dets, cfg.TEST.NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)  # [N, 1]
                cls_dets = cls_dets[keep.view(-1).long()]  # [N, 5]

                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])  # [M,]
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #                  .format(i + 1, num_images, detect_time, nms_time))
        # sys.stdout.flush()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    map = imdb.evaluate_detections(all_boxes, output_dir)
    return map

def eval_one_dataloader(save_dir_test_out, dataloader_t, fasterRCNN, device, imdb, target_num=0,
                        class_agnostic=False, thresh=0.0, max_per_image=100):

    save_name = save_dir_test_out + '_test_in_'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    data_iter = iter(dataloader_t)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):

        data = next(data_iter)

        im_data = data[0].to(device)
        im_info = data[1].to(device)
        gt_boxes = data[2].to(device)
        num_boxes = data[3].to(device)

        if isinstance(fasterRCNN, frcnn_htcn) or isinstance(fasterRCNN, frcnn_htcn_mprn):
            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, d_pred, _, _, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target_num=target_num)
        else:
            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        # d_pred = d_pred.data
        # path = data[4]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()  # [1, 300, 2] -> [300, 2]
        pred_boxes = pred_boxes.squeeze()  # [1, 300, 8] -> [300, 8]
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh, as_tuple=False).view(-1)  # [300]
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]  # [300]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]  # [300, 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)  # [300, 5]
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                # keep = nms(cls_dets, cfg.TEST.NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)  # [N, 1]
                cls_dets = cls_dets[keep.view(-1).long()]  # [N, 5]

                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])  # [M,]
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #                  .format(i + 1, num_images, detect_time, nms_time))
        # sys.stdout.flush()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    map = imdb.evaluate_detections(all_boxes, output_dir)
    return map