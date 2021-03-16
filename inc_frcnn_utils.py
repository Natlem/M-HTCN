import time

import torch

from torch.nn import functional as F


def train_htcn_one_epoch_inc_union(args, FL, total_step,
                         dataloader_s, m_dataloader_t, iters_per_epoch,
                         fasterRCNN, optimizer, device, eta=1, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    dloss_s_s_temp = 0.

    data_iter_s = iter(dataloader_s)
    m_data_iters_t = []
    dloss_dict = {}
    with_source = []

    for i, dataloader_t in enumerate(m_dataloader_t):
        with_source.append(1)
        m_data_iters_t.append(iter(dataloader_t))
        dloss_dict[args.dataset_t[i]] = {}
        dloss_dict[args.dataset_t[i]]['dloss_t_t'] = 0

    with_source[-1] = 1
    count_step = 0

    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)

        count_step += 1
        total_step += 1

        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins, _ = fasterRCNN(im_data, im_info, gt_boxes,
                                                                                      num_boxes, with_feat=True)
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

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
        dloss_s_s = dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5
        dloss_s_s_temp += dloss_s_s.item()
        loss += dloss_s_s * eta


        teachers_mt_feats = []
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

            out_d_pixel, out_d, out_d_mid, out_d_ins, stu_t_feats = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                                                               target=True, with_feat=True)
            out_d_ins_softmax = F.softmax(out_d_ins, 1)

            ######################### da loss 1 #####################################
            # domain label
            domain_t = torch.zeros(out_d.size(0)).long().to(device) + with_source[i]
            dloss_t = 0.5 * FL(out_d, domain_t)

            ######################### da loss 2 #####################################
            # domain label
            domain_t_mid = torch.zeros(out_d_mid.size(0)).long().to(device) + with_source[i]
            ##### mid alignment loss
            dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

            ######################### da loss 4 #####################################
            # instance alignment loss
            domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device) + with_source[i]
            dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##############################################################
            dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
            dloss_dict[args.dataset_t[i]]['dloss_t_t'] += dloss_t_t.item()

            loss += dloss_t_t * eta


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % args.disp_interval == 0:
            end = time.time()

            loss_temp /= count_step
            loss_rpn_cls_temp /= count_step
            loss_rpn_box_temp /= count_step
            loss_rcnn_cls_temp /= count_step
            loss_rcnn_box_temp /= count_step

            if logger:
                logger.log_scalar("loss", loss_temp, total_step)
                logger.log_scalar("loss_rpn_cls", loss_rpn_cls_temp, total_step)
                logger.log_scalar("loss_rpn_box", loss_rpn_box_temp, total_step)
                logger.log_scalar("loss_rcnn_cls", loss_rcnn_cls_temp, total_step)
                logger.log_scalar("loss_rcnn_box", loss_rcnn_box_temp, total_step)
                logger.log_scalar("dloss_s_s", dloss_s_s_temp / count_step, total_step)
                for dataset_name, v in dloss_dict.items():
                    for loss_name, value in v.items():
                        logger.log_scalar("{}:{}".format(dataset_name, loss_name), value / count_step, total_step)
                        v[loss_name] = 0

            count_step = 0
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0
            dloss_s_s_temp = 0


    return total_step

def train_htcn_one_epoch_inc_union_w_old_cst(args, FL, total_step,
                         dataloader_s, m_dataloader_t, iters_per_epoch,
                         fasterRCNN, optimizer, device, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    new_dloss_s_s_temp = 0.
    old_dloss_s_s_temp = 0.

    data_iter_s = iter(dataloader_s)
    m_data_iters_t = []
    dloss_dict = {}
    with_source = []

    for i, dataloader_t in enumerate(m_dataloader_t):
        with_source.append(0)
        m_data_iters_t.append(iter(dataloader_t))
        dloss_dict[args.dataset_t[i]] = {}
        dloss_dict[args.dataset_t[i]]['new_dloss_t_t'] = 0
        if i < len(m_dataloader_t) - 1:
            dloss_dict[args.dataset_t[i]]['old_dloss_t_t'] = 0

    with_source[-1] = 1
    count_step = 0


    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)

        count_step += 1
        total_step += 1

        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins, all_feats = fasterRCNN(im_data, im_info, gt_boxes,
                                                                                      num_boxes, with_feat=True)
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

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
        old_dloss_s_s = dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5
        old_dloss_s_s_temp += old_dloss_s_s.item()
        loss += 0.1 * old_dloss_s_s

        base_feat1, base_feat2, base_feat, pooled_feat, _ = all_feats
        out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN.adv_forward(base_feat1, base_feat2, base_feat, pooled_feat, -1)
        domain_s = torch.zeros(out_d.size(0)).long().to(device)
        dloss_s = 0.5 * FL(out_d, domain_s)
        domain_s_mid = torch.zeros(out_d_mid.size(0)).long().to(device)
        dloss_s_mid = 0.5 * F.cross_entropy(out_d_mid, domain_s_mid)
        dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)
        domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device)
        dloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
        dloss_s_s = dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5
        new_dloss_s_s_temp += dloss_s_s.item()
        loss += dloss_s_s

        teachers_mt_feats = []
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

            out_d_pixel, out_d, out_d_mid, out_d_ins, all_feats = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                                                               target=True, with_feat=True)

            if i < len(m_data_iters_t) - 1:
                ######################### da loss 1 #####################################
                # domain label
                domain_t = torch.zeros(out_d.size(0)).long().to(device) + with_source[i]
                dloss_t = 0.5 * FL(out_d, domain_t)

                ######################### da loss 2 #####################################
                # domain label
                domain_t_mid = torch.zeros(out_d_mid.size(0)).long().to(device) + with_source[i]
                dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

                ######################### da loss 3 #####################################
                # local alignment loss
                dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

                ######################### da loss 4 #####################################
                # instance alignment loss
                domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device) + with_source[i]
                dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
                ##############################################################
                old_dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
                dloss_dict[args.dataset_t[i]]['old_dloss_t_t'] += old_dloss_t_t.item()
                loss += 0.1 * old_dloss_t_t


            base_feat1, base_feat2, base_feat, pooled_feat, _ = all_feats
            out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN.adv_forward(base_feat1, base_feat2, base_feat,
                                                                              pooled_feat, -1)
            domain_t = torch.zeros(out_d.size(0)).long().to(device) + with_source[i]
            dloss_t = 0.5 * FL(out_d, domain_t)
            domain_t_mid = torch.zeros(out_d_mid.size(0)).long().to(device) + with_source[i]
            dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)
            domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device)
            domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device) + with_source[i]
            dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
            dloss_dict[args.dataset_t[i]]['new_dloss_t_t'] += dloss_t_t.item()
            loss += dloss_t_t


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % args.disp_interval == 0:
            end = time.time()

            loss_temp /= count_step
            loss_rpn_cls_temp /= count_step
            loss_rpn_box_temp /= count_step
            loss_rcnn_cls_temp /= count_step
            loss_rcnn_box_temp /= count_step

            if logger:
                logger.log_scalar("loss", loss_temp, total_step)
                logger.log_scalar("loss_rpn_cls", loss_rpn_cls_temp, total_step)
                logger.log_scalar("loss_rpn_box", loss_rpn_box_temp, total_step)
                logger.log_scalar("loss_rcnn_cls", loss_rcnn_cls_temp, total_step)
                logger.log_scalar("loss_rcnn_box", loss_rcnn_box_temp, total_step)
                logger.log_scalar("old_dloss_s_s", old_dloss_s_s_temp / count_step, total_step)
                logger.log_scalar("new_dloss_s_s", new_dloss_s_s_temp / count_step, total_step)
                for dataset_name, v in dloss_dict.items():
                    for loss_name, value in v.items():
                        logger.log_scalar("{}:{}".format(dataset_name, loss_name), value / count_step, total_step)
                        v[loss_name] = 0

            count_step = 0
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0
            new_dloss_s_s_temp = 0
            old_dloss_s_s_temp = 0


    return total_step


def train_htcn_one_epoch_inc_union_old_momentum(args, FL, total_step,
                         dataloader_s, m_dataloader_t, iters_per_epoch,
                         fasterRCNN, optimizer, device, adv_num, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    new_dloss_s_s_temp = 0.
    old_dloss_s_s_temp = 0.

    data_iter_s = iter(dataloader_s)
    m_data_iters_t = []
    dloss_dict = {}
    with_source = []

    for i, dataloader_t in enumerate(m_dataloader_t):
        with_source.append(0)
        m_data_iters_t.append(iter(dataloader_t))
        dloss_dict[args.dataset_t[i]] = {}
        dloss_dict[args.dataset_t[i]]['new_dloss_t_t'] = 0
        if i < len(m_dataloader_t) - 1:
            dloss_dict[args.dataset_t[i]]['old_dloss_t_t'] = 0

    with_source[-1] = 1
    count_step = 0

    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)

        count_step += 1
        total_step += 1

        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins, all_feats = fasterRCNN.adv_sel_forward(im_data, im_info, gt_boxes,
                                                                                     num_boxes, adv_num=adv_num, with_feat=True)
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

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
        old_dloss_s_s = dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5
        old_dloss_s_s_temp += old_dloss_s_s.item()
        loss += 0.1 * old_dloss_s_s

        base_feat1, base_feat2, base_feat, pooled_feat, _ = all_feats
        out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN.adv_forward(base_feat1, base_feat2, base_feat,
                                                                          pooled_feat, -1)
        domain_s = torch.zeros(out_d.size(0)).long().to(device)
        dloss_s = 0.5 * FL(out_d, domain_s)
        domain_s_mid = torch.zeros(out_d_mid.size(0)).long().to(device)
        dloss_s_mid = 0.5 * F.cross_entropy(out_d_mid, domain_s_mid)
        dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)
        domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device)
        dloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
        dloss_s_s = dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5
        new_dloss_s_s_temp += dloss_s_s.item()
        loss += dloss_s_s

        teachers_mt_feats = []
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

            out_d_pixel, out_d, out_d_mid, out_d_ins, all_feats = fasterRCNN.adv_sel_forward(im_data, im_info, gt_boxes, num_boxes,
                                                                             target=True, adv_num=adv_num, with_feat=True)

            if i < len(m_data_iters_t) - 1:
                ######################### da loss 1 #####################################
                # domain label
                domain_t = torch.zeros(out_d.size(0)).long().to(device) + with_source[i]
                dloss_t = 0.5 * FL(out_d, domain_t)

                ######################### da loss 2 #####################################
                # domain label
                domain_t_mid = torch.zeros(out_d_mid.size(0)).long().to(device) + with_source[i]
                dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

                ######################### da loss 3 #####################################
                # local alignment loss
                dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

                ######################### da loss 4 #####################################
                # instance alignment loss
                domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device) + with_source[i]
                dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
                ##############################################################
                old_dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
                dloss_dict[args.dataset_t[i]]['old_dloss_t_t'] += old_dloss_t_t.item()
                loss += 0.1 * old_dloss_t_t

            base_feat1, base_feat2, base_feat, pooled_feat, _ = all_feats
            out_d_pixel, out_d, out_d_mid, out_d_ins = fasterRCNN.adv_forward(base_feat1, base_feat2, base_feat,
                                                                              pooled_feat, -1)
            domain_t = torch.zeros(out_d.size(0)).long().to(device) + with_source[i]
            dloss_t = 0.5 * FL(out_d, domain_t)
            domain_t_mid = torch.zeros(out_d_mid.size(0)).long().to(device) + with_source[i]
            dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)
            domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device)
            domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device) + with_source[i]
            dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
            dloss_dict[args.dataset_t[i]]['new_dloss_t_t'] += dloss_t_t.item()
            loss += dloss_t_t

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % args.disp_interval == 0:
            end = time.time()

            loss_temp /= count_step
            loss_rpn_cls_temp /= count_step
            loss_rpn_box_temp /= count_step
            loss_rcnn_cls_temp /= count_step
            loss_rcnn_box_temp /= count_step

            if logger:
                logger.log_scalar("loss", loss_temp, total_step)
                logger.log_scalar("loss_rpn_cls", loss_rpn_cls_temp, total_step)
                logger.log_scalar("loss_rpn_box", loss_rpn_box_temp, total_step)
                logger.log_scalar("loss_rcnn_cls", loss_rcnn_cls_temp, total_step)
                logger.log_scalar("loss_rcnn_box", loss_rcnn_box_temp, total_step)
                logger.log_scalar("old_dloss_s_s", old_dloss_s_s_temp / count_step, total_step)
                logger.log_scalar("new_dloss_s_s", new_dloss_s_s_temp / count_step, total_step)
                for dataset_name, v in dloss_dict.items():
                    for loss_name, value in v.items():
                        logger.log_scalar("{}:{}".format(dataset_name, loss_name), value / count_step, total_step)
                        v[loss_name] = 0

            count_step = 0
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0
            new_dloss_s_s_temp = 0
            old_dloss_s_s_temp = 0
    return total_step

def train_htcn_one_epoch_inc_adv_num(args, FL, total_step,
                         dataloader_s, dataloader_t, iters_per_epoch,
                         fasterRCNN, optimizer, device, adv_num, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    dloss_s_s_temp = 0.
    dloss_t_t_temp = 0.

    data_iter_s = iter(dataloader_s)
    data_iter_t = iter(dataloader_t)
    dloss_dict = {}
    with_source = []


    count_step = 0

    for step in range(1, iters_per_epoch + 1):
        try:
            data_s = next(data_iter_s)
        except:
            data_iter_s = iter(dataloader_s)
            data_s = next(data_iter_s)

        count_step += 1
        total_step += 1

        im_data = data_s[0].to(device)
        im_info = data_s[1].to(device)
        gt_boxes = data_s[2].to(device)
        num_boxes = data_s[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins, all_feats = fasterRCNN.adv_sel_forward(im_data, im_info, gt_boxes,
                                                                                     num_boxes, adv_num=adv_num, with_feat=True)
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

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
        dloss_s_s = dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5
        dloss_s_s_temp += dloss_s_s.item()
        loss += dloss_s_s

        try:
            data_t = next(data_iter_t)
        except:
            data_iter_t = iter(dataloader_t)
            data_t = next(data_iter_t)
        im_data = data_t[0].to(device)
        im_info = data_t[1].to(device)

        gt_boxes = torch.zeros((1, 1, 5)).to(device)
        num_boxes = torch.zeros([1]).to(device)

        out_d_pixel, out_d, out_d_mid, out_d_ins, all_feats = fasterRCNN.adv_sel_forward(im_data, im_info, gt_boxes, num_boxes,
                                                                         target=True, adv_num=adv_num, with_feat=True)

        ######################### da loss 1 #####################################
        # domain label
        domain_t = torch.ones(out_d.size(0)).long().to(device)
        dloss_t = 0.5 * FL(out_d, domain_t)

        ######################### da loss 2 #####################################
        # domain label
        domain_t_mid = torch.ones(out_d_mid.size(0)).long().to(device)
        dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

        ######################### da loss 3 #####################################
        # local alignment loss
        dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

        ######################### da loss 4 #####################################
        # instance alignment loss
        domain_gt_ins = torch.ones(out_d_ins.size(0)).long().to(device)
        dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
        ##############################################################
        dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
        dloss_t_t_temp += dloss_t_t.item()
        loss += dloss_t_t

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % args.disp_interval == 0:
            end = time.time()

            loss_temp /= count_step
            loss_rpn_cls_temp /= count_step
            loss_rpn_box_temp /= count_step
            loss_rcnn_cls_temp /= count_step
            loss_rcnn_box_temp /= count_step

            if logger:
                logger.log_scalar("loss", loss_temp, total_step)
                logger.log_scalar("loss_rpn_cls", loss_rpn_cls_temp, total_step)
                logger.log_scalar("loss_rpn_box", loss_rpn_box_temp, total_step)
                logger.log_scalar("loss_rcnn_cls", loss_rcnn_cls_temp, total_step)
                logger.log_scalar("loss_rcnn_box", loss_rcnn_box_temp, total_step)
                logger.log_scalar("dloss_s_s", dloss_s_s_temp / count_step, total_step)
                logger.log_scalar("dloss_t_t", dloss_t_t_temp / count_step, total_step)
                for dataset_name, v in dloss_dict.items():
                    for loss_name, value in v.items():
                        logger.log_scalar("{}:{}".format(dataset_name, loss_name), value / count_step, total_step)
                        v[loss_name] = 0

            count_step = 0
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0
            dloss_s_s_temp = 0
            dloss_t_t_temp = 0
    return total_step