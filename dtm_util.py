import torch
from torch.nn import functional as F
from pathlib import Path


def get_mask_for_target(args, FL, total_step,
                                       dataloader_s, iters_per_epoch,
                                       fasterRCNN, mask, optimizer, device, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    dloss_s_s_temp = 0.
    dloss_t_t_temp = 0.

    data_iter_s = iter(dataloader_s)
    m_data_iters_t = []
    dloss_dict = {}
    with_source = []

    count_step = 0

    fasterRCNN.zero_grad()

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

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel_s, out_d_s, out_d_mid_s, out_d_ins_s, _ = fasterRCNN(mask(im_data), im_info, gt_boxes,
                                                                                     num_boxes, with_feat=True, eta=-1)
        loss = 0.
        # loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
        #        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        #
        # loss_temp += loss.item()
        # loss_rpn_cls_temp += rpn_loss_cls.mean().item()
        # loss_rpn_box_temp += rpn_loss_box.mean().item()
        # loss_rcnn_cls_temp += RCNN_loss_cls.mean().item()
        # loss_rcnn_box_temp += RCNN_loss_bbox.mean().item()

        ######################### da loss 1 #####################################
        # domain label
        domain_s = torch.ones(out_d_s.size(0)).long().to(device)
        # global alignment loss
        dloss_s = 0.5 * FL(out_d_s, domain_s)

        ######################### da loss 2 #####################################
        # domain label
        domain_s_mid = torch.ones(out_d_mid_s.size(0)).long().to(device)
        ##### mid alignment loss
        dloss_s_mid = 0.5 * F.cross_entropy(out_d_mid_s, domain_s_mid)

        ######################### da loss 3 #####################################
        # local alignment loss
        dloss_s_p = 0.5 * torch.mean((1 - out_d_pixel_s) ** 2)

        ######################### da loss 4 #####################################
        # instance alignment loss
        domain_gt_ins = torch.ones(out_d_ins_s.size(0)).long().to(device)
        dloss_s_ins = 0.5 * FL(out_d_ins_s, domain_gt_ins)
        ##############################################################
        dloss_s_s = dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5
        dloss_s_s_temp += dloss_s_s.item()
        loss += dloss_s_s
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_temp += loss.item()

        if step % args.disp_interval == 0:
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
                dloss_s_p = dloss_s_p.item()


            if logger:
                logger.log_scalar("loss", loss_temp, total_step)
                # logger.log_scalar("loss_rpn_cls", loss_rpn_cls_temp, total_step)
                # logger.log_scalar("loss_rpn_box", loss_rpn_box_temp, total_step)
                # logger.log_scalar("loss_rcnn_cls", loss_rcnn_cls_temp, total_step)
                # logger.log_scalar("loss_rcnn_box", loss_rcnn_box_temp, total_step)

            count_step = 0
            loss_temp_last = loss_temp
            loss_temp = 0
            loss_rpn_cls_temp = 0
            loss_rpn_box_temp = 0
            loss_rcnn_cls_temp = 0
            loss_rcnn_box_temp = 0

def get_umap_feats(args, FL, total_step,
                                       dataloader_s, m_dataloader_t, iters_per_epoch,
                                       fasterRCNN, mask, device, feat_num=2, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    dloss_s_s_temp = 0.
    dloss_t_t_temp = 0.

    data_iter_s = iter(dataloader_s)
    m_data_iters_t = []
    dloss_dict = {}
    with_source = []
    t_feats_l = {}
    for i, dataloader_t in enumerate(m_dataloader_t):
        with_source.append(0)
        m_data_iters_t.append(iter(dataloader_t))
        t_feats_l[args.dataset_t[i]] = []

    count_step = 0

    fasterRCNN.zero_grad()
    s_feats_l = []
    m_feats_l = []

    min_w = 9999
    min_h = 9999


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

        with torch.no_grad():
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_pixel_s, out_d_s, out_d_mid_s, out_d_ins_s, s_feats = fasterRCNN(im_data, im_info, gt_boxes,
                                                                                         num_boxes, with_feat=True, eta=-1)

            if s_feats[feat_num].shape[2] < min_h:
                min_h = s_feats[feat_num].shape[2]
            if s_feats[feat_num].shape[3] < min_w:
                min_w = s_feats[feat_num].shape[3]



            s_feats_l.append(s_feats[feat_num].detach().cpu())

            if mask is not None:
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, out_d_pixel_s, out_d_s, out_d_mid_s, out_d_ins_s, m_feats = fasterRCNN(mask(im_data), im_info, gt_boxes,
                                                                                             num_boxes, with_feat=True, eta=-1)
                m_feats_l.append(m_feats[feat_num].detach().cpu())

                if m_feats[feat_num].shape[2] < min_h:
                    min_h = m_feats[feat_num].shape[2]
                if m_feats[feat_num].shape[3] < min_w:
                    min_w = m_feats[feat_num].shape[3]

        for i, data_iter_t in enumerate(m_data_iters_t):
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(m_dataloader_t[i])
                data_t = next(data_iter_t)

            im_data = data_t[0].to(device)
            im_info = data_t[1].to(device)

            with torch.no_grad():
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, out_d_pixel_s, out_d_s, out_d_mid_s, out_d_ins_s, t_feats = fasterRCNN(im_data, im_info,
                                                                                                   gt_boxes,
                                                                                                   num_boxes,
                                                                                                   with_feat=True,
                                                                                                   eta=-1)

                if t_feats[feat_num].shape[2] < min_h:
                    min_h = t_feats[feat_num].shape[2]
                if t_feats[2].shape[3] < min_w:
                    min_w = t_feats[feat_num].shape[3]

                t_feats_l[args.dataset_t[i]].append(t_feats[feat_num].detach().cpu())
    return s_feats_l, m_feats_l, t_feats_l, min_h, min_w

def get_only_images(args, FL, total_step,
                                       dataloader_s, m_dataloader_t, iters_per_epoch,
                                       fasterRCNN, mask, device, feat_num=2, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    dloss_s_s_temp = 0.
    dloss_t_t_temp = 0.

    data_iter_s = iter(dataloader_s)
    m_data_iters_t = []
    dloss_dict = {}
    with_source = []
    t_feats_l = {}
    for i, dataloader_t in enumerate(m_dataloader_t):
        with_source.append(0)
        m_data_iters_t.append(iter(dataloader_t))
        t_feats_l[args.dataset_t[i]] = []

    count_step = 0

    fasterRCNN.zero_grad()
    s_feats_l = []
    m_feats_l = []

    min_w = 9999
    min_h = 9999


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


        if im_data.shape[2] < min_h:
            min_h = im_data.shape[2]
        if im_data.shape[3] < min_w:
            min_w = im_data.shape[3]



        s_feats_l.append(im_data.detach().cpu())

        if mask is not None:
            im_t_data = mask(im_data)
            m_feats_l.append(im_t_data.detach().cpu())

        for i, data_iter_t in enumerate(m_data_iters_t):
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(m_dataloader_t[i])
                data_t = next(data_iter_t)

            im_data = data_t[0].to(device)
            im_info = data_t[1].to(device)

            if im_data.shape[2] < min_h:
                min_h = im_data.shape[2]
            if im_data.shape[3] < min_w:
                min_w = im_data.shape[3]

            t_feats_l[args.dataset_t[i]].append(im_data.detach().cpu())
    return s_feats_l, m_feats_l, t_feats_l, min_h, min_w


def train_htcn_one_epoch_ida_with_dtm(args, FL, total_step,
                                      dataloader_s, masks_dict, dataloader_t, iters_per_epoch,
                                      fasterRCNN, optimizer, device, eta=1.0, alpha=1, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    dloss_s_s_temp = 0.
    dloss_t_t_temp = 0.

    data_iter_s = iter(dataloader_s)
    data_iter_t = iter(dataloader_t)
    mloss_dict ={}
    for m_name, mask in masks_dict.items():
        mloss_dict[m_name] = {}
        mloss_dict[m_name]['mask_d_loss'] = 0

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

        for m_name, mask in masks_dict.items():
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_pixel, out_d, out_d_mid, out_d_ins, _ = fasterRCNN(mask(im_data), im_info, gt_boxes,
                                                                                 num_boxes, with_feat=True)
            sup_mask_loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            #mloss_dict[m_name]['mask_sup_loss'] += sup_mask_loss.item()
            #loss += sup_mask_loss

            domain_s = torch.zeros(out_d.size(0)).long().to(device) + 1
            # global alignment loss
            dloss_s = 0.5 * FL(out_d, domain_s)

            ######################### da loss 2 #####################################
            # domain label
            domain_s_mid = torch.zeros(out_d_mid.size(0)).long().to(device) + 1
            ##### mid alignment loss
            dloss_s_mid = 0.5 * F.cross_entropy(out_d_mid, domain_s_mid)

            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

            ######################### da loss 4 #####################################
            # instance alignment loss
            domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device) + 1
            dloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##############################################################
            dloss_s_s = dloss_s + dloss_s_p + dloss_s_mid * 0.15 + dloss_s_ins * 0.5
            mloss_dict[m_name]['mask_d_loss'] += dloss_s_s.item()
            loss += dloss_s_s * eta * alpha

        try:
            data_t = next(data_iter_t)
        except:
            data_iter_t = iter(dataloader_t)
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
        domain_t = torch.zeros(out_d.size(0)).long().to(device) + 1
        dloss_t = 0.5 * FL(out_d, domain_t)

        ######################### da loss 2 #####################################
        # domain label
        domain_t_mid = torch.zeros(out_d_mid.size(0)).long().to(device) + 1
        ##### mid alignment loss
        dloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

        ######################### da loss 3 #####################################
        # local alignment loss
        dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

        ######################### da loss 4 #####################################
        # instance alignment loss
        domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device) + 1
        dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
        ##############################################################
        dloss_t_t = dloss_t + dloss_t_p + dloss_t_mid * 0.15 + dloss_t_ins * 0.5
        dloss_t_t_temp += dloss_t_t.item()

        loss += dloss_t_t * eta


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % args.disp_interval == 0:
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
                for dataset_name, v in mloss_dict.items():
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


def da_loss_fn(FL, device, out_d, out_d_ins, out_d_mid, out_d_pixel, label_num):
    domain_label = torch.zeros(out_d.size(0)).long().to(device)
    # global alignment loss
    dloss = 0.5 * FL(out_d, domain_label)
    ######################### da loss 2 #####################################
    # domain label
    domain_label_mid = torch.zeros(out_d_mid.size(0)).long().to(device) + label_num
    ##### mid alignment loss
    dloss_mid = 0.5 * F.cross_entropy(out_d_mid, domain_label_mid)
    ######################### da loss 3 #####################################
    # local alignment loss
    if label_num == 0:
        dloss_p = 0.5 * torch.mean(out_d_pixel ** 2)
    else:
        dloss_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)
    ######################### da loss 4 #####################################
    # instance alignment loss
    domain_gt_ins = torch.zeros(out_d_ins.size(0)).long().to(device) + label_num
    dloss_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
    ##############################################################
    dloss = dloss + dloss_p + dloss_mid * 0.15 + dloss_ins * 0.5
    return dloss

def eval_samples_for_herding_gradient(args, FL, total_step,
                                      dataloader_t, iters_per_epoch,
                                    fasterRCNN, optimizer, device, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    dloss_s_s_temp = 0.
    dloss_t_t_temp = 0.

    data_iter_t = iter(dataloader_t)


    for step in range(1, iters_per_epoch + 1):
        try:
            data_t = next(data_iter_t)
        except:
            data_iter_t = iter(dataloader_t)
            data_t = next(data_iter_t)
        im_data = data_t[0].to(device)
        im_info = data_t[1].to(device)

        gt_boxes = torch.zeros((1, 1, 5)).to(device)
        num_boxes = torch.zeros([1]).to(device)
        data_p = Path(data_t[4][0]).stem

        out_d_pixel, out_d, out_d_mid, out_d_ins, stu_t_feats = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                                                           target=True, with_feat=True)
        dloss_t_t = da_loss_fn(FL, device, out_d, out_d_ins, out_d_mid, out_d_pixel, 1)

        dloss_t_t.backward()
        total_abs_grad = {}
        local_grad = 0.
        for n, p in fasterRCNN.named_parameters():
            if hasattr(p, 'grad') or p.grad is not None:
                local_grad += torch.abs(p.grad.mean()).cpu()
        total_abs_grad[data_p] = local_grad
        optimizer.zero_grad()
    print("2")


    return total_abs_grad

def eval_mask_for_target(args, FL, total_step,
                                       dataloader_s, m_dataloader_t, iters_per_epoch,
                                       fasterRCNN, mask, optimizer, device, logger=None):
    loss_temp = 0
    loss_rpn_cls_temp = 0
    loss_rpn_box_temp = 0
    loss_rcnn_cls_temp = 0
    loss_rcnn_box_temp = 0
    dloss_s_s_temp = 0.
    dloss_t_t_temp = 0.

    data_iter_s = iter(dataloader_s)
    m_data_iters_t = []
    dloss_dict = {}
    with_source = []


    count_step = 0
    correct_pred_1 = 0
    correct_pred_2 = 0

    fasterRCNN.zero_grad()

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

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel_s, out_d_s, out_d_mid_s, out_d_ins_s, _ = fasterRCNN(mask(im_data), im_info, gt_boxes,
                                                                                     num_boxes, with_feat=True, eta=-1)
        loss = 0.
        ######################### da loss 1 #####################################
        # domain label
        domain_s = torch.ones(out_d_s.size(0)).long().to(device)
        pred = out_d_s.data.max(1, keepdim=True)[1]
        correct_pred_1 += pred.eq(domain_s.data.view_as(pred)).sum()

        ######################### da loss 2 #####################################
        # domain label
        domain_s_mid = torch.ones(out_d_mid_s.size(0)).long().to(device)
        ##### mid alignment loss
        pred = out_d_mid_s.data.max(1, keepdim=True)[1]
        correct_pred_2 += pred.eq(domain_s_mid.data.view_as(pred)).sum()


        ######################### da loss 3 #####################################
        # local alignment loss
        dloss_s_p = 0.5 * torch.mean((1 - out_d_pixel_s) ** 2)

        ######################### da loss 4 #####################################
        # instance alignment loss
        domain_gt_ins = torch.ones(out_d_ins_s.size(0)).long().to(device)
        dloss_s_ins = 0.5 * FL(out_d_ins_s, domain_gt_ins)

    correct_pred_1 /= len(dataloader_s)
    correct_pred_2 /= len(dataloader_s)

    print(correct_pred_1)
    print(correct_pred_2)

