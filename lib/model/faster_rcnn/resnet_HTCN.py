

from model.utils.config import cfg

from model.faster_rcnn.faster_rcnn_HTCN import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

from model.faster_rcnn.resnet import resnet50, resnet101, resnet152

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

def split_r_c(fea, r, c):
    # fea.shape = [1, 256, 190, 150]
    f_rows = fea.chunk(r, 2)  # a tuple, shape = [r], f_rows[0].shape = [1, 256, 19, 150]
    r_c = []
    for i in range(r):
        r_c.append(f_rows[i].chunk(c, 3))  # size=[r,c], r_c[0,0].shape = [1, 256, 19, 30]

    for i in range(r):
        if i == 0:
            f_new = torch.cat(r_c[i], 1)
        else:
            f_new_t = torch.cat(r_c[i], 1)
            f_new = torch.cat((f_new, f_new_t), 1)
    # f_new.shape = [1, 12800, 19, 30]
    return f_new

def merge_r_c(fea, r, c):
    # fea.shape = [1, 50, 19, 30]
    f_new_s = fea.chunk(r * c, 1)
    for i in range(r):
        if i == 0:
            f_re = torch.cat([f_new_s[k] for k in range(i * c, i * c + c)], 3)
        else:
            f_re_t = torch.cat([f_new_s[k] for k in range(i * c, i * c + c)], 3)
            f_re = torch.cat((f_re, f_re_t), 2) # [1, 1, 190, 150]
    return f_re


class netD_m_pixel(nn.Module):
    def __init__(self,r = 10, c = 5):
        super(netD_m_pixel, self).__init__()
        self.row = r
        self.col = c
        self.group = int(r * c)
        self.channels_in = int(256 * r * c)
        self.channels_mid = int(128 * r * c)
        self.channels_out = int(r * c)
        self.conv1 = nn.Conv2d(self.channels_in, self.channels_mid, kernel_size=1, stride=1,
                               padding=0, bias=False, groups = self.group)
        self.conv2 = nn.Conv2d(self.channels_mid, self.channels_mid, kernel_size=1, stride=1,
                               padding=0, bias=False, groups = self.group)
        self.conv3 = nn.Conv2d(self.channels_mid, self.channels_out, kernel_size=1, stride=1,
                               padding=0, bias=False, groups = self.group)
        self._init_weights()
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                # m.bias.data.zero_()

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = split_r_c(x, self.row,self.col) # [1, 12800, 19, 30]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x) # [1, 50, 19, 30]
        x = merge_r_c(x, self.row,self.col) #[1, 1, 190, 150]
        return F.sigmoid(x)

class netD_pixel(nn.Module):
    def __init__(self,context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          return F.sigmoid(x)

class netD_mid(nn.Module):
    def __init__(self,context=False):
        super(netD_mid, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat#torch.cat((feat1,feat2),1)#F
        else:
          return x


class netD(nn.Module):
    def __init__(self,context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(1024, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x
class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x

class netD_da(nn.Module):
    def __init__(self, feat_d):
        super(netD_da, self).__init__()
        self.fc1 = nn.Linear(feat_d,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x  #[256, 2]



class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)# 2
        self.output_dim = output_dim
        self.random_matrix = [torch.rand(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False,lc=False,gc=False, la_attention = False
               ,mid_attention = False):
    self.model_path = cfg.RESNET_PATH
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.lc = lc
    self.gc = gc
    self.layers = num_layers
    if self.layers == 50:
      self.model_path = cfg.RESNET50_PATH
    if self.layers == 152:
      self.model_path = cfg.RESNET152_PATH


    _fasterRCNN.__init__(self, classes, class_agnostic,lc,gc, la_attention, mid_attention)

  def _init_modules(self):

    resnet = resnet101()
    if self.layers == 50:
      resnet = resnet50()
    elif self.layers == 152:
      resnet = resnet152()
    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
    # Build resnet.
    self.RCNN_base1 = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1)
    self.RCNN_base2 = nn.Sequential(resnet.layer2)
    self.RCNN_base3 = nn.Sequential(resnet.layer3)

    self.netD_pixel = netD_pixel(context=self.lc)
    self.netD = netD(context=self.gc)
    self.netD_mid = netD_mid(context=self.gc)

    self.RCNN_top = nn.Sequential(resnet.layer4)
    feat_d = 2048
    feat_d2 = 384
    feat_d3 = 1024

    self.RandomLayer = RandomLayer([feat_d, feat_d2], feat_d3)
    self.RandomLayer.cuda()

    self.netD_da = netD_da(feat_d3)

    self.stu_feature_adap = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                          nn.ReLU())


    self.RCNN_cls_score = nn.Linear(feat_d+feat_d2, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(feat_d+feat_d2, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(feat_d+feat_d2, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base1[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base1[1].parameters(): p.requires_grad=False

    # assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    # if cfg.RESNET.FIXED_BLOCKS >= 3:
    #   for p in self.RCNN_base1[6].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 2:
    #   for p in self.RCNN_base1[5].parameters(): p.requires_grad=False
    #if cfg.RESNET.FIXED_BLOCKS >= 1:
    #  for p in self.RCNN_base1[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base1.apply(set_bn_fix)
    self.RCNN_base2.apply(set_bn_fix)
    self.RCNN_base3.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base1.eval()
      self.RCNN_base1[4].train()
      self.RCNN_base2.train()
      self.RCNN_base3.train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base1.apply(set_bn_eval)
      self.RCNN_base2.apply(set_bn_eval)
      self.RCNN_base3.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
