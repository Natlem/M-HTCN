# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------


import os
import os.path as osp
import PIL
#from model.utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from model.utils.config import cfg
import pdb

ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')

class imdb(object):
  """Image database."""

  def __init__(self, name, classes=None):
    self._name = name
    self._num_classes = 0
    if not classes:
      self._classes = []
    else:
      self._classes = classes
    self._image_index = []
    self._obj_proposer = 'gt'
    self._roidb = None
    self._roidb_handler = self.default_roidb
    # Use this dict for storing dataset specific config options
    self.config = {}

  @property
  def name(self):
    return self._name

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def image_index(self):
    return self._image_index

  @property
  def roidb_handler(self):
    return self._roidb_handler

  @roidb_handler.setter
  def roidb_handler(self, val):
    self._roidb_handler = val

  def set_proposal_method(self, method):
    method = eval('self.' + method + '_roidb')
    self.roidb_handler = method

  @property
  def roidb(self):
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   gt_overlaps
    #   gt_classes
    #   flipped
    if self._roidb is not None:
      return self._roidb
    self._roidb = self.roidb_handler()
    return self._roidb

  @property
  def cache_path(self):
    cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
    if not os.path.exists(cache_path):
      os.makedirs(cache_path)
    return cache_path

  @property
  def num_images(self):
    return len(self.image_index)

  def image_path_at(self, i):
    raise NotImplementedError

  def image_id_at(self, i):
    raise NotImplementedError

  def default_roidb(self):
    raise NotImplementedError

  def evaluate_detections(self, all_boxes, output_dir=None):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    raise NotImplementedError

  def _get_widths(self):
    return [PIL.Image.open(self.image_path_at(i)).size[0]
            for i in range(self.num_images)]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()

    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      if "wildtrack" in self.name:
        for b in range(len(boxes)):
          if boxes[b][2] < boxes[b][0]:
            boxes[b][0] = 0

      try:
        assert (boxes[:, 2] >= boxes[:, 0]).all()
      except:
        print('bounding error. This may be ignored for some dataset since their train set is not properly annotated and we do not need labels')
        print(boxes[:, 2] >= boxes[:, 0])
        print(boxes)
        print(widths[i])
      if 'seg_map' in self.roidb[i].keys():
        seg_map = self.roidb[i]['seg_map'][::-1, :]
        entry = {'boxes': boxes,
                 'gt_overlaps': self.roidb[i]['gt_overlaps'],
                 'gt_classes': self.roidb[i]['gt_classes'],
                 'flipped': True,
                 'seg_map':seg_map}
      else:
        entry = {'boxes': boxes,
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'gt_classes': self.roidb[i]['gt_classes'],
               'flipped': True}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2




  @staticmethod
  def merge_roidbs(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
      a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
      a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                      b[i]['gt_classes']))
      a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                 b[i]['gt_overlaps']])
      a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                     b[i]['seg_areas']))
    return a

  def competition_mode(self, on):
    """Turn competition mode on or off."""
    pass
