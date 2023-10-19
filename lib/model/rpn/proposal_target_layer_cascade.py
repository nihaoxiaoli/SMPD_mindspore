# -*- coding:utf-8 -*-
from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import mindspore.nn as nn 
from mindspore import Tensor, Parameter 
from mindspore.common import dtype as mstype

import numpy as np
import numpy.random as npr
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch, clip_boxes, bbox_transform_inv
import copy

import pdb

class _ProposalTargetLayer(nn.Cell):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = Tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=mstype.float32)
        self.BBOX_NORMALIZE_STDS = Tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=mstype.float32)
        self.BBOX_INSIDE_WEIGHTS = Tensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS, dtype=mstype.float32)

    def construct(self, all_rois, gt_boxes, gt_boxes_sens, num_boxes, jitter, im_info): #todo: fix_mindspore
        pass
        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

        
