# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from model.faster_rcnn.faster_rcnn import _fasterRCNN 
from model.faster_rcnn.vgg16_mindspore import vgg16_mindspore
import copy

import mindspore.nn as nn 

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg_c = vgg16_mindspore()

    # if self.pretrained: pass # todo: fix_mindspore
   
    vgg_t = copy.deepcopy(vgg_c)    

    vgg_c.classifier = nn.SequentialCell(*list(vgg_c.classifier)[:-1])
    vgg_t.classifier = nn.SequentialCell(*list(vgg_t.classifier)[:-1])


    # not using the last maxpool layer
    self.RCNN_base_c = nn.SequentialCell(*list(vgg_c.layers)[:-8])

    self.RCNN_base_t = nn.SequentialCell(*list(vgg_t.layers)[:-8])
    # finer feature map
    self.RCNN_base_fused = nn.SequentialCell(*list(vgg_c.layers)[-7:-1])
    self.RCNN_base_f1 = copy.deepcopy(self.RCNN_base_fused)
    self.RCNN_base_f2 = copy.deepcopy(self.RCNN_base_fused)

    # # Fix the layers before conv3:  #todo: fix_mindspore
    # for layer in range(10):
    #   for p in self.RCNN_base_c[layer].parameters(): p.requires_grad = False
    #   for p in self.RCNN_base_t[layer].parameters(): p.requires_grad = False

    self.RCNN_top = vgg_c.classifier
    self.RCNN_top_align = vgg_t.classifier
    self.RCNN_top_ref = copy.deepcopy(vgg_c.classifier)
    self.RCNN_top_sens = copy.deepcopy(vgg_c.classifier)


    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Dense(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Dense(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Dense(4096, 4 * self.n_classes) 
 
    self.RCNN_bbox_align_pred = nn.Dense(4096, 2) 

    self.confidence_ref = nn.Dense(4096, self.n_classes)
    self.confidence_sens = nn.Dense(4096, self.n_classes)  

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.shape[0], -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

  def _head_to_tail_align(self, pool5):
    
    pool5_flat = pool5.view(pool5.shape[0], -1)
    fc7 = self.RCNN_top_align(pool5_flat)

    return fc7

  def ref_branch(self, pooled_feat_c):
    return self.confidence_ref(self.RCNN_top_ref(pooled_feat_c.view(pooled_feat_c.shape[0], -1)))
  

  def sens_branch(self, pooled_feat_t):
    return self.confidence_sens(self.RCNN_top_sens(pooled_feat_t.view(pooled_feat_t.shape[0], -1)))

