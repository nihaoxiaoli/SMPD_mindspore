# add uncertainty for enhance

import random

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F

import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import copy
import pdb
from model.rpn.bbox_transform import bbox_contextual_batch, clip_boxes, bbox_transform_inv, bbox_transform_ref
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from model.dloss import ce_loss

def DS_Combin(output, class_num = 2):
    def DS_Combin_two(alpha1, alpha2):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        #print(alpha1.shape, alpha2.shape)
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = mindspore.ops.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand_as(E[v]))
            u[v] = class_num /S[v]

        # b^0 @ b^(0+1)
        bb = mindspore.ops.bmm(b[0].view(-1, class_num, 1), b[1].view(-1, 1, class_num))
        # b^0 * u^1
        uv1_expand = u[1].expand_as(b[0])
        bu = mindspore.ops.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand_as(b[0])
        ub = mindspore.ops.mul(b[1], uv_expand)
        # calculate C
        bb_sum = mindspore.ops.sum(bb, dim=(1, 2))
        bb_diag = mindspore.ops.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (mindspore.ops.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand_as(b[0]))
        # calculate u^a
        u_a = mindspore.ops.mul(u[0], u[1])/((1-C).view(-1, 1).expand_as(u[0]))

        # calculate new S
        S_a = class_num / u_a
        # calculate new e_k
        e_a = mindspore.ops.mul(b_a, S_a.expand_as(b_a))
        alpha_a = e_a + 1
        return alpha_a

    alpha = [mindspore.ops.exp(mindspore.ops.clamp(o, -10, 10))+1 for o in output]
    for v in range(len(alpha)-1):
        if v==0:
            print('alpha_a', alpha[0].shape, alpha[1].shape)
            alpha_a = DS_Combin_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
    return alpha_a


def Modal_Enhancement(x, x_lwir, x_conf, x_lwir_conf):

    x_conf = x_conf.unsqueeze(1).unsqueeze(2)
    x_lwir_conf = x_lwir_conf.unsqueeze(1).unsqueeze(2)

    weight_conf = x_conf / (x_conf + x_lwir_conf)

    substracted = x - x_lwir
    subtracted_weight = nn.AdaptiveAvgPool2d((1, 1))(substracted)
    excitation_weight = mindspore.ops.tanh(subtracted_weight)

    substracted2 = x_lwir - x
    subtracted_weight2 = nn.AdaptiveAvgPool2d((1, 1))(substracted2)
    excitation_weight2 = mindspore.ops.tanh(subtracted_weight2)

    x_weight = x * excitation_weight * weight_conf
    x_lwir_weight = x_lwir * excitation_weight2 * (1-weight_conf)

    x_mix =  x_lwir_weight + x
    x_lwir_mix = x_lwir + x_weight

    return x_mix, x_lwir_mix


class _fasterRCNN(nn.Cell):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, pool_scale=8.0):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn_c = _RPN(self.dout_base_model)
        self.RCNN_rpn_t = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = P.ROIAlign(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
                                   1.0/pool_scale, 2, 0)


    def construct(self, im_data, im_info, gt_boxes, gt_boxes_sens, num_boxes):
        im_data = mindspore.ops.swapaxes(im_data, 0, 1)
        batch_size = im_data[0].shape[0]

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        gt_boxes_sens = gt_boxes_sens.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat_c = self.RCNN_base_c(im_data[0])
        base_feat_t = self.RCNN_base_t(im_data[1])
        base_feat_fused = 0.5 * (base_feat_c + base_feat_t)
        base_feat_fused = self.RCNN_base_fused(base_feat_fused)
        conv5_c = self.RCNN_base_f1(base_feat_c)
        conv5_t = self.RCNN_base_f2(base_feat_t)


        # feed fused base feature map to RPN to obtain rois
        rois_c, rpn_loss_cls_c, rpn_loss_bbox_c = self.RCNN_rpn_c(conv5_c, im_info, gt_boxes, num_boxes)
        rois_t, rpn_loss_cls_t, rpn_loss_bbox_t = self.RCNN_rpn_t(conv5_t, im_info, gt_boxes, num_boxes)

        rois = mindspore.ops.cat([rois_c, rois_t], 1)

        rpn_loss_cls = rpn_loss_cls_c + rpn_loss_cls_t
        rpn_loss_bbox = rpn_loss_bbox_c + rpn_loss_bbox_t


        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            pass  # todo: fix_mindspore
        else:
            rois_jittered = copy.deepcopy(rois)
            rois_label = None
            rois_target = None
            rois_align_target = None
            rois_inside_ws = None
            rois_outside_ws = None


        pooled_feat_c = self.RCNN_roi_pool(conv5_c, rois.view(-1, 5))
        pooled_feat_t = self.RCNN_roi_pool(conv5_t, rois.view(-1, 5))


        if self.training:
            cls_score_ref = self.ref_branch(pooled_feat_c)
            cls_score_sens = self.sens_branch(pooled_feat_t)


        else:
            feat_map = []
            feat_map.append(mindspore.ops.sum(pooled_feat_c, 1))
            feat_map.append(mindspore.ops.sum(pooled_feat_t, 1))

            cls_score_ref = self.ref_branch(pooled_feat_c)
            cls_score_sens = self.sens_branch(pooled_feat_t)


            alpha_ref = mindspore.ops.exp(mindspore.ops.clamp(cls_score_ref, -10, 10))+1
            alpha_ses = mindspore.ops.exp(mindspore.ops.clamp(cls_score_sens, -10, 10))+1
            
            conf1 = mindspore.ops.sum(alpha_ref[:, 1:], dim=1, keepdim=True)
            conf2 = mindspore.ops.sum(alpha_ses[:, 1:], dim=1, keepdim=True)
            
            pooled_feat_c_mix, pooled_feat_t_mix = Modal_Enhancement(pooled_feat_c, pooled_feat_t, conf1, conf2)
            
            cls_score_ref = self.ref_branch(pooled_feat_c_mix)
            cls_score_sens = self.sens_branch(pooled_feat_t_mix)


        if self.training:
            bs = pooled_feat_c.shape[0]
            weight = mindspore.ops.rand(bs).cuda().unsqueeze(1).unsqueeze(2).unsqueeze(3)
            pooled_feat_c = weight * pooled_feat_c
            pooled_feat_t = (1-weight) * pooled_feat_t

        pooled_feat = pooled_feat_c + pooled_feat_t


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = mindspore.ops.gather_elements(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = nn.Softmax(1)(cls_score)


        RCNN_loss_cls = 0
        RCNN_loss_cls_ref = 0
        RCNN_loss_cls_sens = 0
        RCNN_loss_bbox = 0
        RCNN_loss_cls_fusion = 0

        cls_fusion = DS_Combin([cls_score, cls_score_ref, cls_score_sens], self.n_classes)
   
        cls_fusion_log = mindspore.ops.Log()(cls_fusion)
        cls_fusion_prob = nn.Softmax(1)(cls_fusion_log)

        if self.training:
            pass #todo: fix_mindspore

        cls_prob = cls_fusion_prob.view(batch_size, rois.shape[1], -1)
        bbox_pred = bbox_pred.view(batch_size, rois.shape[1], -1)  

        if self.training:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_cls_ref, RCNN_loss_cls_sens, RCNN_loss_bbox, rois_label, RCNN_loss_cls_fusion
        else:
            return rois, cls_prob, bbox_pred

    def create_architecture(self):
        self._init_modules()
        #self._init_weights()  #fix_mindspore
