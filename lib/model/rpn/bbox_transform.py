# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import numpy as np
import pdb
import mindspore


def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = mindspore.ops.exp(dw) * widths.unsqueeze(2)
    pred_h = mindspore.ops.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas#.clone()  #fix_mindspore
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes



def bbox_contextual_batch(rois, scale=2):
    new_rois = rois.new(rois.size()).zero_()
       
    if rois.dim() == 2:
        widths = rois[:, 3] - rois[:, 1] + 1.0
        heights = rois[:, 4] - rois[:, 2] + 1.0
        ctr_x = rois[:, 1] + 0.5 * widths
        ctr_y = rois[:, 2] + 0.5 * heights

        widths *= scale
        heights *= scale
        # xmin
        new_rois[:, 1] = ctr_x - 0.5 * widths
        # ymin
        new_rois[:, 2] = ctr_y - 0.5 * heights
        # xmax
        new_rois[:, 3] = ctr_x + 0.5 * widths
        # ymax
        new_rois[:, 4] = ctr_y + 0.5 * heights

    elif rois.dim() == 3:
        widths = rois[:, :, 3] - rois[:, :, 1] + 1.0
        heights = rois[:,:, 4] - rois[:,:, 2] + 1.0
        ctr_x = rois[:, :, 1] + 0.5 * widths
        ctr_y = rois[:, :, 2] + 0.5 * heights

        widths *= scale
        heights *= scale
        # xmin
        new_rois[:, :, 1] = ctr_x - 0.5 * widths
        # ymin
        new_rois[:, :, 2] = ctr_y - 0.5 * heights
        # xmax
        new_rois[:, :, 3] = ctr_x + 0.5 * widths
        # ymax
        new_rois[:, :, 4] = ctr_y + 0.5 * heights

    else:
        raise ValueError('ex_roi input dimension is not correct.')

    return new_rois
    

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4] = boxes[i,:,0::4].clamp(0.0, im_shape[i, 1]-1)
        boxes[i,:,1::4] = boxes[i,:,1::4].clamp(0.0, im_shape[i, 0]-1)
        boxes[i,:,2::4] = boxes[i,:,2::4].clamp(0.0, im_shape[i, 1]-1)
        boxes[i,:,3::4] = boxes[i,:,3::4].clamp(0.0, im_shape[i, 0]-1)

    return boxes