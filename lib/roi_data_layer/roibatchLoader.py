
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from PIL import Image

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

import mindspore
from mindspore import Tensor, Parameter 
from mindspore.common import dtype as mstype
import mindspore.ops as ops

class roibatchLoader:
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = Tensor(np.zeros(self.data_size), mstype.float32)
    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = np.array(1)

        self.ratio_list_batch[left_idx:(right_idx+1)] = Tensor(target_ratio, mstype.float32) # trainset ratio list ,each batch is same number


  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    blobs = get_minibatch(minibatch_db, self._num_classes)
    data_c = Tensor(blobs['data'][0])
    data_t = Tensor(blobs['data'][1])
    im_info = Tensor(blobs['im_info'])
    # we need to random shuffle the bounding box.
    #print(data_c.shape)
    data_height, data_width = data_c.shape[1], data_c.shape[2]
    if self.training:  # todo: need to verified
        state = np.random.get_state()
        np.random.shuffle(blobs['gt_boxes'])
        np.random.set_state(state)
        np.random.shuffle(blobs['gt_boxes_sens'])
        gt_boxes = Tensor(blobs['gt_boxes'])
        gt_boxes_sens = Tensor(blobs['gt_boxes_sens'])

        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        ratio = self.ratio_list_batch[index]

        if self._roidb[index_ratio]['need_crop']:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
             
                _, min_y = int(ops.ArgMinWithValue()(gt_boxes[:,1]))
                _, max_y = int(ops.ArgMaxWithValue()(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                if trim_size > data_height:
                    trim_size = data_height                
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data_c = data_c[:, y_s:(y_s + trim_size), :, :]
                data_t = data_t[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)
                gt_boxes_sens[:, 1] = gt_boxes_sens[:, 1] - float(y_s)
                gt_boxes_sens[:, 3] = gt_boxes_sens[:, 3] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)
                gt_boxes_sens[:, 1].clamp_(0, trim_size - 1)
                gt_boxes_sens[:, 3].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                _, min_x = int(ops.ArgMinWithValue()(gt_boxes[:,0]))
                _, max_x = int(ops.ArgMaxWithValue()(gt_boxes[:,2]))

                trim_size = int(np.ceil(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width                
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                gt_boxes_sens[:, 0] = gt_boxes_sens[:, 0] - float(x_s)
                gt_boxes_sens[:, 2] = gt_boxes_sens[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)
                gt_boxes_sens[:, 0].clamp_(0, trim_size - 1)
                gt_boxes_sens[:, 2].clamp_(0, trim_size - 1)

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            padding_data_c = mindspore.tensor.zeros((int(np.ceil(data_width / ratio)), data_width, 3), mindspore.float32)
            padding_data_c[:data_height, :, :] = data_c[0]

            padding_data_t = mindspore.tensor.zeros((int(np.ceil(data_width / ratio)), data_width, 3), mindspore.float32)
            padding_data_t[:data_height, :, :] = data_t[0]

            # update im_info
            im_info[0, 0] = padding_data_c.shape[0]
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data_c = mindspore.tensor.zeros((int(np.ceil(data_width * ratio)), data_width, 3), mindspore.float32)
            padding_data_c[:, :data_width, :] = data_c[0]
            padding_data_t = mindspore.tensor.zeros((int(np.ceil(data_width * ratio)), data_width, 3), mindspore.float32)
            padding_data_t[:, :data_width, :] = data_t[0]

            im_info[0, 1] = padding_data_c.shape[1]
        else:
            trim_size = min(data_height, data_width)
            padding_data_c = mindspore.tensor.zeros((trim_size, trim_size, 3), mindspore.float32)
            padding_data_c = data_c[0][:trim_size, :trim_size, :]
            padding_data_t = mindspore.tensor.zeros((trim_size, trim_size, 3), mindspore.float32)
            padding_data_t = data_t[0][:trim_size, :trim_size, :]
            gt_boxes[:, :4].clamp_(0, trim_size)
            gt_boxes_sens[:, :4].clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size


        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3]) \
                    | (gt_boxes_sens[:,0] == gt_boxes_sens[:,2]) | (gt_boxes_sens[:,1] == gt_boxes_sens[:,3])
        
        not_keep = mindspore.tensor.equal(not_keep, 0) 
        keep = mindspore.tensor.nonzero(not_keep).view(-1)
        gt_boxes_padding = mindspore.tensor.zeros((self.max_num_box, gt_boxes.shape[1]), mindspore.float32) 
        gt_boxes_sens_padding = mindspore.tensor.zeros((self.max_num_box, gt_boxes_sens.shape[1]), mindspore.float32)
        
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            gt_boxes_sens = gt_boxes_sens[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
            gt_boxes_sens_padding[:num_boxes,:] = gt_boxes_sens[:num_boxes]
        else:
            num_boxes = 0

        padding_data_c = padding_data_c.permute(2, 0, 1)
        padding_data_t = padding_data_t.permute(2, 0, 1)
        padding_data = [padding_data_c, padding_data_t]

        im_info = im_info.view(3)

        return padding_data, im_info, gt_boxes_padding, gt_boxes_sens_padding, num_boxes
    else:
        data_c = data_c.permute(0, 3, 1, 2).view(3, data_height, data_width)
        data_t = data_t.permute(0, 3, 1, 2).view(3, data_height, data_width)
        data_c = mindspore.ops.expand_dims(data_c, axis=0)
        data_t = mindspore.ops.expand_dims(data_t, axis=0)
        data = mindspore.ops.cat((data_c, data_t), axis=0)

        im_info = im_info.view(3)

        gt_boxes = Tensor([1,1,1,1,1])  #fix_mindspore
        gt_boxes_sens = Tensor([1,1,1,1])
        num_boxes = 0

        return data, im_info, gt_boxes, gt_boxes_sens, num_boxes

  def __len__(self):
    return len(self._roidb)


