# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import mindspore 
import mindspore.ops as ops 
from mindspore import Tensor, Parameter 
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
import mindspore.dataset as ds

mindspore.set_context(device_target="CPU")  # according to your device


from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes

from model.rpn.bbox_transform import bbox_transform_inv
from model.faster_rcnn.vgg16 import vgg16
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():

  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a AR-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='kaist', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16',
                      default='vgg16', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save detection results', default="./output/",
                      type=str)
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                        default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=17783, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--reasonable', dest='reasonable',
                      help='eval for the reasonable subset',
                      action='store_true')
  parser.add_argument('--sx', dest='shift_x',
                      help='shift along x axis',
                      default='0', type=str)
  parser.add_argument('--sy', dest='shift_y',
                      help='shift along y axis',
                      default='0', type=str)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "kaist":
      args.imdb_name = "kaist_trainval"
      args.imdbval_name = "kaist_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32, 64]', 'MAX_NUM_GT_BOXES', '20', 'FEAT_STRIDE','[8,]', \
                        'SHIFT_X', args.shift_x, 'SHIFT_Y', args.shift_y]
  else:
      print("dataset is not defined")
      pdb.set_trace()


  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.noise = False
  cfg.aug = False
  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  load_name = './smpd_mindspore.ckpt'

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = mindspore.load_checkpoint(load_name)
  mindspore.load_param_into_net(fasterRCNN, checkpoint)
  print('load model successfully!')
  

  start = time.time()
  max_per_image = 100
  thresh = 0.00

  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  dataset_generator = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                         imdb.num_classes, training=False, normalize = False)
  dataloader = ds.GeneratorDataset(dataset_generator, column_names=["data", "im_info", "gt_boxes", "gt_boxes_sens", "num_boxes"], shuffle=False)
  dataloader = dataloader.batch(1, True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}

  if args.reasonable:
      ignore_thresh = 40
  else:
      ignore_thresh = 0

  nms = P.NMSWithMask(cfg.TEST.NMS)
  
  result_path =  './result_{}/'.format(args.dataset)

  path = os.path.join(result_path,'det')
  if not os.path.exists(path):
    os.makedirs(path)

  #fasterRCNN.eval() #fix_mindspore
  fasterRCNN.set_train(False)
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):
        has_person = False
        pred_dets = None
        data = next(data_iter)

        im_data = Parameter(data[0])
        im_info = Parameter(data[1])
        gt_boxes = Parameter(data[2])
        gt_boxes_sens = Parameter(data[3])
        num_boxes = Parameter(data[4])

        det_tic = time.time()
        rois, cls_prob, bbox_pred = fasterRCNN(im_data, im_info, gt_boxes, gt_boxes_sens, num_boxes)

        scores = cls_prob
        boxes = rois[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              if args.class_agnostic:
                box_deltas = box_deltas.reshape(-1, 4) * Tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + Tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.reshape(1, -1, 4)
              else:
                box_deltas = box_deltas.reshape(-1, 4) * Tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + Tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.reshape(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info, 1)
        else:
            # Simply repeat the boxes, once for each class
            _ = np.tile(boxes, (1, scores.shape[1]))
            pred_boxes = Tensor(_, dtype=mstype.float32)

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()

        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        for j in xrange(1, imdb.num_classes):
            inds = ops.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
              cls_scores = scores[:,j][inds]

              _, order = ops.sort(cls_scores, 0, True)
              if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
              else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              
              cls_dets = ops.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
              cls_dets = cls_dets[order]

              cls_dets, indices, mask = nms(cls_dets)
              cls_dets = cls_dets[mask]
          
              all_boxes[j][i] = cls_dets.asnumpy()
              pred_dets = cls_dets.asnumpy()
            else:
              all_boxes[j][i] = empty_array
              pred_dets = empty_array
        
      
        image_name_save = imdb.image_index[i].replace('/', '_')
        image_set_file = os.path.join(result_path, 'det', image_name_save + '.txt')

        list_file = open(image_set_file, 'w')
        for box in all_boxes[1][i]:
          width = box[3] - box[1]
          if width < ignore_thresh:
            continue
          image_write_txt = 'person' + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' \
                    + str(box[2]) + ' ' + str(box[3]) + ' ' + str(box[4]*100)
          list_file.write(image_write_txt)
          list_file.write('\n')
        list_file.close()

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()
 
  
