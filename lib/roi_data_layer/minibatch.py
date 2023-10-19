# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
#from scipy.misc import imread
import imageio
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import copy
import cv2
import pdb
import math

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  gt_boxes_sens = np.empty((len(gt_inds), 4), dtype=np.float32)
  gt_boxes_sens[:, 0:4] = roidb[0]['boxes_sens'][gt_inds, :] * im_scales[0]
  blobs['gt_boxes'] = gt_boxes
  blobs['gt_boxes_sens'] = gt_boxes_sens
  # print(roidb[0]['boxes'][gt_inds, :], roidb[0]['boxes_sens'][gt_inds, :], im_scales[0])
  blobs['im_info'] = np.array(
    [[im_blob[0].shape[1], im_blob[0].shape[2], im_scales[0]]],
    dtype=np.float32)

  blobs['img_id'] = roidb[0]['img_id']

  return blobs

def gasuss_noise(image,mean=0,var=0.01):
    '''
    手动添加高斯噪声
    mean : 均值
    var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 正态分布
    out = image + noise
    # if out.min() < 0:
    #     low_clip = -1.
    # else:
    #     low_clip = 0.
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    return out

# def add_fog(img):
#     img_f = img / 255.0
#     (row, col, chs) = img.shape

#     A = 0.3  
#     beta = 0.1  
#     size = math.sqrt(max(row, col))  
#     center = (row // 2, col // 2) 
#     for j in range(row):
#         for l in range(col):
#             d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
#             td = math.exp(-beta * d)
#             img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
#     return img_f * 255

###### add data augment
def _hue_kaist(image,image_lwir, min=0.75, max=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_lwir = cv2.cvtColor(image_lwir, cv2.COLOR_RGB2HSV)
    # x1=hsv[:,:,0]#0-179
    random_br = np.random.uniform(min, max)
    mask = hsv[:,:,0] * random_br > 179
    mask_lwir = hsv_lwir[:,:,0] * random_br> 179
    v_channel = np.where(mask, 179, hsv[:,:,0] * random_br)
    v_channel_lwir = np.where(mask_lwir, 179, hsv_lwir[:,:,0] * random_br)
    hsv[:,:,0] = v_channel
    hsv_lwir[:,:,0] = v_channel_lwir

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), cv2.cvtColor(hsv_lwir, cv2.COLOR_HSV2RGB)

def _saturation_kaist(image,image_lwir, min=0.75, max=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_lwir = cv2.cvtColor(image_lwir, cv2.COLOR_RGB2HSV)
    # x1 = hsv[:, :, 1]#0-255
    random_br = np.random.uniform(min, max)
    mask = hsv[:,:,1] * random_br > 255
    mask_lwir = hsv_lwir[:,:,1] * random_br> 255
    v_channel = np.where(mask, 255, hsv[:,:,1] * random_br)
    v_channel_lwir = np.where(mask_lwir, 255, hsv_lwir[:,:,1] * random_br)
    hsv[:,:,1] = v_channel
    hsv_lwir[:,:,1] = v_channel_lwir

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), cv2.cvtColor(hsv_lwir, cv2.COLOR_HSV2RGB)

def _brightness_kaist(image,image_lwir, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.
    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    hsv_lwir = cv2.cvtColor(image_lwir, cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min,max)
    mask = hsv[:,:,2] * random_br > 255
    mask_lwir = hsv_lwir[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    v_channel_lwir = np.where(mask_lwir, 255, hsv_lwir[:,:,2] * random_br)
    hsv[:,:,2] = v_channel
    hsv_lwir[:,:,2] = v_channel_lwir
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB),cv2.cvtColor(hsv_lwir,cv2.COLOR_HSV2RGB)

#################



def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  assert isinstance(cfg.SHIFT_X, int) and isinstance(cfg.SHIFT_X, int), \
         'wrong shift number, please check'

  for i in range(num_images):
    im = []
    # the reference and sensed modality
    for j in range(2):
      imgs = imageio.imread(roidb[i]['image'][j])

      if len(imgs.shape) == 2:
        imgs = np.stack([imgs]*3, 2)

      if imgs.shape[2] == 4: 
        print(roidb[i]['image'][j])

      im.append(imgs[:,:,:3])

      if len(im[j].shape) == 2: 
        im[j] = im[j][:,:,np.newaxis]
        im[j] = np.concatenate((im[j],im[j],im[j]), axis=2)
      # flip the channel, since the original one using cv2
      # rgb -> bgr
      im[j] = im[j][:,:,::-1]

      if j==1 and (cfg.SHIFT_X!=0 or cfg.SHIFT_Y!=0):
        new_img = np.zeros(im[j].shape)
        if cfg.SHIFT_X>0:
          if cfg.SHIFT_Y>0:
            new_img[:-cfg.SHIFT_Y,cfg.SHIFT_X:,:] = im[j][cfg.SHIFT_Y:,:-cfg.SHIFT_X,:]
          elif cfg.SHIFT_Y<0:
            new_img[-cfg.SHIFT_Y:,cfg.SHIFT_X:,:] = im[j][:cfg.SHIFT_Y,:-cfg.SHIFT_X,:]
          else:
            new_img[:,cfg.SHIFT_X:,:] = im[j][:,:-cfg.SHIFT_X,:]
        elif cfg.SHIFT_X<0:
          if cfg.SHIFT_Y>0:
            new_img[:-cfg.SHIFT_Y,:cfg.SHIFT_X,:] = im[j][cfg.SHIFT_Y:,-cfg.SHIFT_X:,:]
          elif cfg.SHIFT_Y<0:
            new_img[-cfg.SHIFT_Y:,:cfg.SHIFT_X,:] = im[j][:cfg.SHIFT_Y,-cfg.SHIFT_X:,:]
          else:
            new_img[:,:cfg.SHIFT_X,:] = im[j][:,-cfg.SHIFT_X:,:]
        else:
          if cfg.SHIFT_Y>0:
            new_img[:-cfg.SHIFT_Y,:,:] = im[j][cfg.SHIFT_Y:,:,:]
          elif cfg.SHIFT_Y<0:
            new_img[-cfg.SHIFT_Y:,:,:] = im[j][:cfg.SHIFT_Y,:,:] 
          else:
            pass
        im[j] = new_img

      if roidb[i]['flipped']:
        im[j] = im[j][:, ::-1, :]

    if cfg.aug:
      #print("----add aug---")
      img_lwir, img_rgb = im[0], im[1]
      if np.random.randint(0, 2) == 0:
          img_rgb,img_lwir = _brightness_kaist(img_rgb,img_lwir, min=0.5, max=2)
      #5.10
      # random hue
      if  np.random.randint(0, 2) == 0:
          img_rgb, img_lwir = _hue_kaist(img_rgb, img_lwir)
      # random saturation
      if np.random.randint(0, 2) == 0:
          img_rgb, img_lwir = _saturation_kaist(img_rgb, img_lwir)
      im[0], im[1] = img_lwir, img_rgb
    
    if im[0].shape != im[1].shape:
      im[0] = cv2.resize(im[0], (im[1].shape[1], im[1].shape[0]))
    

    for j in range(2):
      target_size = cfg.TRAIN.SCALES[scale_inds[i]]
      im[j], im_scale = prep_im_for_blob(im[j], cfg.PIXEL_MEANS, target_size,
                        cfg.TRAIN.MAX_SIZE)
    
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
