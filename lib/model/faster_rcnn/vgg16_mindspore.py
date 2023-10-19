# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Image classifiation.
"""
import math
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer

class KaimingInit(init.Initializer):
    r"""
    Base Class. Initialize the array with He kaiming algorithm.

    Args:
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function, recommended to use only with
            ``'relu'`` or ``'leaky_relu'`` (default).
    """
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingInit, self).__init__()
        self.mode = mode
        self.gain = _calculate_gain(nonlinearity, a)
    def _initialize(self, arr):
        pass

class KaimingNormal(KaimingInit):
    r"""
    Initialize the array with He kaiming normal algorithm. The resulting tensor will
    have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Input:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.

    Examples:
        >>> w = np.empty(3, 5)
        >>> KaimingNormal(w, mode='fan_out', nonlinearity='relu')
    """

    def _initialize(self, arr):
        fan = _select_fan(arr, self.mode)
        std = self.gain / math.sqrt(fan)
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)


def default_recurisive_init(custom_cell):
    """default_recurisive_init"""
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                  cell.weight.shape,
                                                  cell.weight.dtype))
            if cell.bias is not None:
                fan_in, _ = _calculate_in_and_out(cell.weight)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(init.initializer(init.Uniform(bound),
                                                    cell.bias.shape,
                                                    cell.bias.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                  cell.weight.shape,
                                                  cell.weight.dtype))
            if cell.bias is not None:
                fan_in, _ = _calculate_in_and_out(cell.weight)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(init.initializer(init.Uniform(bound),
                                                    cell.bias.shape,
                                                    cell.bias.dtype))
        elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
            pass


def _make_layer(base, args, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight = 'ones'
            if args.initialize_mode == "XavierUniform":
                weight_shape = (v, in_channels, 3, 3)
                weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32)#.to_tensor()

            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=args.padding,
                               pad_mode=args.pad_mode,
                               has_bias=args.has_bias,
                               weight_init=weight)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg(nn.Cell):
    """
    VGG network definition.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        num_classes (int): Class numbers. Default: 1000.
        batch_norm (bool): Whether to do the batchnorm. Default: False.
        batch_size (int): Batch size. Default: 1.

    Returns:
        Tensor, infer output tensor.

    Examples:
        >>> Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        >>>     num_classes=1000, batch_norm=False, batch_size=1)
    """

    def __init__(self, base, num_classes=1000, batch_norm=False, batch_size=1, args=None, phase="train"):
        super(Vgg, self).__init__()
        _ = batch_size
        self.layers = _make_layer(base, args, batch_norm=batch_norm)
        self.flatten = nn.Flatten()
        dropout_ratio = 0.5
        if not args.has_dropout or phase == "test":
            dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, num_classes)])
        if args.initialize_mode == "KaimingNormal":
            default_recurisive_init(self)
            self.custom_init_weight()

    def construct(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def custom_init_weight(self):
        """
        Init the weight of Conv2d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(
                    KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(
                    init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))


cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

from easydict import EasyDict as edict

# config for vgg16, cifar10
cifar_cfg = edict({
    "num_classes": 10,
    "lr": 0.01,
    "lr_init": 0.01,
    "lr_max": 0.1,
    "lr_epochs": '30,60,90,120',
    "lr_scheduler": "step",
    "warmup_epochs": 5,
    "batch_size": 64,
    "max_epoch": 70,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "loss_scale": 1.0,
    "label_smooth": 0,
    "label_smooth_factor": 0,
    "buffer_size": 10,
    "image_size": '224,224',
    "pad_mode": 'same',
    "padding": 0,
    "has_bias": True,
    "batch_norm": True,
    "keep_checkpoint_max": 10,
    "initialize_mode": "XavierUniform",
    "has_dropout": False
})


def vgg16_mindspore(num_classes=1000, args=None, phase="train"):
    """
    Get Vgg16 neural network with batch normalization.

    Args:
        num_classes (int): Class numbers. Default: 1000.
        args(namespace): param for net init.
        phase(str): train or test mode.

    Returns:
        Cell, cell instance of Vgg16 neural network with batch normalization.

    Examples:
        >>> vgg16(num_classes=1000, args=args)
    """
    if args is None:
        args = cifar_cfg
    net = Vgg(cfg['16'], num_classes=num_classes, args=args, batch_norm=False, phase=phase)
    return net