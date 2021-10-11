import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU, Tanh, Sigmoid
from torch.nn import Dropout
import math
import IPython
import random
import torch
import torch.autograd as A
import torch.nn.functional as F
import numpy as np
import csv
import pickle

import util.linalg as utils_la
import util.mytorch as mytorch

from models import resnet_conv
from models import resnet_VNECT_sep2D

from util import linalg

import matplotlib
from matplotlib import pyplot as plt

import src.models.unet_utils as utils_unet
import src.models.unet_bg_utils as utils_unet_bg
import src.models.dilated_unet as dilated_unet_bg

from mpl_toolkits.mplot3d import Axes3D

import sys, os, shutil

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


class LeastSquares:
    def __init__(self):
        pass

    def lstq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        # Assuming A to be full column rank
        cols = A.shape[1]
        if cols == torch.matrix_rank(A):
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
            Y_dash = A.permute(1, 0) @ Y
            x = self.lstq(A_dash, Y_dash)
        return x

class unet_background(nn.Module):
    def __init__(self, feature_scale=8, n_classes=3, is_deconv=True, in_channels=3, is_batchnorm=True, nb_stage=1):
        super(unet_background, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.nb_stage = nb_stage

        filters = [64, 128, 256, 512, 1024]
        # filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        for ns in range(self.nb_stage):
            setattr(self, 'conv_1_stage' + str(ns),
                    utils_unet_bg.unetConv2(self.in_channels, filters[0], self.is_batchnorm, padding=1))
            setattr(self, 'pool_1_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_2_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[0], filters[1], self.is_batchnorm, padding=1))
            setattr(self, 'pool_2_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_3_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[1], filters[2], self.is_batchnorm, padding=1))
            setattr(self, 'pool_3_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_4_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[2], filters[3], self.is_batchnorm, padding=1))
            setattr(self, 'pool_4_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_5_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[3], filters[4], self.is_batchnorm, padding=1))

            setattr(self, 'upconv_1_stage' + str(ns), utils_unet_bg.unetUp(filters[4], filters[3], self.is_deconv, 1))
            setattr(self, 'upconv_2_stage' + str(ns), utils_unet_bg.unetUp(filters[3], filters[2], self.is_deconv, 1))
            setattr(self, 'upconv_3_stage' + str(ns), utils_unet_bg.unetUp(filters[2], filters[1], self.is_deconv, 1))
            setattr(self, 'upconv_4_stage' + str(ns), utils_unet_bg.unetUp(filters[1], filters[0], self.is_deconv, 1))

            setattr(self, 'final_stage' + str(ns), nn.Conv2d(filters[0], n_classes, 1))

        self.relu = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=False)
        self.dropout = Dropout(inplace=True, p=0.3)

    def forward(self, inputs):

        for ns in range(self.nb_stage):
            conv1 = getattr(self, 'conv_1_stage' + str(ns))(inputs)
            maxpool1 = getattr(self, 'pool_1_stage' + str(ns))(conv1)
            conv2 = getattr(self, 'conv_2_stage' + str(ns))(maxpool1)
            maxpool2 = getattr(self, 'pool_2_stage' + str(ns))(conv2)
            conv3 = getattr(self, 'conv_3_stage' + str(ns))(maxpool2)
            maxpool3 = getattr(self, 'pool_3_stage' + str(ns))(conv3)
            conv4 = getattr(self, 'conv_4_stage' + str(ns))(maxpool3)
            maxpool4 = getattr(self, 'pool_4_stage' + str(ns))(conv4)
            center = getattr(self, 'conv_5_stage' + str(ns))(maxpool4)
            up4 = getattr(self, 'upconv_1_stage' + str(ns))(conv4, center)
            up3 = getattr(self, 'upconv_2_stage' + str(ns))(conv3, up4)
            up2 = getattr(self, 'upconv_3_stage' + str(ns))(conv2, up3)
            up1 = getattr(self, 'upconv_4_stage' + str(ns))(conv1, up2)
            output = getattr(self, 'final_stage' + str(ns))(up1)
        return output


class unet_background_medium(nn.Module):
    def __init__(self, feature_scale=8, n_classes=3, is_deconv=True, in_channels=3, is_batchnorm=True, nb_stage=1):
        super(unet_background_medium, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.nb_stage = nb_stage

        filters = [64, 128, 256, 512, 1024, 2048]
        filters = [int(x / self.feature_scale) for x in filters]

        for ns in range(self.nb_stage):
            setattr(self, 'conv_1_stage' + str(ns),
                    utils_unet_bg.unetConv2(self.in_channels, filters[0], self.is_batchnorm, padding=1))
            setattr(self, 'pool_1_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_2_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[0], filters[1], self.is_batchnorm, padding=1))
            setattr(self, 'pool_2_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_3_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[1], filters[2], self.is_batchnorm, padding=1))
            setattr(self, 'pool_3_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_4_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[2], filters[3], self.is_batchnorm, padding=1))
            setattr(self, 'pool_4_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_5_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[3], filters[4], self.is_batchnorm, padding=1))
            setattr(self, 'pool_5_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_6_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[4], filters[5], self.is_batchnorm, padding=1))

            setattr(self, 'upconv_1_stage' + str(ns), utils_unet_bg.unetUp(filters[5], filters[4], self.is_deconv, 1))
            setattr(self, 'upconv_2_stage' + str(ns), utils_unet_bg.unetUp(filters[4], filters[3], self.is_deconv, 1))
            setattr(self, 'upconv_3_stage' + str(ns), utils_unet_bg.unetUp(filters[3], filters[2], self.is_deconv, 1))
            setattr(self, 'upconv_4_stage' + str(ns), utils_unet_bg.unetUp(filters[2], filters[1], self.is_deconv, 1))
            setattr(self, 'upconv_5_stage' + str(ns), utils_unet_bg.unetUp(filters[1], filters[0], self.is_deconv, 1))

            setattr(self, 'final_stage' + str(ns), nn.Conv2d(filters[0], n_classes, 1))

        self.relu = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=False)
        self.dropout = Dropout(inplace=True, p=0.3)

    def forward(self, inputs):

        for ns in range(self.nb_stage):
            conv1 = getattr(self, 'conv_1_stage' + str(ns))(inputs)
            maxpool1 = getattr(self, 'pool_1_stage' + str(ns))(conv1)
            conv2 = getattr(self, 'conv_2_stage' + str(ns))(maxpool1)
            maxpool2 = getattr(self, 'pool_2_stage' + str(ns))(conv2)
            conv3 = getattr(self, 'conv_3_stage' + str(ns))(maxpool2)
            maxpool3 = getattr(self, 'pool_3_stage' + str(ns))(conv3)
            conv4 = getattr(self, 'conv_4_stage' + str(ns))(maxpool3)
            maxpool4 = getattr(self, 'pool_4_stage' + str(ns))(conv4)
            conv5 = getattr(self, 'conv_5_stage' + str(ns))(maxpool4)
            maxpool5 = getattr(self, 'pool_5_stage' + str(ns))(conv5)

            center = getattr(self, 'conv_6_stage' + str(ns))(maxpool5)

            up5 = getattr(self, 'upconv_1_stage' + str(ns))(conv5, center)
            up4 = getattr(self, 'upconv_2_stage' + str(ns))(conv4, up5)
            up3 = getattr(self, 'upconv_3_stage' + str(ns))(conv3, up4)
            up2 = getattr(self, 'upconv_4_stage' + str(ns))(conv2, up3)
            up1 = getattr(self, 'upconv_5_stage' + str(ns))(conv1, up2)

            output = getattr(self, 'final_stage' + str(ns))(up1)
        return output


class unet_background_large(nn.Module):
    def __init__(self, feature_scale=8, n_classes=3, is_deconv=True, in_channels=3, is_batchnorm=True, nb_stage=1):
        super(unet_background_large, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.nb_stage = nb_stage

        filters = [64, 128, 256, 512, 1024, 2048, 4096]
        # filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # filters = [12, 24, 48, 96, 192, 384, 768]

        for ns in range(self.nb_stage):
            setattr(self, 'conv_1_stage' + str(ns),
                    utils_unet_bg.unetConv2(self.in_channels, filters[0], self.is_batchnorm, padding=1))
            setattr(self, 'pool_1_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_2_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[0], filters[1], self.is_batchnorm, padding=1))
            setattr(self, 'pool_2_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_3_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[1], filters[2], self.is_batchnorm, padding=1))
            setattr(self, 'pool_3_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_4_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[2], filters[3], self.is_batchnorm, padding=1))
            setattr(self, 'pool_4_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_5_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[3], filters[4], self.is_batchnorm, padding=1))
            setattr(self, 'pool_5_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_6_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[4], filters[5], self.is_batchnorm, padding=1))
            setattr(self, 'pool_6_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_7_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[5], filters[6], self.is_batchnorm, padding=1))

            setattr(self, 'upconv_1_stage' + str(ns), utils_unet_bg.unetUp(filters[6], filters[5], self.is_deconv, 1))
            setattr(self, 'upconv_2_stage' + str(ns), utils_unet_bg.unetUp(filters[5], filters[4], self.is_deconv, 1))
            setattr(self, 'upconv_3_stage' + str(ns), utils_unet_bg.unetUp(filters[4], filters[3], self.is_deconv, 1))
            setattr(self, 'upconv_4_stage' + str(ns), utils_unet_bg.unetUp(filters[3], filters[2], self.is_deconv, 1))
            setattr(self, 'upconv_5_stage' + str(ns), utils_unet_bg.unetUp(filters[2], filters[1], self.is_deconv, 1))
            setattr(self, 'upconv_6_stage' + str(ns), utils_unet_bg.unetUp(filters[1], filters[0], self.is_deconv, 1))

            setattr(self, 'final_stage' + str(ns), nn.Conv2d(filters[0], n_classes, 1))

        self.relu = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=False)
        self.dropout = Dropout(inplace=True, p=0.3)

    def forward(self, inputs):

        for ns in range(self.nb_stage):
            conv1 = getattr(self, 'conv_1_stage' + str(ns))(inputs)
            maxpool1 = getattr(self, 'pool_1_stage' + str(ns))(conv1)
            conv2 = getattr(self, 'conv_2_stage' + str(ns))(maxpool1)
            maxpool2 = getattr(self, 'pool_2_stage' + str(ns))(conv2)
            conv3 = getattr(self, 'conv_3_stage' + str(ns))(maxpool2)
            maxpool3 = getattr(self, 'pool_3_stage' + str(ns))(conv3)
            conv4 = getattr(self, 'conv_4_stage' + str(ns))(maxpool3)
            maxpool4 = getattr(self, 'pool_4_stage' + str(ns))(conv4)
            conv5 = getattr(self, 'conv_5_stage' + str(ns))(maxpool4)
            maxpool5 = getattr(self, 'pool_5_stage' + str(ns))(conv5)
            conv6 = getattr(self, 'conv_6_stage' + str(ns))(maxpool5)
            maxpool6 = getattr(self, 'pool_6_stage' + str(ns))(conv6)

            center = getattr(self, 'conv_7_stage' + str(ns))(maxpool6)

            up6 = getattr(self, 'upconv_1_stage' + str(ns))(conv6, center)
            up5 = getattr(self, 'upconv_2_stage' + str(ns))(conv5, up6)
            up4 = getattr(self, 'upconv_3_stage' + str(ns))(conv4, up5)
            up3 = getattr(self, 'upconv_4_stage' + str(ns))(conv3, up4)
            up2 = getattr(self, 'upconv_5_stage' + str(ns))(conv2, up3)
            up1 = getattr(self, 'upconv_6_stage' + str(ns))(conv1, up2)

            output = getattr(self, 'final_stage' + str(ns))(up1)
        return output


class unet_background_larger_receptive(nn.Module):
    def __init__(self, feature_scale=8, n_classes=3, is_deconv=True, in_channels=3, is_batchnorm=True, nb_stage=1):
        super(unet_background_larger_receptive, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.nb_stage = nb_stage

        filters = [64, 128, 256, 512, 1024]
        # filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        for ns in range(self.nb_stage):
            setattr(self, 'conv_1_stage' + str(ns),
                    utils_unet_bg.unetConv2(self.in_channels, filters[0], self.is_batchnorm, padding=1))
            setattr(self, 'pool_1_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_2_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[0], filters[1], self.is_batchnorm, padding=1))
            setattr(self, 'pool_2_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_3_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[1], filters[2], self.is_batchnorm, padding=1))
            setattr(self, 'pool_3_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_4_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[2], filters[3], self.is_batchnorm, padding=1))
            setattr(self, 'pool_4_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            setattr(self, 'conv_5_stage' + str(ns),
                    utils_unet_bg.unetConv2(filters[3], filters[4], self.is_batchnorm, padding=1))

            setattr(self, 'upconv_1_stage' + str(ns), utils_unet_bg.unetUp(filters[4], filters[3], self.is_deconv, 1))
            setattr(self, 'upconv_2_stage' + str(ns), utils_unet_bg.unetUp(filters[3], filters[2], self.is_deconv, 1))
            setattr(self, 'upconv_3_stage' + str(ns), utils_unet_bg.unetUp(filters[2], filters[1], self.is_deconv, 1))
            setattr(self, 'upconv_4_stage' + str(ns), utils_unet_bg.unetUp(filters[1], filters[0], self.is_deconv, 1))

            setattr(self, 'final_stage' + str(ns), nn.Conv2d(filters[0], n_classes, 1))

        self.relu = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=False)
        self.dropout = Dropout(inplace=True, p=0.3)

    def forward(self, inputs):

        for ns in range(self.nb_stage):
            conv1 = getattr(self, 'conv_1_stage' + str(ns))(inputs)
            maxpool1 = getattr(self, 'pool_1_stage' + str(ns))(conv1)
            conv2 = getattr(self, 'conv_2_stage' + str(ns))(maxpool1)
            maxpool2 = getattr(self, 'pool_2_stage' + str(ns))(conv2)
            conv3 = getattr(self, 'conv_3_stage' + str(ns))(maxpool2)
            maxpool3 = getattr(self, 'pool_3_stage' + str(ns))(conv3)
            conv4 = getattr(self, 'conv_4_stage' + str(ns))(maxpool3)
            maxpool4 = getattr(self, 'pool_4_stage' + str(ns))(conv4)
            center = getattr(self, 'conv_5_stage' + str(ns))(maxpool4)

            center_max, _ = torch.max(center, dim=3, keepdim=True)
            center_max, _ = torch.max(center_max, dim=2, keepdim=True)
            center_expanded = center_max.expand_as(center)

            up4 = getattr(self, 'upconv_1_stage' + str(ns))(conv4, center_expanded)
            up3 = getattr(self, 'upconv_2_stage' + str(ns))(conv3, up4)
            up2 = getattr(self, 'upconv_3_stage' + str(ns))(conv2, up3)
            up1 = getattr(self, 'upconv_4_stage' + str(ns))(conv1, up2)
            output = getattr(self, 'final_stage' + str(ns))(up1)
        return output


class unet_encoder(nn.Module):
    def __init__(self, input_key, bottlneck_feature_dim, num_encoding_layers):
        super(unet_encoder, self).__init__()
        self.input_key = input_key

        # filters = [64, 128, 256, 512, 1024]
        self.filters = [64, 128, 256, 512, 512, 512]
        self.feature_scale = self.filters[-1] / bottlneck_feature_dim
        self.filters = [int(x / self.feature_scale) for x in self.filters]

        setattr(self, 'conv_1_stage',
                unetConv2(self.in_channels, self.filters[0], self.is_batchnorm, padding=1))
        setattr(self, 'pool_1_stage', nn.MaxPool2d(kernel_size=2))
        for li in range(2,
                        num_encoding_layers):  # note, first layer(li==1) is already created, last layer(li==num_encoding_layers) is created externally
            setattr(self, 'conv_' + str(li) + '_stage',
                    unetConv2(self.filters[li - 2], self.filters[li - 1], self.is_batchnorm, padding=1))
            setattr(self, 'pool_' + str(li) + '_stage', nn.MaxPool2d(kernel_size=2))

        if from_latent_hidden_layers:
            setattr(self, 'conv_' + str(num_encoding_layers) + '_stage', nn.Sequential(
                unetConv2(self.filters[num_encoding_layers - 2], self.filters[num_encoding_layers - 1],
                          self.is_batchnorm, padding=1),
                nn.MaxPool2d(kernel_size=2)
            ))
        else:
            setattr(self, 'conv_' + str(num_encoding_layers) + '_stage',
                    unetConv2(self.filters[num_encoding_layers - 2], self.filters[num_encoding_layers - 1],
                              self.is_batchnorm, padding=1))

    def forward(self, input_dict):
        out_enc_conv = input_dict['img_crop']
        for li in range(1,
                        self.num_encoding_layers):  # note, first layer(li==1) is already created, last layer(li==num_encoding_layers) is created externally
            out_enc_conv = getattr(self, 'conv_' + str(li) + '_stage')(out_enc_conv)
            out_enc_conv = getattr(self, 'pool_' + str(li) + '_stage')(out_enc_conv)
        out_enc_conv = getattr(self, 'conv_' + str(self.num_encoding_layers) + '_stage')(out_enc_conv)

        # fully-connected
        # broken look up in V1 TODO: !
        center_flat = out_enc_conv.view(batch_size * ST_size, -1)
        if has_fg:
            latent_fg = self.to_fg(center_flat)
            latent_fg = ST_split(latent_fg)


class unet_decoder(nn.Module):
    def __init__(self, bottlneck_feature_dim, num_encoding_layers, output_channels, is_deconv):
        super(unet_decoder, self).__init__()
        self.num_encoding_layers = num_encoding_layers
        self.is_deconv = is_deconv

        self.filters = [64, 128, 256, 512, 512, 512]

        self.feature_scale = self.filters[-1] // bottlneck_feature_dim  # put back
        assert self.feature_scale == self.filters[-1] / bottlneck_feature_dim  # integer division?
        self.filters = [x // self.feature_scale for x in self.filters]

        upper_conv = self.is_deconv and not upper_billinear
        lower_conv = self.is_deconv and not lower_billinear

        for li in range(1, num_encoding_layers - 1):
            setattr(self, 'upconv_' + str(li) + '_stage',
                    utils_unet.unetUpNoSKip(self.filters[num_encoding_layers - li],
                                            self.filters[num_encoding_layers - li - 1],
                                            upper_conv, padding=1, batch_norm=False))

        setattr(self, 'upconv_' + str(num_encoding_layers - 1) + '_stage',
                utils_unet.unetUpNoSKip(self.filters[1], self.filters[0], lower_conv, padding=1, batch_norm=False))

        setattr(self, 'final_stage', nn.Conv2d(self.filters[0], output_channels, 1))

        self.relu = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=False)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        self.dropout = Dropout(inplace=True, p=0.3)

        self.reset_params()

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            utils_unet.weight_init(m)

    def forward(self, x):
        out_deconv = x

        for li in range(1, self.num_encoding_layers - 1):
            out_deconv = getattr(self, 'upconv_' + str(li) + '_stage')(out_deconv)

        out_deconv = getattr(self, 'upconv_' + str(self.num_encoding_layers - 1) + '_stage')(
            out_deconv)
        out_before_activation = getattr(self, 'final_stage')(out_deconv)
        return getattr(self, 'final_stage')(out_deconv)


def construct_bump_function(in_resolution, type):
    if type == True:
        return None
    xs = torch.linspace(-1, 1, in_resolution, dtype=torch.float)
    xs = xs.view(1, -1).expand(in_resolution, in_resolution)
    ys = torch.linspace(-1, 1, in_resolution, dtype=torch.float)
    ys = ys.view(-1, 1).expand(in_resolution, in_resolution)
    if type == 'Gauss':
        bump_function = torch.exp(- 2 * (xs ** 2 + ys ** 2))  # Gaussian with std=0.5
    elif type == 'GaussB':
        bump_function = torch.exp(
            1 - 1 / (1 - (xs ** 2 + ys ** 2)))  # classical bump function, rescaled to have maxima == 1
    elif type == 'GaussBsq':
        bump_function = torch.exp(1 - 1 / (1 - (xs ** 4 + ys ** 4)))  # sharper version with larger plateau
    elif type == 'GaussBSqSqr':
        bump_function = torch.exp(1 - 1 / (1 - np.sqrt(xs ** 4 + ys ** 4)))  # sharper version with larger plateau
    else:
        bump_function = torch.ones(xs.shape)

    for r in range(in_resolution):
        for c in range(in_resolution):
            if type == 'Gauss':
                continue
            if type == 'GaussB':
                put_zero = (xs[r, c] ** 2 + ys[r, c] ** 2) >= 1
            if type == 'GaussBsq':
                put_zero = (xs[r, c] ** 4 + ys[r, c] ** 4) >= 1
            if type == 'GaussBSqSqr':
                put_zero = np.sqrt(xs[r, c] ** 4 + ys[r, c] ** 4) >= 1
            if put_zero:
                bump_function[r, c] = 0
    return bump_function.cuda()


class unet(nn.Module):
    bump_function = None  # static variable

    def __init__(self, feature_scale=4,  # to reduce dimensionality
                 in_resolution=256,
                 output_channels=3, is_deconv=True,
                 upper_billinear=False,
                 lower_billinear=False,
                 in_channels=3, is_batchnorm=True,
                 skip_background=True,
                 estimate_background=True,
                 reconstruct_type='full',
                 receptive_field='small',
                 bbox_random=False,
                 bg_recursion=1,
                 num_joints=17, nb_dims=3,  # ecoding transformation
                 encoderType='UNet',
                 num_encoding_layers=5,
                 dimension_bg=256,
                 dimension_fg=256,
                 dimension_3d=3 * 64,  # needs to be devidable by 3
                 latent_dropout=0.3,
                 shuffle_fg=True,
                 shuffle_3d=True,
                 shuffle_prob=0.5,
                 from_latent_hidden_layers=0,
                 n_hidden_to3Dpose=2,
                 subbatch_size=4,
                 params_per_box=4,
                 offset_consistency_type='loss',
                 implicit_rotation=False,
                 predict_rotation=False,
                 spatial_transformer=False,
                 spatial_transformer_num=1,
                 spatial_transformer_bounds=1,
                 choose_cell='Random',
                 only_center_grid='False',
                 offset_range=1.0,
                 masked_blending=True,
                 scale_mask_max_to_1=True,
                 nb_stage=1,  # number of U-net stacks
                 output_types=['3D', 'img', 'shuffled_pose', 'shuffled_appearance'],
                 num_cameras=4,
                 predict_transformer_depth=False,
                 pass_transformer_depth=False,
                 normalize_mask_density=False,
                 match_crops=False,
                 shuffle_crops=False,
                 offset_crop=False,
                 mode='NVS',
                 transductive_training=[],
                 similarity_bandwidth=10,
                 disable_detector=False,
                 volume_size=64,
                 cuboid_side=2,
                 img_mean=None,
                 img_std=None,
                 ):

        super(unet, self).__init__()
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.nb_stage = nb_stage
        self.dimension_bg = dimension_bg
        self.dimension_fg = dimension_fg
        self.dimension_3d = dimension_3d
        self.shuffle_fg = shuffle_fg
        self.shuffle_3d = shuffle_3d
        self.shuffle_prob = shuffle_prob
        self.num_encoding_layers = num_encoding_layers
        self.output_types = output_types
        self.encoderType = encoderType
        assert dimension_3d % 3 == 0
        self.implicit_rotation = implicit_rotation
        self.params_per_box = params_per_box
        self.offset_consistency_type = offset_consistency_type # multicam offset consistency
        self.predict_rotation = predict_rotation
        self.num_cameras = num_cameras
        self.predict_transformer_depth = predict_transformer_depth
        self.pass_transformer_depth = pass_transformer_depth
        self.spatial_transformer_bounds = spatial_transformer_bounds
        self.normalize_mask_density = normalize_mask_density
        self.match_crops = match_crops and spatial_transformer_num > 1
        self.shuffle_crops = shuffle_crops
        self.offset_crop = offset_crop
        self.mode = mode
        self.transductive_training = transductive_training
        self.similarity_bandwidth = similarity_bandwidth
        self.disable_detector = disable_detector
        self.estimate_background = estimate_background
        self.reconstruct_type = reconstruct_type
        self.receptive_field = receptive_field
        self.bbox_random = bbox_random
        self.bg_recursion = bg_recursion
        self.choose_cell = choose_cell
        self.only_center_grid = only_center_grid
        self.offset_range = offset_range
        self.img_mean = img_mean
        self.img_std = img_std
        self.itr_check = 0

        #create least squares
        self.ls = LeastSquares()

        # volumetric parameters
        self.volume_size = volume_size
        self.cuboid_side = cuboid_side
        self.xxx, self.yyy, self.zzz = np.meshgrid(np.arange(self.volume_size),
                                                   np.arange(self.volume_size),
                                                   np.arange(self.volume_size))

        self.xxx = torch.from_numpy(self.xxx).cuda()
        self.yyy = torch.from_numpy(self.yyy).cuda()
        self.zzz = torch.from_numpy(self.zzz).cuda()

        self.xxx_visu, self.yyy_visu, self.zzz_visu = np.meshgrid(np.arange(self.volume_size + 1),
                                                                  np.arange(self.volume_size + 1),
                                                                  np.arange(self.volume_size + 1))

        self.xxx_visu = torch.from_numpy(self.xxx_visu).cuda()
        self.yyy_visu = torch.from_numpy(self.yyy_visu).cuda()
        self.zzz_visu = torch.from_numpy(self.zzz_visu).cuda()

        assert not skip_background

        self.subbatch_size = subbatch_size
        self.latent_dropout = latent_dropout
        self.masked_blending = masked_blending

        self.bottlneck_feature_dim = 512 // feature_scale
        self.bottleneck_resolution = in_resolution // (2 ** (num_encoding_layers - 1))
        num_bottlneck_features = self.bottleneck_resolution ** 2 * self.bottlneck_feature_dim
        print('bottleneck_resolution', self.bottleneck_resolution, 'num_bottlneck_features', num_bottlneck_features)

        self.spatial_transformer = spatial_transformer
        self.ST_size = spatial_transformer_num
        ST_size = self.ST_size
        self.scale_mask_max_to_1 = scale_mask_max_to_1

        self.itr = 0

        ################################################
        ############ Spatial transformer ###############
        if self.spatial_transformer:
            # params_per_transformer = 4# translation x, translation y, and scale x and y
            params_per_transformer = self.params_per_box
            # if self.predict_transformer_depth: Note, commented to always always predict depth, to allow weight transfer
            # params_per_transformer += 1
            affine_dimension = ST_size * params_per_transformer
            self.detection_resolution = self.in_resolution
            self.detector = resnet_conv.resnet18(num_classes=params_per_transformer, num_channels=3,
                                                 input_width=self.detection_resolution, nois_stddev=0,
                                                 output_key='affine')

            # construct the bump function
            unet.bump_function = construct_bump_function(self.in_resolution, type=self.spatial_transformer)

        ############################################################
        ############ Rotation prediction transformer ###############
        if self.predict_rotation:
            assert False

        ####################################
        ############ encoder ###############
        if self.encoderType == "ResNet":
            self.encoder = resnet_VNECT_sep2D.resnet50(pretrained=True, input_key='img_crop',
                                                       output_keys=['latent_3d', '2D_heat'],
                                                       input_width=in_resolution, net_type='high_res',
                                                       num_classes=self.dimension_fg + self.dimension_3d, path_unsup='../pretrained/lemniscate_resnet50.pth.tar')

        else:
            self.encoder = unet_encoder(input_key='img_crop', output_key='latent_3d',
                                        bottlneck_feature_dim=self.bottlneck_feature_dim)

        ####################################
        ####background encoder&decoder######
        if self.receptive_field == 'small':
            self.bg_unet = unet_background()
        elif self.receptive_field == 'medium':
            self.bg_unet = unet_background_medium(feature_scale=4)
        elif self.receptive_field == 'large':
            self.bg_unet = unet_background_large()
        elif self.receptive_field == 'dilate':
            self.bg_unet = dilated_unet_bg.DilatedUNet(in_channels=3, classes=3, depth=3,
                                                       first_channels=32, padding=1,
                                                       bottleneck_depth=6, bottleneck_type='cascade',
                                                       batch_norm=False, up_mode='deconv',
                                                       activation=nn.ReLU(inplace=True))

        ##################################################
        ############ latent transformation ###############

        assert self.dimension_fg < self.bottlneck_feature_dim
        num_bottlneck_features_3d = self.bottleneck_resolution ** 2 * (self.bottlneck_feature_dim - self.dimension_fg)

        self.to_3d = nn.Sequential(Linear(num_bottlneck_features, self.dimension_3d),
                                   Dropout(inplace=True, p=self.latent_dropout)  # removing dropout degrades results
                                   )

        if self.implicit_rotation:
            print("WARNING: doing implicit rotation!")
            rotation_encoding_dimension = 128
            self.encode_angle = nn.Sequential(Linear(3 * 3, rotation_encoding_dimension // 2),
                                              Dropout(inplace=True, p=self.latent_dropout),
                                              ReLU(inplace=False),
                                              Linear(rotation_encoding_dimension // 2, rotation_encoding_dimension),
                                              Dropout(inplace=True, p=self.latent_dropout),
                                              ReLU(inplace=False),
                                              Linear(rotation_encoding_dimension, rotation_encoding_dimension),
                                              )

            self.rotate_implicitely = nn.Sequential(
                Linear(self.dimension_3d + rotation_encoding_dimension, self.dimension_3d),
                Dropout(inplace=True, p=self.latent_dropout),
                ReLU(inplace=False))

        dimension_depth = 0
        if self.pass_transformer_depth:
            dimension_depth = 1
        if from_latent_hidden_layers:
            hidden_layer_dimension = 1024
            if self.dimension_fg > 0:
                self.to_fg = nn.Sequential(Linear(num_bottlneck_features, 256),
                                           Dropout(inplace=True, p=self.latent_dropout),
                                           ReLU(inplace=False),
                                           Linear(256, self.dimension_fg),
                                           Dropout(inplace=True, p=self.latent_dropout),
                                           ReLU(inplace=False))
            self.from_latent = nn.Sequential(Linear(self.dimension_3d + dimension_depth, hidden_layer_dimension),
                                             Dropout(inplace=True, p=self.latent_dropout),
                                             ReLU(inplace=False),
                                             Linear(hidden_layer_dimension, num_bottlneck_features_3d),
                                             Dropout(inplace=True, p=self.latent_dropout),
                                             ReLU(inplace=False))
        else:
            if self.dimension_fg > 0:
                self.to_fg = nn.Sequential(Linear(num_bottlneck_features, self.dimension_fg),
                                           Dropout(inplace=True, p=self.latent_dropout),
                                           ReLU(inplace=False))
            self.from_latent = nn.Sequential(Linear(self.dimension_3d + dimension_depth, num_bottlneck_features_3d),
                                             Dropout(inplace=True, p=self.latent_dropout),
                                             ReLU(inplace=False))

        ####################################
        ############ decoder ###############

        if self.masked_blending:
            output_channels_combined = output_channels + 1
        else:
            output_channels_combined = output_channels

        self.decoder = unet_decoder(bottlneck_feature_dim=self.bottlneck_feature_dim,
                                    num_encoding_layers=num_encoding_layers,
                                    output_channels=output_channels_combined,
                                    is_deconv=is_deconv)

    def roll_segment_random(self, list, start, end, prob=0.5):
        selected = list[start:end]
        if self.training:
            if np.random.random([1])[0] < prob:  # now 80%. 50% rotation worked well, by percentage of camera breaks..
                selected = np.roll(selected, 1).tolist()  # flip (in case of pairs)
        else:  # deterministic shuffling for testing
            selected = np.roll(selected, 1).tolist()
        list[start:end] = selected

    def UNUSED_shuffle_segment(self, list, start, end):
        selected = list[start:end]
        if self.training:
            random.shuffle(selected)  # in place shuffling
        else:  # deterministic rolling by 1 for testing
            selected = np.roll(selected, 1).tolist()
        list[start:end] = selected

    def flip_segment(self, list, start, width):
        selected = list[start:start + width]
        list[start:start + width] = list[start + width:start + 2 * width]
        list[start + width:start + 2 * width] = selected

    def distance(self, P0, P1):
        # generate all line direction vectors
        n = (P1 - P0) / torch.norm(P1 - P0, dim=1).unsqueeze(1)  # normalized

    def intersect(self, P0, P1):
        """P0 and P1 are NxD arrays defining N lines.
        D is the dimension of the space. This function 
        returns the least squares intersection of the N
        lines from the system given by eq. 13 in 
        http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
        """

        # generate all line direction vectors
        n = (P1 - P0) / torch.norm(P1 - P0, dim=1).unsqueeze(1)  # normalized

        # generate the array of all projectors
        projs = torch.eye(n.shape[1]).float().cuda() - n.unsqueeze(2) * n.unsqueeze(1)  # I - n*n.T

        # generate R matrix and q vector
        R = projs.sum(0)
        q = (projs @ P0.unsqueeze(2)).sum(0)

        # solve the least squares problem for the
        # intersection point p: Rp = q
        p = torch.lstsq(q, R)[0]

        return p

    def pairwise_distance(self, P0, P1, n_view):
        tensor_view = torch.arange(n_view).cuda()
        combinations_view = torch.combinations(tensor_view)
        distances = torch.zeros(combinations_view.shape[0]).cuda()
        d = (P1 - P0) / torch.norm(P1 - P0, dim=1).unsqueeze(1)  # normalized direction vectors of the cam group

        for comb in range(combinations_view.shape[0]):
            # generate all line direction vectors

            a1 = P0[combinations_view[comb][0],:] #point vector of the first line
            a2 = P0[combinations_view[comb][1],:] #point vector of the second line

            b1 = d[combinations_view[comb][0], :] #direction vector of the first line
            b2 = d[combinations_view[comb][1], :] #direction vector of the second line
            n = torch.cross(b1, b2)
            n_hat = n / torch.norm(n)
            distances[comb] = torch.abs(torch.dot((a1 - a2), n_hat))
        return torch.mean(distances)

    def intersect_custom_lstsq(self, P0, P1, ls):
        """P0 and P1 are NxD arrays defining N lines.
        D is the dimension of the space. This function 
        returns the least squares intersection of the N
        lines from the system given by eq. 13 in 
        http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
        """

        # generate all line direction vectors
        n1 = (P1 - P0) / torch.norm(P1 - P0, dim=1).unsqueeze(1)  # normalized

        # generate the array of all projectors
        projs = torch.eye(n1.shape[1]).float().cuda() - n1.unsqueeze(2) * n1.unsqueeze(1)  # I - n*n.T

        # generate R matrix and q vector
        R1 = projs.sum(0)
        q1 = (projs @ P0.unsqueeze(2)).sum(0)

        # solve the least squares problem for the
        # intersection point p: Rp = q
        p1 = ls.lstq(R1, q1)

        return p1

    def euclidean_to_homogeneous(self, points):
        return torch.cat([points, torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).float().cuda()], dim=0)

    def offset_consistency(self, input_dict, b_ind_r, b_ind_c, affine_matrix, grid_size,itr, subbatch_size, type='loss'):
        #type is 'loss' or 'projection'

        H_img = input_dict['img'].shape[-2]
        W_img = input_dict['img'].shape[-1]
        batch_size = input_dict['img'].shape[0]
        n_views = subbatch_size
        bbox_center_3d = torch.zeros(batch_size // subbatch_size, subbatch_size,  3).float().cuda()

        extrinsics = input_dict['camera_extrinsics'].view((batch_size, 3, 4)).float().cuda()
        cam_ext = torch.cat((extrinsics, torch.FloatTensor([0, 0, 0, 1]).cuda().view(1, -1).unsqueeze(0).repeat(batch_size, 1, 1)), dim=1)

        extrinsics_inverse = input_dict['inverse_camera_extrinsics'].view((batch_size, 4, 4)).float().cuda()
        intrinsics_inverse = input_dict['inverse_intrinsic'].view((batch_size, 3, 3)).float().cuda()
        intrinsics = input_dict['intrinsic'].view((batch_size, 3, 3)).float().cuda()
        cam_int = torch.cat((intrinsics, torch.zeros(batch_size, 3, 1).float().cuda()), dim=-1)

        cam_total = cam_int @ cam_ext

        x_scale = affine_matrix[:, 0]
        y_scale = affine_matrix[:, 1]
        box_center_norm_x = torch.clamp((affine_matrix[:, 2] + 1 / grid_size + ((2 / grid_size) * b_ind_c) - 1), min=-1, max=1)
        box_center_norm_y = torch.clamp((affine_matrix[:, 3] + 1 / grid_size + ((2 / grid_size) * b_ind_r) - 1), min=-1, max=1)
        bbox_center_x = W_img * ( ( box_center_norm_x + 1 - x_scale) / 2) + (x_scale * W_img) / 2
        bbox_center_y = H_img * ((box_center_norm_y + 1 - y_scale) / 2) + (y_scale * H_img) / 2

        cam_position = (torch.FloatTensor([0, 0, 0, 1])).cuda().unsqueeze(0).repeat(batch_size, 1).view(batch_size, 4, 1)  # N*4*1
        cam_position_w = torch.bmm(extrinsics_inverse, cam_position)[:, 0:3]


        bbox_centers = torch.cat((bbox_center_x.view(-1, 1), bbox_center_y.view(-1, 1), torch.ones(batch_size,1).cuda()), dim=1).view(-1, 3, 1)
        center_m = torch.bmm(intrinsics_inverse, bbox_centers)
        center_w = torch.bmm(extrinsics_inverse, torch.cat((center_m * 10, torch.empty(batch_size, 1, 1).fill_(1).cuda()), dim=1))[:, 0:3, :]
        bbox_centers = center_w.squeeze(-1)
        bbox_centers = bbox_centers.contiguous().view(-1, subbatch_size, bbox_centers.shape[-1])

        #compute the closest 3d point to the two lines for each cam group
        cam_position_w = cam_position_w.squeeze(-1)
        cam_position_w = cam_position_w.contiguous().view(-1, subbatch_size, cam_position_w.shape[-1])

        if type == 'projection':
            for cam_group in range(bbox_centers.shape[0]):
                base_point = self.intersect_custom_lstsq(cam_position_w[cam_group], bbox_centers[cam_group], self.ls)
                bbox_center_3d[cam_group, :, :] = base_point[:, 0]

            bbox_center_3d = bbox_center_3d.contiguous().view(batch_size, 3).unsqueeze(-1)
            bbox_center_3d = torch.cat((bbox_center_3d, torch.ones(batch_size,1,1).cuda()), dim=1)

            bbox_center_2d = torch.bmm(cam_total, bbox_center_3d)
            invalid_mask = bbox_center_2d[:, 2, :] <= 0
            if invalid_mask.sum() > 0:
                print("PROJECTED MEAN INVALID")

            bbox_center_2d_refined = (bbox_center_2d[:, :-1, :] / bbox_center_2d[:, 2, :].unsqueeze(1))

            box_center_norm_x_refined = (((bbox_center_2d_refined[:, 0, 0] - (x_scale * W_img) / 2) / W_img) * 2) + x_scale - 1
            box_center_norm_y_refined = (((bbox_center_2d_refined[:, 1, 0] - (y_scale * H_img) / 2) / H_img) * 2) + y_scale - 1

            (affine_matrix[:, 2])[(invalid_mask == False).squeeze(-1)] = (box_center_norm_x_refined + 1 - ((2 / grid_size) * b_ind_c) - (1 / grid_size))[(invalid_mask == False).squeeze(-1)]
            (affine_matrix[:, 3])[(invalid_mask == False).squeeze(-1)] = (box_center_norm_y_refined + 1 - ((2 / grid_size) * b_ind_r) - (1 / grid_size))[(invalid_mask == False).squeeze(-1)]

            return affine_matrix

        elif type == 'loss':
            pairwise_dist = torch.zeros(bbox_centers.shape[0]).cuda()
            for cam_group in range(bbox_centers.shape[0]):
                pairwise_dist[cam_group] = self.pairwise_distance(cam_position_w[cam_group], bbox_centers[cam_group], subbatch_size)

            return pairwise_dist

    def unproject_2Dgrid_to_3Dgrid(self, input_dict, coord_volumes, coord_volumes_visu, base_points, grid_dim, grid_probability, subbatch_size):
        H_img = input_dict['img'].shape[-2]
        W_img = input_dict['img'].shape[-1]
        batch_size = input_dict['img'].shape[0]
        n_views = subbatch_size
        volume_shape = coord_volumes.shape[1:4]
        extrinsics = input_dict['camera_extrinsics'].view((batch_size, 3, 4)).float().cuda()
        cam_ext = torch.cat((extrinsics, torch.FloatTensor([0, 0, 0, 1]).cuda().view(1, -1).unsqueeze(0).repeat(batch_size, 1, 1)), dim=1)
        intrinsics = input_dict['intrinsic'].view((batch_size, 3, 3)).float().cuda()
        cam_int = torch.cat((intrinsics, torch.zeros(batch_size, 3, 1).float().cuda()), dim=-1)
        cam_centers = input_dict['extrinsic_pos'].view((batch_size, 3)).float().cuda()
        volume_batch_to_aggregate = torch.zeros(batch_size, n_views, volume_shape[0], volume_shape[1], volume_shape[2]).cuda()
        volume_batch_to_aggregate_final = torch.zeros(batch_size, volume_shape[0], volume_shape[1], volume_shape[2]).cuda()
        volume_to_grid_view = torch.zeros(batch_size, n_views, volume_shape[0], volume_shape[1], volume_shape[2], 2).cuda().fill_(-1)
        volume_to_grid = torch.zeros(batch_size, volume_shape[0], volume_shape[1], volume_shape[2], 2).cuda().fill_(-1)
        volume_probability = torch.zeros(batch_size, volume_shape[0], volume_shape[1], volume_shape[2]).cuda()
        #volume_probability -= 9
        grid_coord = coord_volumes.reshape((batch_size, -1, 3))
        cam_total = cam_int @ cam_ext
        world_grid_4d = torch.cat([grid_coord, torch.ones(batch_size, grid_coord.shape[1], 1).float().cuda()], dim=-1)
        cam_grid_3d = torch.bmm(cam_total, world_grid_4d.permute(0, 2, 1))
        invalid_mask = cam_grid_3d[:, 2, :] > 0
        cam_grid_3d[:, 2, :][cam_grid_3d[:, 2, :] == 0] = 1
        pixel_2d = (cam_grid_3d[:, :-1, :] / cam_grid_3d[:,2,: ].unsqueeze(1))
        pixel_2d_w = pixel_2d[:, 0, :]
        pixel_2d_h = pixel_2d[:, 1, :]
        is_valid_w = (pixel_2d_w >= 0) & (pixel_2d_w < W_img)
        is_valid_h = (pixel_2d_h >= 0) & (pixel_2d_h < H_img)
        is_valid_pixel = invalid_mask * is_valid_h * is_valid_w

        valid_pixels_unravel = is_valid_pixel.view(batch_size, volume_shape[0], volume_shape[1], volume_shape[2]).nonzero().t()
        nb_pixels_width = torch.Tensor([W_img // grid_dim[1]]).cuda()
        nb_pixels_height = torch.Tensor([H_img // grid_dim[0]]).cuda()
        pixel_2d[:, 0, :] = pixel_2d[:, 0, :].long() / nb_pixels_width.long()
        pixel_2d[:, 1, :] = pixel_2d[:, 1, :].long() / nb_pixels_height.long()
        pixel_2d_unravel = pixel_2d.view(batch_size, 2, volume_shape[0], volume_shape[1], volume_shape[2])
        volume_to_grid[valid_pixels_unravel[0], valid_pixels_unravel[1], valid_pixels_unravel[2], valid_pixels_unravel[3]] = pixel_2d_unravel[valid_pixels_unravel[0], :, valid_pixels_unravel[1], valid_pixels_unravel[2], valid_pixels_unravel[3]]
        volume_to_grid_view = volume_to_grid.view(batch_size//subbatch_size, subbatch_size, volume_shape[0], volume_shape[1], volume_shape[2], 2).repeat(1, subbatch_size,1,1,1,1).view(batch_size, subbatch_size,volume_shape[0], volume_shape[1], volume_shape[2],2)

        volume_batch_to_aggregate_final[valid_pixels_unravel[0], valid_pixels_unravel[1], valid_pixels_unravel[2], valid_pixels_unravel[3]] = 1
        volume_batch_to_aggregate_final = volume_batch_to_aggregate_final.view(batch_size//subbatch_size, subbatch_size, volume_shape[0], volume_shape[1], volume_shape[2]).sum(1).long() / torch.Tensor([n_views]).long().cuda()
        volume_batch_to_aggregate_final = volume_batch_to_aggregate_final.unsqueeze(1).repeat(1, subbatch_size, 1, 1, 1).view(batch_size, volume_shape[0], volume_shape[1], volume_shape[2])
        overall_valid_voxels = volume_batch_to_aggregate_final.nonzero().t()
        valid_grid_c = volume_to_grid[overall_valid_voxels[0], overall_valid_voxels[1], overall_valid_voxels[2], overall_valid_voxels[3]][:, 0]
        valid_grid_r = volume_to_grid[overall_valid_voxels[0], overall_valid_voxels[1], overall_valid_voxels[2], overall_valid_voxels[3]][:, 1]
        valid_grid_r = valid_grid_r.long()
        valid_grid_c = valid_grid_c.long()

        volume_probability[overall_valid_voxels[0], overall_valid_voxels[1], overall_valid_voxels[2], overall_valid_voxels[3]] = grid_probability[overall_valid_voxels[0], valid_grid_r, valid_grid_c]
        volume_probability = torch.prod(volume_probability.view(batch_size//subbatch_size, subbatch_size, volume_shape[0], volume_shape[1], volume_shape[2]), dim=1)
        #volume_probability = torch.sum(volume_probability.view(batch_size//subbatch_size, subbatch_size, volume_shape[0], volume_shape[1], volume_shape[2]), dim=1)
        volume_probability = volume_probability.unsqueeze(1).repeat(1, subbatch_size, 1, 1, 1).view(batch_size, volume_shape[0], volume_shape[1], volume_shape[2])
        return volume_batch_to_aggregate_final, volume_to_grid_view, volume_to_grid, volume_probability


    def voting_3D_grid(self, input_dict, subbatch_size):
        batch_size = input_dict['img'].shape[0]
        height_img = input_dict['img'].shape[-2]
        width_img = input_dict['img'].shape[-1]
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3).cuda()
        coord_volumes_visu = torch.zeros(batch_size, self.volume_size + 1, self.volume_size + 1, self.volume_size + 1,
                                         3).cuda()
        cam_centers = input_dict['extrinsic_pos'].view((batch_size, 3)).float().cuda()

        base_points = torch.zeros(batch_size // subbatch_size, 3).float().cuda()
        cam_2_world = input_dict['extrinsic_rot_inv'].view((batch_size, 3, 3)).float().cuda()
        extrinsics = input_dict['camera_extrinsics'].view((batch_size, 3, 4)).float().cuda()
        extrinsics_inverse = input_dict['inverse_camera_extrinsics'].view((batch_size, 4, 4)).float().cuda()
        intrinsics_inverse = input_dict['inverse_intrinsic'].view((batch_size, 3, 3)).float().cuda()
        intrinsics = input_dict['intrinsic'].view((batch_size, 3, 3)).float().cuda()
        cam_position = (torch.FloatTensor([0, 0, 0, 1])).cuda().unsqueeze(0).repeat(batch_size, 1).view(batch_size, 4, 1)  # N*4*1
        cam_direction = (torch.FloatTensor([0, 0, 30, 1])).cuda().unsqueeze(0).repeat(batch_size, 1).view(batch_size, 4, 1)  # N*4*1
        img_pixel_center = torch.FloatTensor([width_img // 2, height_img // 2, 1]).view(3, 1).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        cam_info_batch = {}
        image_centers = torch.zeros(batch_size, 3).float().cuda()
        cam_info_batch['position'] = torch.zeros(batch_size, 3).float().cuda()
        cam_info_batch['pointtstoward'] = torch.zeros(batch_size, 3).float().cuda()
        cam_info_batch['position'] = torch.bmm(extrinsics_inverse, cam_position)[:, 0:3]
        cam_info_batch['pointtstoward'] = torch.bmm(extrinsics_inverse, cam_direction)[:, 0:3]
        center_m = torch.bmm(intrinsics_inverse, img_pixel_center)
        ############
        # sanity check
        # cam_vector = (cam_info_batch['pointtstoward'] - cam_centers.view(batch_size, 3, 1))
        # unnorm_ext_inv = (extrinsics_inverse[:, :, 2][:, 0:3])
        # assert (cam_vector / torch.norm(cam_vector)).view(-1, 3) == unnorm_ext_inv / torch.norm(unnorm_ext_inv)
        # print('center_m :', center_m)
        ############
        center_w = torch.bmm(extrinsics_inverse, torch.cat((center_m * 10, torch.empty(batch_size, 1, 1).fill_(1).cuda()), dim=1))[:, 0:3, :]
        image_centers = center_w.squeeze(-1)
        image_centers = image_centers.contiguous().view(-1, subbatch_size, image_centers.shape[-1])
        cam_info_batch['position'] = cam_info_batch['position'].squeeze(-1)
        cam_info_batch['position'] = cam_info_batch['position'].contiguous().view(-1, subbatch_size,
                                                                                  cam_info_batch['position'].shape[-1])
        for cam_group in range(image_centers.shape[0]):
            base_point = self.intersect(cam_info_batch['position'][cam_group], image_centers[cam_group])
            base_points[cam_group] = base_point[:, 0]
        sides = torch.FloatTensor([self.cuboid_side, self.cuboid_side, self.cuboid_side]).cuda()
        base_points_group = base_points.clone()
        base_points = base_points.unsqueeze(1).repeat(1, subbatch_size, 1).view(batch_size, base_points.shape[-1])
        position = base_points - (sides.unsqueeze(0) * (self.volume_size - 1) / self.volume_size) / 2
        position_visu = base_points - (sides.unsqueeze(0)) / 2

        grid = torch.stack([self.xxx, self.yyy, self.zzz], dim=-1).type(torch.float).cuda()
        grid = grid.reshape((-1, 3))
        grid_coord = torch.zeros(batch_size, self.volume_size * self.volume_size * self.volume_size, 3).cuda()

        grid_coord[:, :, 0] = (position[:, 0].view(-1, 1).repeat(1, self.volume_size * self.volume_size * self.volume_size) + ((sides[0] / (self.volume_size)) * grid[:, 0]).unsqueeze(0))
        grid_coord[:, :, 1] = (position[:, 1].view(-1, 1).repeat(1, self.volume_size * self.volume_size * self.volume_size) + ((sides[1] / (self.volume_size)) * grid[:, 1]).unsqueeze(0))
        grid_coord[:, :, 2] = (position[:, 2].view(-1, 1).repeat(1, self.volume_size * self.volume_size * self.volume_size) + ((sides[2] / (self.volume_size)) * grid[:, 2]).unsqueeze(0))

        coord_volumes = grid_coord.reshape(batch_size, self.volume_size, self.volume_size, self.volume_size, 3)

        grid_visu = torch.stack([self.xxx_visu, self.yyy_visu, self.zzz_visu], dim=-1).type(torch.float).cuda()
        grid_visu = grid_visu.reshape((-1, 3))
        grid_coord_visu = torch.zeros(batch_size, (self.volume_size + 1) * (self.volume_size + 1) * (self.volume_size + 1), 3).cuda()

        grid_coord_visu[:, :, 0] = (position_visu[:, 0].view(-1, 1).repeat(1, (self.volume_size + 1) * (self.volume_size + 1) * (self.volume_size + 1)) + ((sides[0] / (self.volume_size)) * grid_visu[:, 0]).unsqueeze(0))
        grid_coord_visu[:, :, 1] = (position_visu[:, 1].view(-1, 1).repeat(1, (self.volume_size + 1) * (self.volume_size + 1) * (self.volume_size + 1)) + ((sides[1] / (self.volume_size)) * grid_visu[:, 1]).unsqueeze(0))
        grid_coord_visu[:, :, 2] = (position_visu[:, 2].view(-1, 1).repeat(1, (self.volume_size + 1) * (self.volume_size + 1) * (self.volume_size + 1)) + ((sides[2] / (self.volume_size)) * grid_visu[:, 2]).unsqueeze(0))

        coord_volumes_visu = grid_coord_visu.reshape(batch_size, self.volume_size + 1, self.volume_size + 1, self.volume_size + 1, 3)

        return coord_volumes, coord_volumes_visu, base_points_group


    def axis_aligned_to_non_axis_aligned(self, mean_3D, diag_values, roll_pitch_yaw):

        rotation = self.euler_to_rotation_matrix(roll_pitch_yaw[:, 0], roll_pitch_yaw[:, 1], roll_pitch_yaw[:, 2])
        cov_matrix = rotation * diag_values * torch.eye(3).cuda() * rotation.t()
        return

    def project_ellipsoid_to_ellipse(self, mean_3D, cov_matrix):

        M = torch.inverse(cov_matrix) @ mean_3D @ mean_3D.t() @ torch.inverse(cov_matrix).t() - (
                                                                                                mean_3D.t() @ torch.inverse(
                                                                                                    cov_matrix) @ mean_3D - 1) @ torch.inverse(
            cov_matrix)
        return

    def euler_to_rotation_matrix(self, roll, pitch, yaw):

        return torch.FloatTensor([[torch.cos(yaw) * torch.cos(pitch),
                                   torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll) - torch.sin(yaw) * torch.cos(
                                       roll),
                                   torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll) + torch.sin(yaw) * torch.sin(
                                       roll)],
                                  [torch.sin(yaw) * torch.cos(pitch),
                                   torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll) + torch.cos(yaw) * torch.cos(
                                       roll),
                                   torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll) - torch.cos(yaw) * torch.sin(
                                       roll)],
                                  [-torch.sin(pitch), torch.cos(pitch) * torch.sin(roll),
                                   torch.cos(pitch) * torch.cos(roll)]]).cuda()

    def forward(self, input_dict, niter=None):

        return self.forward_MultiCam(input_dict, niter)

    def forward_detector(self, input_dict, shuffled_crops=None, niter=None):

        # downscale input image before running the detector
        input_img = input_dict['img'].squeeze()
        batch_size = input_dict['img'].shape[0]
        ST_size = self.ST_size

        def ST_flatten(batch):
            shape_orig = batch.shape
            # assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = [ST_size * batch_size] + list(shape_orig[2:])
            return batch.view(shape_new)

        def ST_split(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size * batch_size
            shape_new = [ST_size, batch_size] + list(shape_orig[1:])
            return batch.view(shape_new)

        ### downscale image ###
        # Warning, squeeze is a hack to remove the additional time/frame dimension from nvvl video loadings
        affine_matrix_downscale = torch.FloatTensor([[1, 0, 0], [0, 1, 0]])
        affine_matrix_downscale = affine_matrix_downscale.unsqueeze(0).repeat(batch_size, 1, 1).cuda()

        # simulate randomized crops to regularize training
        if self.offset_crop and self.training:
            offsets = torch.FloatTensor((np.random.random([batch_size, 2]) - 0.5) * 0.1).cuda()
            affine_matrix_downscale[:, :2, 2] = offsets

        size = torch.Size([batch_size, 3, self.detection_resolution, self.detection_resolution])
        grid_downscale = F.affine_grid(affine_matrix_downscale, size=size)
        input_img_downscaled = F.grid_sample(input_img, grid_downscale, padding_mode='border')

        ### run detector ###
        affine_params_grid = self.detector.forward(input_img_downscaled)['affine']
        # print(affine_params_grid.shape)
        grid_size = affine_params_grid.shape[-1]

        epsilon = 0.000001
        sampling_uniform_prob = 0.01
        affine_params_grid = affine_params_grid.permute(0, 2, 3, 1)

        affine_params_grid[:, :, :, 0] = torch.sigmoid(affine_params_grid[:, :, :, 0])
        affine_params_grid[:, :, :, 1] = torch.sigmoid(affine_params_grid[:, :, :, 1])
        affine_params_grid[:, :, :, 2] = torch.tanh(affine_params_grid[:, :, :, 2]) * (self.offset_range / grid_size)
        affine_params_grid[:, :, :, 3] = torch.tanh(affine_params_grid[:, :, :, 3]) * (self.offset_range / grid_size)

        if self.only_center_grid:
            affine_params_grid[:, 0, :, 5] *= 0
            affine_params_grid[:, -1, :, 5] *= 0
            affine_params_grid[:, :, 0, 5] *= 0
            affine_params_grid[:, :, -1, 5] *= 0
            affine_params_grid[:, 0, :, 5] -= 999999
            affine_params_grid[:, -1, :, 5] -= 999999
            affine_params_grid[:, :, 0, 5] -= 999999
            affine_params_grid[:, :, -1, 5] -= 999999

        self.confidence_before_softmax = affine_params_grid[:, :, :, 5].clone()
        affine_params_grid[:, :, :, 5] = torch.nn.functional.softmax(affine_params_grid[:, :, :, 5].contiguous().view(-1, grid_size * grid_size), dim=1).contiguous().view(-1, grid_size, grid_size)
        self.grid_matrix = affine_params_grid[:, :, :, 5].contiguous().clone()  # .view(batch_size, grid_size, grid_size).clone()

        # create the 3d grid for voting
        input_dict_cpu = {}
        for key in input_dict.keys():
            input_dict_cpu[key] = input_dict[key].cpu()


        if self.training:
            self.coord_volumes, self.coord_volumes_visu, self.base_points = self.voting_3D_grid(input_dict, int(batch_size / len(torch.unique(input_dict['file_name_info'][:, 1:3], dim=0))))
            self.itr_check += 1

        if self.training:
            self.vote_aggregate, self.volume_to_grid_view, self.volume_to_grid, self.voxel_confidence = self.unproject_2Dgrid_to_3Dgrid(input_dict, self.coord_volumes, self.coord_volumes_visu, self.base_points,  [affine_params_grid.shape[-3], affine_params_grid.shape[-2]], affine_params_grid[:, :, :, 5], int(batch_size / len(torch.unique(input_dict['file_name_info'][:, 1:3], dim=0))))

        self.C = self.volume_size * self.volume_size * self.volume_size
        one_over_Z = (1 - self.C * epsilon)
        # Backproject
        # self.backproject_to_3D_grid(input_dict, self.coord_volumes, self.vote_aggregate, self.volume_to_grid, affine_params_grid)

        self.all_offsets = torch.zeros([batch_size, grid_size, grid_size, 2]).cuda()
        self.all_scales = torch.zeros([batch_size, grid_size, grid_size, 2]).cuda()
        for c_x in range(grid_size):
            for c_y in range(grid_size):
                self.all_offsets[:, c_x, c_y, 0] = torch.clamp((affine_params_grid[:, c_y, c_x, 2] + 1 / grid_size + ((2 / grid_size) * c_x) - 1), min=-1, max=1)
                self.all_offsets[:, c_x, c_y, 1] = torch.clamp((affine_params_grid[:, c_y, c_x, 3] + 1 / grid_size + ((2 / grid_size) * c_y) - 1), min=-1, max=1)
                self.all_scales[:, c_x, c_y, 0] = self.spatial_transformer_bounds['min_size'] + (
                                                                                                self.spatial_transformer_bounds[
                                                                                                    'max_size'] -
                                                                                                self.spatial_transformer_bounds[
                                                                                                    'min_size']) * affine_params_grid[
                                                                                                                   :,
                                                                                                                   c_y,
                                                                                                                   c_x,
                                                                                                                   0]
                self.all_scales[:, c_x, c_y, 1] = self.spatial_transformer_bounds['min_size'] + (
                                                                                                self.spatial_transformer_bounds[
                                                                                                    'max_size'] -
                                                                                                self.spatial_transformer_bounds[
                                                                                                    'min_size']) * affine_params_grid[
                                                                                                                   :,
                                                                                                                   c_y,
                                                                                                                   c_x,
                                                                                                                   1]

        matching_colors = ['orange', 'cyan', 'magenta', 'maroon', 'darkgreen', 'blueviolet']
        self.top_k = 3
        if self.choose_cell == 'Uniform':
            if self.only_center_grid:
                b_ind_r = torch.LongTensor(batch_size).random_(1, grid_size - 1).float().cuda()
                b_ind_c = torch.LongTensor(batch_size).random_(1, grid_size - 1).float().cuda()
            else:
                b_ind_r = torch.LongTensor(batch_size).random_(0, grid_size).float().cuda()
                b_ind_c = torch.LongTensor(batch_size).random_(0, grid_size).float().cuda()

        elif self.choose_cell == 'Importance':

            if self.training:

                q_prob = self.voxel_confidence.clone()
                q_prob = q_prob.contiguous().view(-1, self.volume_size * self.volume_size * self.volume_size)
                sum_prob = q_prob.sum(dim=1).unsqueeze(-1)
                q_prob = q_prob / sum_prob

                q_prob = q_prob.contiguous().view(batch_size, self.volume_size, self.volume_size, self.volume_size)
                self.activated_voxels = self.vote_aggregate.contiguous().view(-1, self.volume_size * self.volume_size * self.volume_size).sum(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, self.volume_size, self.volume_size, self.volume_size)
                epsilon = torch.FloatTensor([sampling_uniform_prob]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / self.activated_voxels
                one_over_Z = (1 - self.activated_voxels * epsilon)
                q_prob[self.vote_aggregate == 1] = q_prob[self.vote_aggregate == 1] * one_over_Z[self.vote_aggregate == 1] + epsilon[self.vote_aggregate == 1]
                q_prob = q_prob.contiguous().view(-1, self.volume_size * self.volume_size * self.volume_size)
                proposal_dist = torch.distributions.categorical.Categorical(probs=(q_prob[0::self.subbatch_size]))
                voxel_feasible = False
                counter_feasible = 0
                while not voxel_feasible:
                    # v_ind = proposal_dist.sample()
                    _, v_ind = q_prob[0::self.subbatch_size].max(1)
                    v_ind = v_ind.unsqueeze(1).repeat(1, self.subbatch_size).view(batch_size, -1).squeeze(-1)

                    if (
                    self.vote_aggregate.contiguous().view(-1, self.volume_size * self.volume_size * self.volume_size))[
                        torch.arange(0, batch_size).long(), v_ind].sum().item() < batch_size:
                        counter_feasible += 1
                        if counter_feasible >= 10:
                            print("TRAINING GOT STUCK DUE TO THE INFINITE LOOP OF SAMPLING !!!!")
                        continue
                    v_ind_unravel = unravel_index(v_ind, (self.volume_size, self.volume_size, self.volume_size))
                    prob = self.voxel_confidence[torch.arange(0, batch_size).long(), v_ind_unravel[0], v_ind_unravel[1], v_ind_unravel[2]]
                    voxel_feasible = True
                b_ind = self.volume_to_grid[torch.arange(0, batch_size).long(), v_ind_unravel[0], v_ind_unravel[1], v_ind_unravel[2]]
                b_ind_c = b_ind[:, 0]
                b_ind_r = b_ind[:, 1]
                self.vind = v_ind
                self.bindc = b_ind_c
                self.bindr = b_ind_r

                v_ind_sorted = self.voxel_confidence.clone().contiguous().view(-1,
                                                                               self.volume_size * self.volume_size * self.volume_size).sort(
                    1)[1][:, -1 * self.top_k:]
                v_ind_sorted_unravel = unravel_index(v_ind_sorted,
                                                     (self.volume_size, self.volume_size, self.volume_size))
                matching_list = []
                top_k_list = []
                for tk in range(self.top_k):
                    b_ind_sorted = self.volume_to_grid[
                        torch.arange(0, batch_size).long(), v_ind_sorted_unravel[0][:, tk], v_ind_sorted_unravel[1][:,
                                                                                            tk], v_ind_sorted_unravel[
                                                                                                     2][:, tk]]
                    b_ind_c_sorted = b_ind_sorted[:, 0]
                    b_ind_r_sorted = b_ind_sorted[:, 1]
                    matching_list.append((b_ind_r_sorted, b_ind_c_sorted, matching_colors[tk]))
                    top_k_list.append((b_ind_r_sorted.detach().cpu().numpy(), b_ind_c_sorted.detach().cpu().numpy(),
                                       (v_ind_sorted_unravel[0][:, tk]).detach().cpu().numpy(),
                                       (v_ind_sorted_unravel[1][:, tk]).detach().cpu().numpy(),
                                       (v_ind_sorted_unravel[2][:, tk]).detach().cpu().numpy()))
                self.matching_list = matching_list
                self.top_k_list = top_k_list


            else:  # test
                #####################################
                # only for the consistency of the keys
                # prob_test = self.voxel_confidence.clone().contiguous().view(-1, self.volume_size * self.volume_size * self.volume_size)
                # prob_test = prob_test + 0.00001  # to avoid zero sum when camBatch is 4 or 5
                # sum_prob = prob_test.sum(dim=1).unsqueeze(-1)
                # prob_test = prob_test / sum_prob
                # prob, v_ind = prob_test.max(1)
                # v_ind_unravel = unravel_index(v_ind, (self.volume_size, self.volume_size, self.volume_size))
                # b_ind = self.volume_to_grid[torch.arange(0, batch_size).long(), v_ind_unravel[0], v_ind_unravel[1], v_ind_unravel[2]]
                # b_ind_c = b_ind[:, 0]
                # b_ind_r = b_ind[:, 1]
                # self.vind = v_ind
                # # self.bindc = b_ind_c
                # # self.bindr = b_ind_r
                #
                # v_ind_sorted = self.voxel_confidence.clone().contiguous().view(-1, self.volume_size * self.volume_size * self.volume_size).sort(1)[1][:, -1 * self.top_k:]
                # v_ind_sorted_unravel = unravel_index(v_ind_sorted,(self.volume_size, self.volume_size, self.volume_size))
                # matching_list = []
                # for tk in range(self.top_k):
                #     b_ind_sorted = self.volume_to_grid[
                #         torch.arange(0, batch_size).long(), v_ind_sorted_unravel[0][:, tk], v_ind_sorted_unravel[1][:,
                #                                                                             tk], v_ind_sorted_unravel[
                #                                                                                      2][:, tk]]
                #     b_ind_c_sorted = b_ind_sorted[:, 0]
                #     b_ind_r_sorted = b_ind_sorted[:, 1]
                #     matching_list.append((b_ind_r_sorted, b_ind_c_sorted, matching_colors[tk]))
                #
                # self.matching_list = matching_list
                ######################################

                # at test time pick the max from the affine params
                _, b_ind = affine_params_grid[:, :, :, 5].clone().contiguous().view(-1, grid_size * grid_size).max(1)
                b_ind_r = torch.floor(torch.div(b_ind.float(), grid_size)).float().cuda()
                b_ind_c = torch.floor(torch.remainder(b_ind.float(), grid_size)).float().cuda()

                self.bindc = b_ind_c
                self.bindr = b_ind_r

        self.selected_cell_r = b_ind_r
        self.selected_cell_c = b_ind_c
        self.cell_map = torch.zeros(self.grid_matrix.shape)
        # Select the sampled cell
        self.confidence_before_softmax = self.confidence_before_softmax[torch.arange(batch_size).long(), b_ind_r.long(), b_ind_c.long()]
        affine_params = affine_params_grid[torch.arange(batch_size).long(), b_ind_r.long(), b_ind_c.long(), :]
        self.confidence_shape = affine_params[:, 5].shape
        self.cell_map[torch.arange(batch_size).long(), self.selected_cell_r.long(), self.selected_cell_c.long()] = 1

        #Set p (confidence) and q(proposal)
        if self.training:
            self.confidence = prob # PUT BACK!!!
        else:
            self.confidence = affine_params[:, 5]

        if self.choose_cell == 'Uniform':
            self.proposal = 1 / (torch.ones(self.confidence.shape) * torch.FloatTensor([self.C])).cuda()
        # BE CAREFUL!
        elif self.choose_cell == 'Importance':
            if self.training:
                self.activated_voxels = self.vote_aggregate.contiguous().view(-1,self.volume_size * self.volume_size * self.volume_size).sum(1)
                epsilon = torch.FloatTensor([sampling_uniform_prob]).cuda() / self.activated_voxels
                one_over_Z = (1 - self.activated_voxels * epsilon)
                self.proposal = prob * (one_over_Z.squeeze(0)) + (epsilon.squeeze(0))
            else:

                #self.proposal = prob * (one_over_Z) + (epsilon)
                self.C = grid_size*grid_size
                one_over_Z = (1 - self.C * 0.0001)
                self.proposal = affine_params[:, 5] * (one_over_Z) + (0.0001)

        # apply non-linearities and extract depth
        border_factor = self.spatial_transformer_bounds['border_factor']
        min_size = self.spatial_transformer_bounds['min_size']
        max_size = self.spatial_transformer_bounds['max_size']
        affine_params[:, 0] = min_size + (max_size - min_size) * affine_params[:, 0]
        affine_params[:, 1] = min_size + (max_size - min_size) * affine_params[:, 1]
        self.scale_x = affine_params[:, 0]
        self.scale_y = affine_params[:, 1]

        ########### MultiCam Offset Consistency ###########
        if self.training:
            if self.offset_consistency_type == 'projection':
                affine_params = self.offset_consistency(input_dict, b_ind_r, b_ind_c, affine_params, grid_size, self.itr_check, int(batch_size / len(torch.unique(input_dict['file_name_info'][:, 1:3], dim=0))), type=self.offset_consistency_type)
            elif self.offset_consistency_type == 'loss':
                self.pairwise_dist = self.offset_consistency(input_dict, b_ind_r, b_ind_c, affine_params, grid_size, self.itr_check, int(batch_size / len(torch.unique(input_dict['file_name_info'][:, 1:3], dim=0))), type=self.offset_consistency_type)

        ###################################################
        self.offset_x = affine_params[:, 2]
        self.offset_y = affine_params[:, 3]

        affine_matrix_crop = torch.zeros([batch_size * ST_size, 2, 3]).cuda()
        affine_matrix_crop[:, 0, 0] = affine_params[:, 0]
        affine_matrix_crop[:, 1, 1] = affine_params[:, 1]
        affine_matrix_crop[:, 0, 2] = torch.clamp((affine_params[:, 2] + 1 / grid_size + ((2 / grid_size) * b_ind_c) - 1), min=-1,max=1)  # normalized to -1..1
        affine_matrix_crop[:, 1, 2] = torch.clamp((affine_params[:, 3] + 1 / grid_size + ((2 / grid_size) * b_ind_r) - 1), min=-1, max=1)

        self.selected_cell_centers_x = (- (grid_size - 1) / grid_size + ((2 / grid_size) * b_ind_c))
        self.selected_cell_centers_y = (- (grid_size - 1) / grid_size + ((2 / grid_size) * b_ind_r))

        if self.predict_transformer_depth:
            ST_depth = affine_params[:, 4]
            ST_depth = ST_depth.view(batch_size, ST_size).transpose(1, 0).contiguous()
        else:
            ST_depth = None

        if self.disable_detector:
            affine_matrix_crop[:, 0, 0] = 1
            affine_matrix_crop[:, 1, 1] = 1
            affine_matrix_crop[:, 0:2, 2] = 0  # normalized to -1..1
        affine_matrix_crop_raw = affine_matrix_crop

        if 'spatial_transformer' in input_dict:
            affine_matrix_crop = input_dict['spatial_transformer']

        # inverse of the affine transformation (exploiting the near-diagonal structure)
        affine_matrix_uncrop = torch.zeros(affine_matrix_crop.shape).cuda()
        affine_matrix_uncrop[:, 0, 0] = (1 / affine_matrix_crop[:, 0, 0])
        affine_matrix_uncrop[:, 1, 1] = (1 / affine_matrix_crop[:, 1, 1])

        affine_matrix_uncrop[:, 0, 2] = -affine_matrix_crop[:, 0, 2] / affine_matrix_crop[:, 0, 0]
        affine_matrix_uncrop[:, 1, 2] = -affine_matrix_crop[:, 1, 2] / affine_matrix_crop[:, 1, 1]
        if 1 and 'shuffled_appearance_weight' in input_dict.keys():
            w = input_dict['shuffled_appearance_weight'].item()
            a0 = (1 - w) * affine_matrix_uncrop[0, :, :] + w * affine_matrix_uncrop[2, :,
                                                               :]  # blend between neighboring frames, same ST
            a1 = (1 - w) * affine_matrix_uncrop[1, :, :] + w * affine_matrix_uncrop[3, :,
                                                               :]  # blend between neighboring frames, same ST
            affine_matrix_uncrop[0, :, :] = a0
            affine_matrix_uncrop[1, :, :] = a1

        # make the ST dimension the first one, as needed for cropping
        affine_matrix_crop_multi = affine_matrix_crop.view(batch_size, ST_size, 2, 3).transpose(1, 0).contiguous()
        affine_matrix_uncrop_multi = affine_matrix_uncrop.view(batch_size, ST_size, 2, 3).transpose(1, 0).contiguous()

        # randomize the order, so the detector can not learn to order crops (otherwise he learns left to right)
        if self.shuffle_crops:
            affine_matrix_crop_multi = ST_split(
                torch.index_select(ST_flatten(affine_matrix_crop_multi), dim=0, index=shuffled_crops))
            affine_matrix_uncrop_multi = ST_split(
                torch.index_select(ST_flatten(affine_matrix_uncrop_multi), dim=0, index=shuffled_crops))
            if self.predict_transformer_depth:
                ST_depth = ST_split(torch.index_select(ST_flatten(ST_depth), dim=0, index=shuffled_crops))

        # apply spatial transformers (crop input and bg images)
        img_crop = []
        bg_crop = []
        for j in range(ST_size):
            output_size = torch.Size([batch_size, 3, self.in_resolution, self.in_resolution])
            grid_crop = F.affine_grid(affine_matrix_crop_multi[j, :, :, :],
                                      size=output_size)  # Note, can not output multiple candidates
            img_crop.append(F.grid_sample(input_img, grid_crop))
            if 'bg' in input_dict:
                bg_img = input_dict['bg']
                bg_crop.append(F.grid_sample(bg_img, grid_crop))

        if 'bg' in input_dict:
            bg_crop = torch.cat(bg_crop)
        else:
            bg_crop = torch.ones(batch_size, input_dict['img'].shape[1], input_dict['img'].shape[2], input_dict['img'].shape[3])
        img_crop = torch.cat(img_crop)
        # apply smooth spatial transformer in forward pass
        if self.spatial_transformer and unet.bump_function is not None:
            img_crop = img_crop * unet.bump_function.unsqueeze(0).unsqueeze(0)

        input_dict_cropped = {'img_crop': img_crop,
                              'bg_crop': bg_crop}
        return input_dict_cropped, input_img_downscaled, ST_depth, affine_matrix_crop_multi, affine_matrix_uncrop_multi, affine_matrix_crop_raw

    def forward_MultiCam(self, input_dict, niter=None):
        if 'img' in input_dict.keys():
            batch_size = input_dict['img'].size()[0]
        else:
            batch_size = input_dict['img_crop'].size()[0]
        ST_size = self.ST_size

        num_pose_examples = batch_size // 2
        num_appearance_examples = batch_size // 2
        num_appearance_subbatches = num_appearance_examples // np.maximum(self.subbatch_size, 1)

        def ST_flatten(batch):
            shape_orig = batch.shape
            # assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = [ST_size * batch_size] + list(shape_orig[2:])
            return batch.view(shape_new)

        def ST_split(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size * batch_size
            shape_new = [ST_size, batch_size] + list(shape_orig[1:])
            return batch.view(shape_new)

        def features_flatten(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = list(shape_orig[:2]) + [-1]
            return batch.view(shape_new)

        def features_split3D(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = list(shape_orig[:2]) + [-1, 3]
            return batch.view(shape_new)

        ########################################################
        # Determine shuffling
        shuffled_appearance = list(range(batch_size))
        shuffled_pose = list(range(batch_size))
        shuffled_crops = list(range(batch_size * ST_size))
        num_pose_subbatches = batch_size // np.maximum(self.subbatch_size, 1)

        if len(self.transductive_training):
            subj = input_dict['subj']

        # only if no user input is provided
        rotation_by_user = self.training == False and 'external_rotation_cam' in input_dict.keys()
        if not rotation_by_user:
            if self.shuffle_fg and self.training == True:
                for i in range(0, num_pose_subbatches):
                    self.roll_segment_random(shuffled_appearance, i * self.subbatch_size, (i + 1) * self.subbatch_size)
                for i in range(0, num_pose_subbatches // 2):  # flip first with second subbatch
                    self.flip_segment(shuffled_appearance, i * 2 * self.subbatch_size, self.subbatch_size)
            if self.shuffle_3d:
                for i in range(0, num_pose_subbatches):
                    # don't rotate on test subjects, to mimick that we don't know the rotation
                    if len(self.transductive_training) == 0 or int(
                            subj[i * self.subbatch_size]) not in self.transductive_training:
                        self.roll_segment_random(shuffled_pose, i * self.subbatch_size, (i + 1) * self.subbatch_size,
                                                 prob=self.shuffle_prob)
            if self.shuffle_crops:
                shuffled_crops_array = np.array(shuffled_crops).reshape([ST_size, batch_size])
                for i in range(0, batch_size):
                    index_list = shuffled_crops_array[:, i].tolist()
                    random.shuffle(index_list)
                    shuffled_crops_array[:, i] = index_list
                    shuffled_crops = shuffled_crops_array.reshape(-1).tolist()

        # infer inverse mapping
        shuffled_pose_inv = [-1] * batch_size
        for i, v in enumerate(shuffled_pose):
            shuffled_pose_inv[v] = i

        shuffled_appearance = torch.LongTensor(shuffled_appearance).cuda()
        shuffled_pose = torch.LongTensor(shuffled_pose).cuda()
        shuffled_pose_inv = torch.LongTensor(shuffled_pose_inv).cuda()
        shuffled_crops = torch.LongTensor(shuffled_crops).cuda()

        shuffled_crops_inv = shuffled_crops.clone()
        for i, v in enumerate(shuffled_crops):
            shuffled_crops_inv[v] = i

        if rotation_by_user:
            if 'shuffled_appearance' in input_dict.keys():
                shuffled_appearance = input_dict['shuffled_appearance'].long()

        ###############################################
        # determine shuffled rotation
        if 'extrinsic_rot_inv' not in input_dict:
            input_dict['extrinsic_rot_inv'] = torch.ones((batch_size, 3, 3)).float().cuda()
        if 'external_rotation_cam' not in input_dict:
            input_dict['external_rotation_cam'] = torch.ones((1, 3, 3)).float().cuda()
        if 'extrinsic_rot' not in input_dict:
            input_dict['extrinsic_rot'] = torch.ones((batch_size, 3, 3)).float().cuda()
        if 'external_rotation_global' not in input_dict:
            input_dict['external_rotation_global'] = torch.ones((1, 3, 3)).float().cuda()
            
        cam_2_world = input_dict['extrinsic_rot_inv'].view((batch_size, 3, 3)).float()
        world_2_cam = input_dict['extrinsic_rot'].view((batch_size, 3, 3)).float()
        if rotation_by_user:
            external_cam = input_dict['external_rotation_cam'].view(1, 3, 3).expand((batch_size, 3, 3))
            external_glob = input_dict['external_rotation_global'].view(1, 3, 3).expand((batch_size, 3, 3))
            cam2cam = torch.bmm(external_cam, torch.bmm(world_2_cam, torch.bmm(external_glob, cam_2_world)))
        else:
            world_2_cam_shuffled = torch.index_select(world_2_cam, dim=0, index=shuffled_pose)
            cam2cam = torch.bmm(world_2_cam_shuffled, cam_2_world)

        if 'shuffled_appearance_weight' in input_dict.keys():
            w = input_dict['shuffled_appearance_weight'].item()
            shuffled_poseX = list(range(batch_size))
            shuffled_poseX[0], shuffled_poseX[1] = shuffled_poseX[1], shuffled_poseX[0]
            shuffled_poseX = torch.LongTensor(shuffled_poseX).cuda()
            world_2_cam_shuffledX = torch.index_select(world_2_cam, dim=0, index=shuffled_poseX)
            cam2camX = torch.bmm(world_2_cam_shuffledX, cam_2_world)

            q_int = (1 - w) * linalg.mat2quat(cam2cam[0].cpu().numpy()) + w * linalg.mat2quat(cam2camX[0].cpu().numpy())
            R_int = linalg.quat2mat(q_int)
            cam2cam[0] = torch.FloatTensor(R_int)

        ###############################################
        # spatial transformer
        if self.spatial_transformer:
            input_dict_cropped, input_img_downscaled, ST_depth, affine_matrix_crop_multi, affine_matrix_uncrop_multi, affine_matrix_crop_raw = self.forward_detector(
                input_dict, shuffled_crops, niter)
        else:
            input_dict_cropped = input_dict  # fallback to using crops from dataloader

        ###############################################
        # encoding stage
        output = self.encoder.forward(input_dict_cropped)['latent_3d']
        has_fg = hasattr(self, "to_fg")
        if has_fg:
            latent_fg = output[:, :self.dimension_fg]  # .contiguous().clone() # TODO
            latent_fg = ST_split(latent_fg)
        latent_3d = output[:, self.dimension_fg:self.dimension_fg + self.dimension_3d]
        latent_3d = features_split3D(ST_split(latent_3d))  # transform it into a 3D latent space

        ###############################################
        # rotation prediction (Note, overwrites used GT rotation)
        if self.predict_rotation:
            assert False

        ###############################################
        # latent rotation (to shuffled view)
        if self.implicit_rotation:
            encoded_angle = self.encode_angle(cam2cam.view(batch_size, -1))
            encoded_latent_and_angle = torch.cat([latent_3d.view(batch_size * ST_size, -1), encoded_angle], dim=1)
            latent_3d_rotated = self.rotate_implicitely(encoded_latent_and_angle)
        else:
            cam2cam_replicated = cam2cam.unsqueeze(0).repeat([ST_size, 1, 1, 1])
            cam2cam_replicated_transposed = ST_flatten(cam2cam_replicated).transpose(1, 2)
            latent_3d_rotated = torch.bmm(ST_flatten(latent_3d), cam2cam_replicated_transposed)
            latent_3d_rotated = ST_split(latent_3d_rotated)

        # user input to flip pose
        if 'shuffled_pose_weight' in input_dict.keys():
            w = input_dict['shuffled_pose_weight']
            # weighted average with the last one
            latent_3d_rotated = (1 - w.expand_as(latent_3d)) * latent_3d \
                                + w.expand_as(latent_3d) * latent_3d_rotated[-1:].expand_as(latent_3d)

        # shuffle appearance based on flipping indices or user input
        if has_fg:
            # shuffle the appearance for all candidate crops
            # latent_fg_time_shuffled = torch.index_select(latent_fg, dim=0, index=shuffled_appearance_multiple)
            latent_fg_time_shuffled = torch.index_select(latent_fg, dim=1, index=shuffled_appearance)

        # compute similarity matrix
        if self.match_crops and has_fg:
            # TODO: similarities across time
            latent_fg_target = latent_fg  # this is the bbox and appearance information to which we decode
            latent_fg_source = torch.index_select(latent_fg, dim=1, index=shuffled_pose_inv)  # the one we encode from

            # expand along dim 0 and 1 respectively to compute covariance
            square_shape = [ST_size] + list(latent_fg_target.shape)
            # Note, selecting only the first 16 (out of usually 128) to make not the whole appearance space dependent
            num_matching_channels = 16
            # num_matching_channels = 128
            latent_fg_target_exp = latent_fg_target.unsqueeze(1).expand(square_shape)[:, :, :, :num_matching_channels]
            # TODO: is this expand needed? Broadcasting should work..
            latent_fg_source_exp = latent_fg_source.unsqueeze(0).expand(square_shape)[:, :, :, :num_matching_channels]
            eps = 0.0001
            dot_product = torch.sum(latent_fg_source_exp * latent_fg_target_exp, dim=-1)
            angle_matrix = dot_product / (
                eps + torch.norm(latent_fg_source_exp, dim=-1)
                * torch.norm(latent_fg_target_exp, dim=-1))  # cos angle
            # defined in a way, that it is 1 if the two encodings are identical and -1 if they are opposing
            # the rows of the resulting matrix assign a weighted average to the other (rotated) view, i.e. the first rows represent the weights for the first crop in the output image
            similarity_matrix = angle_matrix  # torch.sum(correlation_matrix, dim=3)
            if torch.isnan(similarity_matrix).any():
                print('WARNING: torch.isnan(correlation_matrix)')
                IPython.embed()

            def softmax2D(c):
                bandwidth = self.similarity_bandwidth  # was 2, higher values will lead to a sharper max
                c_sm0 = F.softmax(c * bandwidth, dim=1)
                return c_sm0  # ensures weigths 1 per crop, but not exclusive

            def hardmax2D(c):
                c = softmax2D(c)  # first to the usual softmax computation with badwidth
                max, arg_max = torch.max(c, dim=1, keepdim=True)
                c_sm0 = torch.zeros(c.shape).cuda()
                for i in range(batch_size):
                    for s in range(ST_size):
                        c_sm0[s, arg_max[s, 0, i], i] = 1
                return c_sm0


            if self.training:
                similarity_matrix_normalized = softmax2D(similarity_matrix)
            else:
                similarity_matrix_normalized = hardmax2D(similarity_matrix)
        else:
            # fixed assignment at test time
            similarity_matrix_normalized = torch.zeros([ST_size, ST_size, batch_size]).cuda()
            similarity_matrix_normalized[:, :, :] = 0
            for STi in range(ST_size):
                similarity_matrix_normalized[STi, STi, :] = 1

        # get the depth of the decoding view?
        latent_combined = features_flatten(latent_3d_rotated)

        ###############################################
        # decoding
        map_from_3d = ST_split(self.from_latent(ST_flatten(latent_combined)))
        map_width = self.bottleneck_resolution
        map_channels = self.bottlneck_feature_dim
        if has_fg:
            latent_fg_time_shuffled_replicated_spatially = latent_fg_time_shuffled.unsqueeze(-1).unsqueeze(-1).expand(
                ST_size, batch_size, self.dimension_fg, map_width, map_width)
            latent_shuffled = torch.cat([latent_fg_time_shuffled_replicated_spatially,
                                         map_from_3d.view(ST_size, batch_size,
                                                          map_channels - self.dimension_fg, map_width,
                                                          map_width)], dim=2)
        else:
            latent_shuffled = map_from_3d.view(ST_size, batch_size,
                                               map_channels, map_width, map_width)

        output_crop_rotated = self.decoder(ST_flatten(latent_shuffled))
        output_crop_rotated = ST_split(output_crop_rotated)
        ###############################################
        # de-shuffling
        output_crop = torch.index_select(output_crop_rotated, dim=1, index=shuffled_pose_inv)

        if self.masked_blending:
            output_img_crop = output_crop[:, :, 0:3, :, :]
            mask_enforced_minimum = 0.0001
            output_mask_raw = output_crop[:, :, 3:4, :, :]
            mask_prior_loss = torch.mean(torch.pow(output_mask_raw, 2))
            output_mask_crop = mask_enforced_minimum + (1.0 - mask_enforced_minimum) * F.sigmoid(output_mask_raw)
            if self.spatial_transformer and unet.bump_function is not None:
                output_mask_crop = output_mask_crop * unet.bump_function.unsqueeze(0).unsqueeze(0)
            output_mask_crop_before_scale = output_mask_crop.clone()
            if self.scale_mask_max_to_1:
                mask_max, max_index = torch.max(features_flatten(output_mask_crop), dim=2)
                output_mask_crop = output_mask_crop / (0.0001 + mask_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        else:
            mask_prior_loss = 0
            output_img_crop = output_crop
        ##############################
        # # BG prediction
        self.margin_per = 0.20  # normally 0.2
        if self.estimate_background:
            blank_locations = affine_matrix_crop_multi.clone().detach()
            if self.bbox_random:
                blank_locations[:, :, 0, 0] = torch.FloatTensor(blank_locations.shape[0],
                                                                blank_locations.shape[1]).uniform_(0.30, 0.50)
                blank_locations[:, :, 1, 1] = torch.FloatTensor(blank_locations.shape[0],
                                                                blank_locations.shape[1]).uniform_(0.30, 0.50)
                blank_locations[:, :, 0, 2] = torch.FloatTensor(blank_locations.shape[0],
                                                                blank_locations.shape[1]).uniform_(-1, 1)
                blank_locations[:, :, 1, 2] = torch.FloatTensor(blank_locations.shape[0],
                                                                blank_locations.shape[1]).uniform_(-1, 1)
            else:

                # add margin
                max_scale = torch.max(blank_locations[:, :, 0, 0].clone(), blank_locations[:, :, 1, 1].clone())
                blank_locations[:, :, 0, 0] = (blank_locations[:, :, 0, 0].clone() + 2 * self.margin_per * max_scale)
                blank_locations[:, :, 1, 1] = (blank_locations[:, :, 1, 1].clone() + 2 * self.margin_per * max_scale)

            ### differentiable version of blacking out ###
            # compute inverse transformation
            Affine_inv = torch.zeros(blank_locations.shape).cuda()
            Affine_inv[:, :, 0, 0] = (1 / blank_locations[:, :, 0, 0])
            Affine_inv[:, :, 1, 1] = (1 / blank_locations[:, :, 1, 1])

            Affine_inv[:, :, 0, 2] = -blank_locations[:, :, 0, 2] / blank_locations[:, :, 0, 0]
            Affine_inv[:, :, 1, 2] = -blank_locations[:, :, 1, 2] / blank_locations[:, :, 1, 1]

            bg_mask_inp = torch.ones(batch_size, 1, self.in_resolution, self.in_resolution).float().cuda()
            grid_uncrop = F.affine_grid(Affine_inv[0],
                                        [batch_size, 1, input_dict['img'].shape[-2], input_dict['img'].shape[-1]])
            bg_mask = F.grid_sample(input=bg_mask_inp, grid=grid_uncrop, padding_mode='zeros')

            bg_predicted_init = input_dict['img'] * (1 - bg_mask)
            if input_dict['img'].shape[-1] < 600:
                bg_predicted_inp = bg_predicted_init.clone()
            # forward pass for bg network
            for i in range(self.bg_recursion):
                bg_predicted_init = self.bg_unet.forward(bg_predicted_init)
            input_img_crop = torch.zeros(bg_predicted_init.shape).cuda()
            inpainting_crop = torch.zeros(bg_predicted_init.shape).cuda()
            inpainting_size = torch.zeros(bg_predicted_init.shape[0]).cuda()

            for p in range(affine_matrix_crop_multi.shape[0]):
                bg_predicted = bg_predicted_init * bg_mask + input_dict['img'] * (1 - bg_mask)
                input_img_crop = input_dict['img'] * bg_mask
                inpainting_crop = bg_predicted_init * bg_mask
                inpainting_size = bg_mask.sum(-1).sum(-1).sum(-1)

        ###############################################
        # undo spatial transformer
        if self.spatial_transformer:
            output_imgs = []
            output_masks = []
            mask_densities = []
            input_img_size = input_dict['img'].squeeze().size()
            for j in range(ST_size):
                grid_uncrop = F.affine_grid(affine_matrix_uncrop_multi[j], input_img_size)
                weights_j = similarity_matrix_normalized[j].unsqueeze(-1).unsqueeze(-1).unsqueeze(
                    -1)  # before, leads to identity flip after rotation
                output_img_crop_j = torch.sum(output_img_crop * weights_j.expand_as(output_img_crop), dim=0)
                output_img_warped = F.grid_sample(output_img_crop_j, grid_uncrop, padding_mode='border')
                output_imgs.append(output_img_warped)

                if self.masked_blending:
                    mask = torch.sum(output_mask_crop * weights_j.expand_as(output_mask_crop), dim=0)
                else:
                    mshape = list(output_img_crop[j].size())
                    mshape[1] = 1  # change from 3 channel image to 1 channel mask
                    if unet.bump_function is not None:
                        mask = unet.bump_function.unsqueeze(0).unsqueeze(0).expand(mshape)
                    else:
                        mask = torch.ones(mshape).cuda()
                if self.normalize_mask_density:
                    mask_density = torch.sum(torch.sum(mask, dim=2, keepdim=True), dim=3, keepdim=True)
                    mask_densities.append(mask_density)
                output_mask_warped = F.grid_sample(mask, grid_uncrop, padding_mode='zeros')
                if self.scale_mask_max_to_1:  # maximize response map V2 (after unwarping, ensures to only consider mask pixels inside the output image, not those cropped)
                    mask_max, max_index = torch.max(output_mask_warped.view(batch_size, -1), dim=1)
                    output_mask_warped = output_mask_warped / (
                    0.0001 + mask_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                output_masks.append(output_mask_warped)
            output_imgs = torch.stack(output_imgs)
            output_masks = torch.stack(output_masks)

            if self.predict_transformer_depth:
                # sqrt_2 = float(np.sqrt(2))
                # sqrt_2_by_pi = float(np.sqrt(2 / np.pi))
                opacity_factor = 1  # opacity/float(np.sqrt(2 / np.pi)) # if sqrt(2) etc is included a factor of five was good
                # Note, the following unsqueezing puts different blob positions in dim=0 and sample positoins in dim=1 (they can be the same)
                c = opacity_factor * output_masks.unsqueeze(1)
                # TODO: is this expand needed? Broadcasting should work..
                mu = ST_depth.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # Note, the sampling position is taken relative to the center of the Gaussians
                mu_offset = 0
                s = ST_depth.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) - mu_offset
                accumulated_density_indi = c * (torch.erf(s - mu) + 1)  # simplified with c = opacity*sqrt_pi/2
                accumulated_density_sum = torch.sum(accumulated_density_indi,
                                                    dim=0)  # sum across all blobs (dim=0), individual for each sample point (dim=1)


                accumulated_density_bg_indi = 2 * opacity_factor * output_masks
                accumulated_density_bg_sum = torch.sum(accumulated_density_bg_indi,
                                                       dim=0)  # sum across all blobs (dim=0)

                transmittance = torch.exp(-accumulated_density_sum)
                transmittance_bg = torch.exp(-accumulated_density_bg_sum)
                # TODO, in principle there should be a factor 2/sqrt(pi) here, to model the emmisiion of a gaussian density, but it does not matter due to the used normalization
                #   emission = output_masks*2/float(np.sqrt(2))
                emission = output_masks
                radiance = transmittance * emission
                # note, assuming the emission of the bg is 1 at infinity, hence:
                radiance_bg = transmittance_bg
                radiance_fg_analytic = 1 - radiance_bg
                radiance_fg_approx = torch.sum(radiance, dim=0)
                normalization_facor = radiance_fg_analytic / (0.0001 + radiance_fg_approx)
                radiance = radiance * normalization_facor  # normalized

                # debugging
                radiance_fg_fixed = radiance_fg_approx * normalization_facor
                if (radiance_fg_fixed > 1).any().item():
                    v = torch.max(radiance_fg_fixed)
                    print('WARNING: radiance normalized to above 1, max =', v)
                    IPython.embed()

                max_depth, _ = torch.max(ST_depth, dim=0)
                bg_dist = max_depth + 1  # put it 1*sigma=1 behind the last point
                depth_map = (bg_dist.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * radiance_bg
                             + torch.sum(ST_depth.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * radiance,
                                         dim=0)).data.cpu()
                eps = 0.0001
            else:
                # sum over STs and ensure that the sum is less than 1. Otherwise scale each mask down respectively
                mask_sum = torch.sum(output_masks, keepdim=True, dim=0)
                radiance = output_masks / torch.clamp(mask_sum, min=1, max=9999)

            # now done properly with background transmittance
            # if self.scale_mask_max_to_1:  # maximize radiance to compensate for self occlusion
            #    radiance_max, max_index = torch.max(torch.sum(radiance, dim=0).view(batch_size,-1), dim=1) # normalize so that the sum of the two should reach one at some point, compute maximum across all pixels
            #    radiance_normalized = radiance / (0.0001 + radiance_max.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            radiance_normalized = radiance

            ######## normalize so that having one mask with a low density is peanalized (density is enforced to be the same) #######
            # Note, this has to happen after the unwarping and radiance computation/normalization, first, because we unwarp separately per crop id.
            # Second, because we normalize height on the full crop
            if self.normalize_mask_density:
                mask_densities = torch.stack(
                    mask_densities)  # note, this density is normalized by the crop size (not computed in absolute coords)
                min_density, _ = torch.min(mask_densities, dim=0, keepdim=True)
                eps = 0.0001
                radiance_normalized = radiance_normalized * (min_density / (eps + mask_densities))

            # weighted sum over spatial transformer (dimension 0)
            if self.reconstruct_type == 'fg':
                output_img = torch.sum(radiance_normalized * output_imgs, dim=0) \
                             + (1 - torch.sum(radiance_normalized, dim=0)) * input_dict['bg']
            elif self.reconstruct_type == 'full':
                output_img = torch.sum(radiance_normalized * output_imgs, dim=0) \
                             + (1 - torch.sum(radiance_normalized, dim=0)) * bg_predicted
            elif self.reconstruct_type == 'bg':
                output_img = bg_predicted
            # undo potential shuffling before color assignment
            if self.shuffle_crops:
                radiance_colored = ST_split(
                    torch.index_select(ST_flatten(radiance_normalized.cpu()), dim=0, index=shuffled_crops_inv.cpu()))
            else:
                radiance_colored = radiance_normalized.cpu()
            # move spatial transformer dimension to color dimension, to get semantic segmentation result
            # [0] to remove singleton dim which is now ST, don't use squeeze as it might remove ST_size==1 case too
            radiance_colored = radiance_colored.transpose(0, 2)[0]

            if radiance_colored.shape[1] == 1 or radiance_colored.shape[1] == 3:
                pass
            elif radiance_colored.shape[1] == 2:
                radiance_colored = radiance_colored.repeat([1, 2, 1, 1])[:, :3, :, :]
                radiance_colored[:, 2, :, :] = 0
            elif radiance_colored.shape[1] > 3 and radiance_colored.shape[1] % 2 == 0:
                s = radiance_colored.shape
                radiance_colored = torch.sum(radiance_colored.reshape([s[0], 2, -1, s[2], s[3]]), dim=2)
            else:  # TODO: do some coloring for odd numbers
                radiance_colored = torch.sum(radiance_normalized, dim=0)
            output_mask_combined = radiance_colored  # drop more than three dimensions
        else:
            if self.masked_blending:
                bg_crop = input_dict['bg_crop']
                output_img = output_mask_crop * output_img_crop + (1 - output_mask_crop) * bg_crop
            else:
                output_img = output_img_crop

        # output stage
        output_dict_all = {  # '3D': ST_flatten(output_pose.transpose(1,0).contiguous()),
            'img_crop': ST_flatten(output_img_crop.transpose(1, 0).contiguous()),
        # transpose to make crops from the same image neighbors
            'img': output_img,
            'shuffled_pose': shuffled_pose,
            'shuffled_pose_inv': shuffled_pose_inv,
            'shuffled_appearance': shuffled_appearance,
            'latent_3d': ST_flatten(latent_3d),
            'latent_3d_rotated': latent_3d_rotated,
            'latent_fg': latent_fg,
            'cam2cam': cam2cam}  # , 'shuffled_appearance' : xxxx, 'shuffled_pose' : xxx}
        if self.spatial_transformer:
            # Undo the shuffling, otherwise the prior is not independent per crop
            if self.shuffle_crops:
                affine_matrix_crop_multi = ST_split(
                    torch.index_select(ST_flatten(affine_matrix_crop_multi), dim=0, index=shuffled_crops_inv))
                # note, needs to transpose because of ST_size x ST_size x batch_size
                sim_flat = ST_flatten(similarity_matrix_normalized.transpose(1, 2).contiguous())
                sim_flat = torch.index_select(sim_flat, dim=0, index=shuffled_crops_inv)
                similarity_matrix_normalized = ST_split(sim_flat).transpose(1, 2)

            output_dict_all['spatial_transformer'] = affine_matrix_crop_multi
            output_dict_all['spatial_transformer_raw'] = affine_matrix_crop_raw
            output_dict_all['radiance_normalized'] = radiance_normalized
            output_dict_all['bg_crop'] = ST_flatten(
                ST_split(input_dict_cropped['bg_crop']).transpose(1, 0).contiguous())
            output_dict_all['spatial_transformer_img_crop'] = ST_flatten(
                ST_split(input_dict_cropped['img_crop']).transpose(1, 0).contiguous())
            output_dict_all['img_downscaled'] = input_img_downscaled
            output_dict_all['similarity_matrix'] = similarity_matrix_normalized
            if self.predict_transformer_depth:
                if self.shuffle_crops:
                    ST_depth = ST_split(torch.index_select(ST_flatten(ST_depth), dim=0, index=shuffled_crops_inv))
                output_dict_all['ST_depth'] = ST_depth
                output_dict_all['depth_map'] = depth_map

            if unet.bump_function is not None:
                output_dict_all['smooth_mask'] = unet.bump_function

            if self.masked_blending:
                output_dict_all['blend_mask'] = output_mask_combined  # output_mask_warped

        if self.masked_blending:
            output_dict_all['blend_mask_crop'] = ST_flatten(output_mask_crop.transpose(1, 0).contiguous())
            output_dict_all['blend_mask_crop_before_scale'] = ST_flatten(
                output_mask_crop_before_scale.transpose(1, 0).contiguous())
        output_dict = {}

        for key in self.output_types:
            if key != 'bg':
                output_dict[key] = output_dict_all[key]

        if self.estimate_background:
            if input_dict['img'].shape[-1] < 600:
                output_dict['bg_inp'] = bg_predicted_inp
            output_dict['bg'] = bg_predicted
        else:
            if input_dict['img'].shape[-1] < 600:
                output_dict['bg_inp'] = input_dict['img']
            output_dict['bg'] = input_dict['bg']
        output_dict['input_img_crop'] = input_img_crop
        output_dict['output_img_crop'] = output_dict['img'] * bg_mask
        output_dict['inpainting_crop'] = inpainting_crop
        output_dict['inpainting_size'] = inpainting_size
        output_dict['grid_matrix'] = self.grid_matrix.contiguous().view(batch_size, 1, self.grid_matrix.shape[-2], self.grid_matrix.shape[-1])
        output_dict['cell_map'] = self.cell_map.contiguous().view(batch_size, 1, self.cell_map.shape[-2], self.cell_map.shape[-1])
        output_dict['confidence'] = self.confidence
        output_dict['confidence_before_softmax'] = self.confidence_before_softmax
        output_dict['proposal'] = self.proposal
        output_dict['confidence_source_view'] = torch.ones(self.confidence_shape).cuda()
        output_dict['proposal_source_view'] = torch.ones(self.confidence_shape).cuda()
        # output_dict['confidence_source_view'] = torch.index_select(self.confidence, dim=0, index=shuffled_pose_inv)
        # output_dict['proposal_source_view'] = torch.index_select(self.proposal, dim=0, index=shuffled_pose_inv)
        output_dict['mask_prior_loss'] = mask_prior_loss
        output_dict['blend_mask_crop_before_scale'] = ST_flatten(
            output_mask_crop_before_scale.transpose(1, 0).contiguous())
        output_dict['fg'] = output_imgs
        if self.training:
            output_dict['volume_to_grid_view'] = self.volume_to_grid_view
            output_dict['volume_to_grid'] = self.volume_to_grid
            output_dict['voxel_confidence'] = self.voxel_confidence
            output_dict['vind']  = self.vind
            output_dict['bindc'] = self.bindc
            output_dict['bindr'] = self.bindr
            output_dict['subbatch_size'] = self.subbatch_size
            output_dict['matching_list'] = self.matching_list
            output_dict['top_k'] = self.top_k


        for i in range(len(output_dict['confidence_source_view'])):
            if shuffled_pose_inv[i] == i:  # identity?
                output_dict['confidence_source_view'][i] = 1
                output_dict['proposal_source_view'][i] = 1

        output_dict['grid_size'] = self.C
        output_dict['cell_center_x'] = self.selected_cell_centers_x
        output_dict['cell_center_y'] = self.selected_cell_centers_y
        output_dict['scale_x'] = self.scale_x
        output_dict['scale_y'] = self.scale_y
        output_dict['offset_x'] = self.offset_x
        output_dict['offset_y'] = self.offset_y
        output_dict['all_offsets'] = self.all_offsets
        output_dict['all_scales'] = self.all_scales
        output_dict['reconstruct_type'] = self.reconstruct_type
        output_dict['training'] = self.training
        if self.reconstruct_type == 'bg':
            output_dict['margin'] = (self.margin_h, self.margin_w)
            output_dict['spatial_transformer'] = blank_locations
        if self.offset_consistency_type == 'loss':
            # If the type is 'loss' not 'projection'
            output_dict['pairwise_dist'] = self.pairwise_dist

        return output_dict

    def forward_pose(self, input_dict):
        assert not self.shuffle_crops

        if 'img' in input_dict.keys():
            batch_size = input_dict['img'].size()[0]
        else:
            batch_size = input_dict['img_crop'].size()[0]
        ST_size = self.ST_size

        def ST_flatten(batch):
            shape_orig = batch.shape
            # assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = [ST_size * batch_size] + list(shape_orig[2:])
            return batch.view(shape_new)

        def ST_split(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size * batch_size
            shape_new = [ST_size, batch_size] + list(shape_orig[1:])
            return batch.view(shape_new)

        def features_flatten(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = list(shape_orig[:2]) + [-1]
            return batch.view(shape_new)

        def features_split3D(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = list(shape_orig[:2]) + [-1, 3]
            return batch.view(shape_new)

        ###############################################
        # spatial transformer
        if self.spatial_transformer:
            input_dict_cropped, input_img_downscaled, ST_depth, affine_matrix_crop_multi, affine_matrix_uncrop_multi, affine_matrix_crop_raw = self.forward_detector(
                input_dict)
        else:
            input_dict_cropped = input_dict  # fallback to using crops from dataloader

        ###############################################
        # encoding stage
        has_fg = hasattr(self, "to_fg")
        output = self.encoder.forward(input_dict_cropped)['latent_3d']
        if has_fg:
            latent_fg = output[:, :self.dimension_fg]  # .contiguous().clone() # TODO
            latent_fg = ST_split(latent_fg)
        latent_3d = output[:, self.dimension_fg:self.dimension_fg + self.dimension_3d]

        ###############################################
        # decoding stage
        pose_3d = self.pose_decoder({'latent_3d': latent_3d})['3D']
        pose_3d = ST_split(pose_3d)

        # flip predicted poses to be sorted left to right (mostly for display purposes)
        if ST_size > 1:
            crop_x = affine_matrix_crop_multi[:, :, 0, 2]
            is_left_right = ((torch.sign(crop_x[1] - crop_x[0]) + 1) / 2).byte()
            not_left_right = 1 - is_left_right
            mask = torch.stack([is_left_right, not_left_right]).unsqueeze(-1)

            pose_3d_left = mytorch.transposed_mask_select(pose_3d, mask, (0, 1))
            pose_3d_right = mytorch.transposed_mask_select(pose_3d, 1 - mask, (0, 1))

            pose_3d = torch.stack([pose_3d_left, pose_3d_right]).view(pose_3d.shape)

        ###############################################
        # 3D pose stage (parallel to image decoder)
        output_dict_all = {'3D': ST_flatten(pose_3d.transpose(1, 0).contiguous()),
                           'latent_3d': latent_3d,
                           #                           'latent_3d_rotated': latent_3d_rotated,
                           'latent_fg': latent_fg,
                           #                           'cam2cam': cam2cam
                           }
        if self.spatial_transformer:
            output_dict_all['spatial_transformer'] = affine_matrix_crop_multi
            output_dict_all['spatial_transformer_img_crop'] = ST_flatten(
                ST_split(input_dict_cropped['img_crop']).transpose(1, 0).contiguous())
            output_dict_all['spatial_transformer_raw'] = affine_matrix_crop_raw
            output_dict_all['img_downscaled'] = input_img_downscaled
            if self.predict_transformer_depth:
                output_dict_all['ST_depth'] = ST_depth

        output_dict = {}
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict
