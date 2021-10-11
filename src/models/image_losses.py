import numpy as np

import torch

from PlottingUtil import util as utils_plt
from util import util as utils_generic
from datasets import transforms as transforms_aug

import matplotlib.pyplot as plt
import IPython
from models import resnet_low_level

class SobelCriterium(torch.nn.Module):
    """
    Approximates horizontal and vertical gradients with the Sobel operator and puts a criterion on these gradient estimates.
    """
    def __init__(self, criterion, weight=1, key='img_crop'):
        super(SobelCriterium, self).__init__()
        self.weight = weight
        self.key = key
        self.criterion = criterion

        kernel_x = np.array([[1, 0, -1], [2,0,-2],  [1, 0,-1]])
        kernel_y = np.array([[1, 2,  1], [0,0, 0], [-1,-2,-1]])

        channels = 3
        kernel_size = 3
        self.conv_x = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.conv_x.weight = torch.nn.Parameter(torch.from_numpy(kernel_x).float().unsqueeze(0).unsqueeze(0).expand([channels,channels,kernel_size,kernel_size]))
        self.conv_x.weight.requires_grad = False
        self.conv_x.cuda()
        self.conv_y = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = torch.nn.Parameter(torch.from_numpy(kernel_y).float().unsqueeze(0).unsqueeze(0).expand([channels,channels,kernel_size,kernel_size]))
        self.conv_y.weight.requires_grad = False
        self.conv_y.cuda()
        
    def forward(self, pred_dict, label_dict):
        label = label_dict[self.key]
        pred  = pred_dict [self.key]

        pred_x = self.conv_x.forward(pred)
        pred_y = self.conv_y(pred)
        label_x = self.conv_x(label)
        label_y = self.conv_y(label)

        return self.weight * (self.criterion(pred_x, label_x) + self.criterion(pred_y, label_y))

class ImageNetCriterium(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """
    def __init__(self, criterion, weight=1, key='img_crop', do_maxpooling=True):
        super(ImageNetCriterium, self).__init__()
        self.weight = weight
        self.key    = key
        self.criterion = criterion

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels = 3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.cuda()
        
    def forward(self, pred_dict, label_dict):
        label = label_dict[self.key]
        pred  = pred_dict [self.key]

        preds_x  = self.net(pred)
        labels_x = self.net(label) #.detatch()
        
        losses = [self.criterion(p, l) for p,l in zip(preds_x,labels_x)]

        return self.weight * sum(losses) / len(losses)


class ImageNetCriteriumUnsupPretrain(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """

    def __init__(self, criterion, weight=1, key='img_crop', do_maxpooling=True):
        super(ImageNetCriteriumUnsupPretrain, self).__init__()
        self.weight = weight
        self.key = key
        self.criterion = criterion
        print("RESNET 18 USEFUL")
        self.net = resnet_low_level.resnet18(pretrained=True, num_channels=3, do_maxpooling=do_maxpooling, path_unsup='../pretrained/lemniscate_resnet18.pth.tar')
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.cuda()

    def forward(self, pred_dict, label_dict):
        label = label_dict[self.key]
        pred = pred_dict[self.key]

        preds_x = self.net(pred)
        labels_x = self.net(label)  # .detatch()

        losses = [self.criterion(p, l) for p, l in zip(preds_x, labels_x)]

        return self.weight * sum(losses) / len(losses)


class ImageNetInpaintingCriterium(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """

    def __init__(self, criterion,  weight=1,gt_key='img', pred_key='bg', crop_size_key='inpainting_size', do_maxpooling=True):
        super(ImageNetInpaintingCriterium, self).__init__()
        self.weight = weight
        self.criterion = criterion
        self.gt_key = gt_key
        self.pred_key = pred_key
        self.crop_size_key = crop_size_key

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels=3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.cuda()

    def forward(self, pred_dict, label_dict):
        #label = label_dict[self.gt_key] # USE THIS!!!
        label = pred_dict[self.gt_key]
        pred = pred_dict[self.pred_key]

        preds_x = self.net(pred)
        labels_x = self.net(label)  # .detatch()

        losses = [((self.criterion(p, l).mean()) / pred_dict[self.crop_size_key]) for p, l in zip(preds_x, labels_x)]

        return self.weight * (sum(losses) / len(losses)).mean()


class ImageNetCriterium_Sampling(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """

    def __init__(self, criterion, weight=1, key='img_crop', confidence_key='confidence', proposal_key='importance', do_maxpooling=True, train=False):
        super(ImageNetCriterium_Sampling, self).__init__()
        self.weight = weight
        self.key = key
        self.confidence_key = confidence_key
        self.proposal_key = proposal_key
        self.confidence_key_source = self.confidence_key + '_source_view'
        self.proposal_key_source = self.proposal_key + '_source_view'
        self.criterion = criterion
        self.train = train

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels=3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.cuda()

    def forward(self, pred_dict, label_dict):
        label = label_dict[self.key]
        pred = pred_dict[self.key]

        preds_x = self.net(pred)
        labels_x = self.net(label)  # .detatch()

        losses = [torch.pow(p-l, 2).mean(-1).mean(-1).mean(-1) for p, l in zip(preds_x, labels_x)]
        #return self.weight * ((pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key_source].data / pred_dict[self.proposal_key_source].data) * (sum(losses) / len(losses))).mean()
        #return self.weight * ((pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data) * (sum(losses) / len(losses))).mean()

        return self.weight * ((pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data) *  (sum(losses) / len(losses))).mean()

        #return self.weight * ((pred_dict[self.confidence_key] / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key_source] / pred_dict[self.proposal_key_source].data) * (sum(losses) / len(losses))).mean()


class ImageNetCriterium_Sampling_Unsup_Pretrain(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """

    def __init__(self, criterion, weight=1, key='img_crop', confidence_key='confidence', proposal_key='importance', do_maxpooling=True, train=False):
        super(ImageNetCriterium_Sampling_Unsup_Pretrain, self).__init__()
        self.weight = weight
        self.key = key
        self.confidence_key = confidence_key
        self.proposal_key = proposal_key
        self.confidence_key_source = self.confidence_key + '_source_view'
        self.proposal_key_source = self.proposal_key + '_source_view'
        self.criterion = criterion
        self.train = train

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels=3, do_maxpooling=do_maxpooling, path_unsup='../pretrained/lemniscate_resnet18.pth.tar')
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.cuda()

    def forward(self, pred_dict, label_dict):
        label = label_dict[self.key]
        pred = pred_dict[self.key]

        preds_x = self.net(pred)
        labels_x = self.net(label)  # .detatch()

        losses = [torch.pow(p-l, 2).mean(-1).mean(-1).mean(-1) for p, l in zip(preds_x, labels_x)]
        #return self.weight * ((pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key_source].data / pred_dict[self.proposal_key_source].data) * (sum(losses) / len(losses))).mean()
        return self.weight * ((pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data) * (sum(losses) / len(losses))).mean()

        #return self.weight * ((pred_dict[self.confidence_key] / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key_source] / pred_dict[self.proposal_key_source].data) * (sum(losses) / len(losses))).mean()


class ImageNetCriteriumExp_BG(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """

    def __init__(self, criterion, weight=1, key='img_crop', aux_key='bg',  do_maxpooling=True):
        super(ImageNetCriteriumExp_BG, self).__init__()
        print("ImageNetExpBG")
        self.weight = weight
        self.key = key
        self.aux_key = aux_key
        self.criterion = criterion

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels=3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.cuda()

    def forward(self, pred_dict, label_dict):
        label = pred_dict[self.aux_key]
        pred = pred_dict[self.key]

        preds_x = self.net(pred)
        labels_x = self.net(label)  # .detatch()

        losses = [torch.exp(-1*torch.pow((p-l), 2)).mean() for p, l in zip(preds_x, labels_x)]

        return self.weight * (sum(losses) / len(losses))

class ImageNetCriterium_BG(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """

    def __init__(self, criterion, weight=1, key='img_crop', aux_key='bg', do_maxpooling=True):
        super(ImageNetCriterium_BG, self).__init__()
        print("ImageNetBG")
        self.weight = weight
        self.key = key
        self.aux_key = aux_key
        self.criterion = criterion

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels=3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.cuda()

    def forward(self, pred_dict, label_dict):
        label = pred_dict[self.aux_key]
        pred = pred_dict[self.key]

        preds_x = self.net(pred)
        labels_x = self.net(label)  # .detatch()

        losses = [torch.exp(-1* torch.pow((p-l), 2)).mean() for p, l in zip(preds_x, labels_x)]

        return self.weight * (sum(losses) / len(losses))


class ImageNetCriteriumBG_Sampling(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """

    def __init__(self, criterion, weight=1, key='img_crop', aux_key='bg', confidence_key='confidence', proposal_key='importance', input_crop_key='input_crop_key', crop_key='crop_key', crop_size_key='crop_size_key', do_maxpooling=True, train=False):
        super(ImageNetCriteriumBG_Sampling, self).__init__()
        print("ImageNetExpBG")
        self.weight = weight
        self.key = key
        self.aux_key = aux_key
        self.confidence_key = confidence_key
        self.input_crop_key = input_crop_key
        self.crop_key = crop_key
        self.crop_size_key = crop_size_key
        self.proposal_key = proposal_key
        self.confidence_key_source = self.confidence_key + '_source_view'
        self.proposal_key_source = self.proposal_key + '_source_view'
        self.criterion = criterion
        self.train = train

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels=3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.cuda()

    def forward(self, pred_dict, label_dict):
        # pred = pred_dict[self.aux_key]
        # label = label_dict[self.key]

        pred = pred_dict[self.crop_key]
        label = pred_dict[self.input_crop_key]

        preds_x = self.net(pred)
        labels_x = self.net(label)  # .detatch()

        losses = [(-1*torch.pow(p.data-l.data, 2)).mean(-1).mean(-1).mean(-1) for p, l in zip(preds_x, labels_x)]
        return self.weight * ((pred_dict[self.confidence_key] / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key_source] / pred_dict[self.proposal_key_source].data) * (sum(losses) / pred_dict[self.crop_size_key])).mean()


class ImageNetCriteriumExp_BG_Voting(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """

    def __init__(self, criterion, weight=1, key='img_crop', aux_key='bg', weight_key='confidence', grid_size_key='grid_size', do_maxpooling=True):
        super(ImageNetCriteriumExp_BG_Voting, self).__init__()
        print("ImageNetExpBG")
        self.weight = weight
        self.key = key
        self.aux_key = aux_key
        self.weight_key = weight_key
        self.grid_size_key = grid_size_key
        self.criterion = criterion

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels=3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.cuda()

    def forward(self, pred_dict, label_dict):
        label = pred_dict[self.aux_key]
        pred = pred_dict[self.key]

        preds_x = self.net(pred)
        preds_x = self.net(pred)
        labels_x = self.net(label)  # .detatch()

        losses = [torch.pow(p.data-l.data, 2).mean(-1).mean(-1).mean(-1) for p, l in zip(preds_x, labels_x)]

        return self.weight * (pred_dict[self.grid_size_key] * pred_dict[self.weight_key] * torch.exp(-1*(sum(losses)))).mean()