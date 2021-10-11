import numpy as np
import numpy as np
import numpy.linalg as la
import sys
sys.path.insert(0, '../')

import torch
import os
from PlottingUtil import util as utils_plt
from PlottingUtil import skeletons
from src.util import util as utils_generic
from datasets import transforms as transforms_aug
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import IPython
import scipy.misc

class PreApplyCriterionList(torch.nn.Module):
    """
    Wraps a loss operating on tensors into one that processes lists of labels and predictions
    """
    def __init__(self, criterions_single, output_indices=None, label_indices=None, sum_losses=True, loss_weights=None):
        super(PreApplyCriterionList, self).__init__()
        self.criterions_single = criterions_single
        self.output_indices = output_indices or list(range(len(criterions_single)))
        self.label_indices = label_indices or list(range(len(criterions_single)))
        self.sum_losses = sum_losses
        self.loss_weights = loss_weights
        assert len(self.criterions_single) == len(self.output_indices) and \
               len(self.output_indices) == len(self.label_indices), \
            "Loss lenghts are not the same: Loss[{}], Output[{}], Lables[{}]".format(len(self.criterions_single),
                                                                                     len(self.output_indices),
                                                                                     len(self.label_indices))

    def forward(self, pred_list, label_list):
        """
        The loss is computed as the sum of all the loss values
        :param pred_list: List containing the predictions
        :param label_list: List containing the labels
        :return: The sum of all the loss values computed
        """
        losslist = []
        for criterion_idx, criterion_single in enumerate(self.criterions_single):
            if not criterion_single:
                # For networks that have multiple outputs we might want to only use some of them.
                # By putting the loss to None to ensure that we only compute the losses that we are interested in.
                continue
            if self.label_indices[criterion_idx] == -1:
                label_i = label_list
            else:
                label_i = label_list[self.label_indices[criterion_idx]]
            if self.output_indices[criterion_idx] == -1:
                pred_i = pred_list
            else:
                pred_i = pred_list[self.output_indices[criterion_idx]]
            loss_i = criterion_single(pred_i, label_i)
            if self.loss_weights is not None:
                loss_i = loss_i * self.loss_weights[criterion_idx]
            losslist.append(loss_i)

        if self.sum_losses:
            return sum(losslist)
        else:
            return losslist
        
class PreApplyCriterionListDict(torch.nn.Module):
    """
    Wraps a loss operating on tensors into one that processes dict of labels and predictions
    """
    def __init__(self, criterions_single, sum_losses=True, loss_weights=None):
        super(PreApplyCriterionListDict, self).__init__()
        self.criterions_single = criterions_single
        self.sum_losses = sum_losses
        self.loss_weights = loss_weights

    def forward(self, pred_dict, label_dict):
        """
        The loss is computed as the sum of all the loss values
        :param pred_dict: List containing the predictions
        :param label_dict: List containing the labels
        :return: The sum of all the loss values computed
        """
        losslist = []
        for criterion_idx, criterion_single in enumerate(self.criterions_single):
            loss_i = criterion_single(pred_dict, label_dict)
            if self.loss_weights is not None:
                loss_i = loss_i * self.loss_weights[criterion_idx]
            losslist.append(loss_i)

        if self.sum_losses:
            return sum(losslist)
        else:
            return losslist

class PreApplyNetworkList(torch.nn.Module):
    """
    Allows the network to accept a list as input.
    (I guess that it was supposed to be so that it can take multiple inputs but looking at the code it only reads the
    first one)
    """
    def __init__(self, network):
        super(PreApplyNetworkList, self).__init__()
        self.network = network
        # TODO : Create a model class and move this function there?

    def forward(self, input_list):
        # TODO : Why have it as a list since it only reads the first element?
        return self.network(input_list[0])

class PreApplyNetworkDict(torch.nn.Module):
    """
    Allows the network to accept a list as input although originally it assumes a single tensor.
    """
    def __init__(self, network, key='img_crop'):
        super(PreApplyNetworkList, self).__init__()
        self.network = network
        self.key = key
        # TODO : Create a model class and move this function there?

    def forward(self, input_dict):
        # TODO : Why have it as a list since it only reads the first element?
        return self.network(input_dict[self.key])

class PostWrapPredToList(torch.nn.Module):
    """
    Wraps a network that yields a tensor into one that outputs lists of predictions.
    Call this function if the output of the network is not already a list. This is to ensure that no matter the number
    of outputs the network has it always outputs a list and that it is consistent in this regard.
    """
    def __init__(self, single):
        super(PostWrapPredToList, self).__init__()
        self.single = single
        # TODO : Create a model class and move this function there?

    def forward(self, input_val):
        return [self.single(input_val)]

class PostWrapPredToDict(torch.nn.Module):
    """
    Wraps a network that yields a tensor into one that output dict of predictions.
    Call this function if the output of the network is not already a list. This is to ensure that no matter the number
    of outputs the network has it always outputs a dist and that it is consistent in this regard.
    """
    def __init__(self, single, key='3D'):
        super(PostWrapPredToDict, self).__init__()
        self.single = single
        self.key = key
        # TODO : Create a model class and move this function there?

    def forward(self, input_val):
        return {self.key: self.single(input_val)}

class SelectSingleLabel(torch.nn.Module):
    """
    Apply a loss 'single' on a single label 'key'    
    """
    def __init__(self, single, key='3D'):
        super(SelectSingleLabel, self).__init__()
        self.single = single
        self.key = key

        # TODO : Create a model class and move this function there?

    def forward(self, pred_dict, label_dict):

        return self.single(pred_dict[self.key], label_dict[self.key])



class InpaintingPixelLoss(torch.nn.Module):
    """
    Apply a loss 'single' on a single label 'key'    
    """
    def __init__(self, weight, gt_key='img', pred_key='bg', crop_size_key='inpainting_size'):
        super(InpaintingPixelLoss, self).__init__()
        self.weight = weight
        self.gt_key = gt_key
        self.pred_key = pred_key
        self.crop_size_key = crop_size_key

    def forward(self, pred_dict, label_dict):

        return self.weight * (torch.pow((pred_dict[self.pred_key]- label_dict[self.gt_key]), 2).mean(1).sum(1).sum(1) / pred_dict[self.crop_size_key]).mean()

class InpaintingPixelL1Loss(torch.nn.Module):
    """
    Apply a loss 'single' on a single label 'key'    
    """
    def __init__(self, weight, gt_key='img', pred_key='bg', crop_size_key='inpainting_size'):
        super(InpaintingPixelL1Loss, self).__init__()
        self.weight = weight
        self.gt_key = gt_key
        self.pred_key = pred_key
        self.crop_size_key = crop_size_key

    def forward(self, pred_dict, label_dict):
        #pred_dict[self.gt_key] should be label_dict[self.gt_key]
        return self.weight * (torch.abs(pred_dict[self.pred_key]- pred_dict[self.gt_key]).mean(1).sum(1).sum(1) / pred_dict[self.crop_size_key]).mean()


class SegMaskFire(torch.nn.Module):
    def __init__(self, weight, percent):
        super(SegMaskFire, self).__init__()
        self.weight = weight
        self.percent = percent
        self.key = 'radiance_normalized'

    def forward(self, pred_dict, label_dict):
        if self.key == 'radiance_normalized':
            pred_dict[self.key] = pred_dict[self.key].squeeze(0)
        seg_mask_fire_loss = 0
        #level = torch.zeros((pred_dict[self.key].shape[0])).cuda()
        for im in range(pred_dict[self.key].shape[0]):
            #level[im] = pred_dict[self.key][im].mean() - self.percent
            if pred_dict[self.key][im].mean() < self.percent:
                seg_mask_fire_loss += 2*self.percent - torch.mean(torch.abs(pred_dict[self.key][im]))
            else:
                seg_mask_fire_loss += torch.mean(torch.abs(pred_dict[self.key][im]))
        #pred_dict['seg_mask_fire_level'] = level
        return self.weight * seg_mask_fire_loss / pred_dict[self.key].shape[0]

class VShapeSegMaskFire(torch.nn.Module):
    def __init__(self, weight, percent, key='radiance_normalized'):
        super(VShapeSegMaskFire, self).__init__()
        self.weight = weight
        self.percent = percent
        self.key = key

    def forward(self, pred_dict, label_dict):
        if self.key == 'radiance_normalized':
            pred_dict[self.key] = pred_dict[self.key].squeeze(0)
        return self.weight * (torch.abs(torch.mean(torch.abs(pred_dict[self.key])) - self.percent) + self.percent)

class UShapeSegMaskFire(torch.nn.Module):
    def __init__(self, weight, lambda1, lambda2):
        super(UShapeSegMaskFire, self).__init__()
        self.weight = weight
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.key = 'radiance_normalized'

    def forward(self, pred_dict, label_dict):
        if self.key == 'radiance_normalized':
            pred_dict[self.key] = pred_dict[self.key].squeeze(0)
        seg_mask_fire_loss = 0
        for im in range(pred_dict[self.key].shape[0]):
            seg_mask_fire_loss += F.relu(-pred_dict[self.key][im].mean()+self.lambda1) + F.relu(pred_dict[self.key][im].mean() - self.lambda2)
        return self.weight * seg_mask_fire_loss / pred_dict[self.key].shape[0]

class TemporalROIFeatures(torch.nn.Module):
    def __init__(self, weight):  # type L2, dot
        super(TemporalROIFeatures, self).__init__()
        self.weight = weight

    def forward(self, pred_dict, label_dict):
        features = pred_dict['roi_bbox_features']
        if len(torch.unique(pred_dict['file_name_info'][:,2])) > 1:
            return torch.zeros(1).float().cuda().mean()
        if len(torch.unique(pred_dict['file_name_info'][:,0])) > 1:
            return torch.zeros(1).float().cuda().mean()
        return torch.abs(features - features.mean(axis=0)).mean()


class CentralizeSegMask(torch.nn.Module):
    def __init__(self, weight):  # type L2, dot
        super(CentralizeSegMask, self).__init__()
        self.weight = weight

    def forward(self, pred_dict, label_dict):
        seg_batch_size = pred_dict['blend_mask_crop'].shape[0]
        seg_mask_w, seg_mask_h = pred_dict['blend_mask_crop'].shape[-2:]
        coeff = pred_dict['blend_mask_crop'].squeeze(1)

        width_indices = torch.arange(1, seg_mask_w+1).unsqueeze(0).unsqueeze(-1).expand(seg_batch_size, seg_mask_w,1).expand(seg_batch_size, seg_mask_w, seg_mask_h).cuda()
        height_indices = torch.arange(1, seg_mask_h+1).unsqueeze(0).unsqueeze(0).expand(seg_batch_size, 1,seg_mask_h).expand(seg_batch_size, seg_mask_w, seg_mask_h).cuda()
        mass_x = coeff*width_indices
        center_x = mass_x.sum(-1).sum(-1) / coeff.sum(-1).sum(-1)
        mass_y = coeff*height_indices
        center_y = mass_y.sum(-1).sum(-1) / coeff.sum(-1).sum(-1)
        return torch.abs(center_x - seg_mask_w/2).mean() + torch.abs(center_y - seg_mask_h/2).mean()
        #for w in width_indices:
        #    mass_x

class CoherentMotionLoss(torch.nn.Module):
    def __init__(self, weight, type='L1'):  # type L2, dot
        super(CoherentMotionLoss, self).__init__()
        self.weight = weight
        self.type = type
        self.target_key = 'bin_optical_flow'
        self.seg_key = 'radiance_normalized'

    def forward(self, pred_dict, label_dict):

        if self.type == 'L1':
            return self.weight * (torch.abs(pred_dict[self.target_key] - pred_dict[self.seg_key].squeeze(0).squeeze(1))*pred_dict['enable_optical_flow']).mean()

        if self.type == 'L2':
            return self.weight * (torch.pow((pred_dict[self.target_key] - pred_dict[self.seg_key].squeeze(0).squeeze(1)) , 2)*pred_dict['enable_optical_flow']).mean()

class ContourAlignmentLoss(torch.nn.Module):
    """s
    Apply a loss 'single' on a single label 'key'    
    """

    def __init__(self, weight, type='L2'): #type L2, dot
        super(ContourAlignmentLoss, self).__init__()
        self.weight = weight
        self.target_key = 'smooth_target_edge'
        self.seg_key = 'seg_edge'
        self.type = type

    def forward(self, pred_dict, label_dict):

        target_x = pred_dict[self.target_key][0]
        target_y = pred_dict[self.target_key][1]
        target_edge = pred_dict[self.target_key][2]
        target_mag = ((torch.pow(pred_dict[self.target_key][0], 2) + torch.pow(pred_dict[self.target_key][1], 2)))
        target_dir = torch.atan2(pred_dict[self.target_key][1], pred_dict[self.target_key][0])

        seg_x = pred_dict[self.seg_key][0]
        seg_y = pred_dict[self.seg_key][1]
        seg_edge = pred_dict[self.seg_key][2]
        seg_mag = ((torch.pow(pred_dict[self.seg_key][0], 2) + torch.pow(pred_dict[self.seg_key][1], 2)))
        seg_dir = torch.atan2(pred_dict[self.seg_key][1], pred_dict[self.seg_key][0])

        cos = torch.nn.CosineSimilarity(dim=1)
        cos_sim = cos(seg_edge, target_edge)
        min_cos = 0.1

        if self.type == 'Dot':

            return self.weight * ((seg_mag*(min_cos + (1 - torch.pow(cos_sim, 2)))).mean())

        elif self.type == 'DotExp':
            return self.weight * ((seg_mag* (torch.exp(-(target_mag)))*(min_cos + (1 - torch.pow(cos_sim, 2)))).mean())


class ContourAlignmentLossOpticalFlow(torch.nn.Module):
    def __init__(self, weight, type='L2'): #type L2, dot
        super(ContourAlignmentLossOpticalFlow, self).__init__()
        self.weight = weight
        self.target_key = 'optical_flow_edge'
        self.seg_key = 'seg_edge'
        self.type = type

    def forward(self, pred_dict, label_dict):

        target_x = pred_dict[self.target_key][0]
        target_y = pred_dict[self.target_key][1]
        target_edge = pred_dict[self.target_key][2]
        target_mag = torch.sqrt((torch.pow(pred_dict[self.target_key][0], 2) + torch.pow(pred_dict[self.target_key][1], 2)))
        target_dir = torch.atan2(pred_dict[self.target_key][1], pred_dict[self.target_key][0])

        seg_x = pred_dict[self.seg_key][0]
        seg_y = pred_dict[self.seg_key][1]
        seg_edge = pred_dict[self.seg_key][2]
        seg_mag = ((torch.pow(pred_dict[self.seg_key][0], 2) + torch.pow(pred_dict[self.seg_key][1], 2)))
        seg_dir = torch.atan2(pred_dict[self.seg_key][1], pred_dict[self.seg_key][0])


        # if self.type == 'Angle':
        #     return self.weight * ( ((seg_mag * target_mag_r * torch.cos(2* (seg_dir - target_dir_r))).mean()) +  ((seg_mag * target_mag_g * torch.cos(2* (seg_dir - target_dir_g))).mean()) + ((seg_mag * target_mag_b * torch.cos(2* (seg_dir - target_dir_b))).mean()))/3

        if self.type == 'L2_Segx_Segy':
            return self.weight * (torch.pow(seg_x, 2) + torch.pow(seg_y, 2)).mean()

        elif self.type == 'L1_Segx_Segy':
            return self.weight * (torch.abs(seg_x) + torch.abs(seg_y)).mean()

        elif self.type == 'L1_Targetx_Segx_Targety_Segy':
            return self.weight * ((torch.abs(target_x - seg_x)).mean() + (torch.abs(target_y - seg_y)).mean() )

        # elif self.type == 'Exp_Magnitude':
        #     return self.weight * ((torch.abs(seg_x) * (torch.exp(- target_x))).mean() + (torch.abs(seg_y) *(torch.exp(- target_y))).mean())

        elif self.type == 'Trial':
            return self.weight * ((torch.abs(seg_x) * torch.exp((torch.abs(target_x) - torch.abs(seg_x)))) + (torch.abs(seg_y) * torch.exp((torch.abs(target_y) - torch.abs(seg_y))))).mean()

        elif self.type == 'Trial2':
            return self.weight * ((torch.abs(seg_x) * torch.exp(torch.abs((torch.abs(target_x) - torch.abs(seg_x))))) + (torch.abs(seg_y) * torch.exp(torch.abs((torch.abs(target_y) - torch.abs(seg_y)))))).mean()

        elif self.type == 'Trial3':
            return self.weight * ((torch.exp(target_y*seg_x - target_x*seg_y)) * (torch.abs(seg_x) + torch.abs(seg_y))).mean()

        elif self.type == 'Trial4':
            if any((torch.abs(seg_dir - target_dir).sum().unsqueeze(0)) == float('inf')):
                import pdb
                pdb.set_trace()
            if any(torch.isnan(torch.abs(seg_dir - target_dir).sum().unsqueeze(0))):
                import pdb
                pdb.set_trace()
            return self.weight *  ((torch.abs(seg_x) + torch.abs(seg_y)) * torch.abs(torch.sin((torch.abs(seg_dir - target_dir))))).mean()

        elif self.type == 'Cosine':
            # dot_product = (seg_x * target_x) + (seg_y * target_y)
            # cosine = dot_product / ((seg_mag) * (target_mag) + 0.0001)
            # #return self.weight * ((torch.abs(seg_x) + torch.abs(seg_y)) * (1-torch.pow(cosine, 2))).mean()
            # return self.weight * (1-torch.pow(cosine, 2)).mean()
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_sim = cos(seg_edge, target_edge)
            return self.weight * ((1 - torch.pow(cos_sim, 2)).mean())

        elif self.type == 'Cosine_L1':
            # dot_product = (seg_x * target_x) + (seg_y * target_y)
            # cosine = dot_product / ((seg_mag) * (target_mag) + 0.0001)
            # #return self.weight * ((torch.abs(seg_x) + torch.abs(seg_y)) * (1-torch.pow(cosine, 2))).mean()
            # return self.weight * (1-torch.pow(cosine, 2)).mean()
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_sim = cos(seg_edge, target_edge)
            return self.weight * (((torch.abs(seg_x) + torch.abs(seg_y)) * (1 - torch.pow(cos_sim, 2))).mean())

        elif self.type == 'Cosine_Target':
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_sim = cos(seg_edge, target_edge)
            return self.weight * ((torch.abs(seg_x) * torch.exp((torch.abs(target_x) - torch.abs(seg_x))) * torch.exp((1 - torch.pow(cos_sim, 2)))) + (torch.abs(seg_y) * torch.exp((torch.abs(target_y) - torch.abs(seg_y))) * torch.exp((1 - torch.pow(cos_sim, 2))))).mean()

        elif self.type == 'Dot':
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_sim = cos(seg_edge, target_edge)

            return self.weight * ((-1*seg_mag*torch.pow(cos_sim, 2)).mean())


class DiffIOU(torch.nn.Module):

    def __init__(self, weight): #type L2, dot
        super(DiffIOU, self).__init__()
        self.weight = weight
        self.key = 'radiance_normalized'

    def forward(self, pred_dict, label_dict):

        current_seg = pred_dict[self.key].clone()
        if self.key == 'radiance_normalized':
            current_seg = current_seg.squeeze(0)
            current_seg = current_seg.squeeze(1)
        bs = current_seg.shape[0]
        prev_seg = torch.zeros(current_seg.shape).float().cuda()
        next_seg = torch.zeros(current_seg.shape).float().cuda()

        prev_seg[1:bs, : , :] = current_seg[0:bs-1, :, :].clone()
        next_seg[:bs-1, :, :] = current_seg[1:bs, :, :].clone()
        prev_seg[0, :, :] = current_seg[0, :, :]
        next_seg[bs-1, :, :] = current_seg[bs-1, :, :]
        intersection_prev = torch.abs(current_seg*prev_seg)
        intersection_next  = torch.abs(current_seg*next_seg)
        union_prev = torch.abs(current_seg + prev_seg - intersection_prev)
        union_next = torch.abs(current_seg + next_seg - intersection_next)

        iou_prev = (intersection_prev / (union_prev + 0.00001)).mean()
        iou_next = (intersection_next / (union_next + 0.00001)).mean()

        return self.weight * (1 - ((iou_prev + iou_next)/2))


class BBoxSmoothness(torch.nn.Module):
    def __init__(self, weight): #type L2, dot
        super(BBoxSmoothness, self).__init__()
        self.weight = weight
        self.key = 'spatial_transformer'

    def forward(self, pred_dict, label_dict):

        current_box = pred_dict[self.key].clone()
        current_box = current_box.squeeze(0)
        bs = current_box.shape[0]
        prev_box = torch.zeros(current_box.shape).float().cuda()
        next_box = torch.zeros(current_box.shape).float().cuda()

        prev_box[1:bs, : , :] = current_box[0:bs-1, :, :].clone()
        next_box[:bs-1, :, :] = current_box[1:bs, :, :].clone()

        prev_box[0, :, :] = current_box[0, :, :]
        next_box[bs-1, :, :] = current_box[bs-1, :, :]

        scale_prev = (current_box[:,0, 0]*current_box[:, 1, 1]/ prev_box[:,0, 0]*prev_box[:, 1, 1])
        scale_next = (current_box[:,0, 0]*current_box[:, 1, 1]/ next_box[:,0, 0]*next_box[:, 1, 1])

        bbox_x_prev = torch.abs(current_box[:, 0, 2] - prev_box[:, 0, 2])/ current_box[:,0, 0]
        bbox_y_prev = torch.abs(current_box[:, 1, 2] - prev_box[:, 1, 2])/ current_box[:,1, 1]

        bbox_x_next = torch.abs(current_box[:, 0, 2] - next_box[:, 0, 2])/ current_box[:,0, 0]
        bbox_y_next = torch.abs(current_box[:, 1, 2] - next_box[:, 1, 2])/ current_box[:,1, 1]


        return self.weight * ((scale_prev.mean() + scale_next.mean()) / 2 + (bbox_x_next.mean() + bbox_x_prev.mean())/2 + (bbox_y_next.mean()+bbox_y_prev.mean())/2)

class SelectSingleLabel_Sampling(torch.nn.Module):
    """
    Apply a loss 'single' on a single label 'key'    
    """

    def __init__(self, single, key='3D', confidence_key='confidence', proposal_key='importance', train=False):
        super(SelectSingleLabel_Sampling, self).__init__()
        self.single = single
        self.key = key
        self.confidence_key = confidence_key
        self.proposal_key = proposal_key
        self.confidence_key_source = self.confidence_key +   '_source_view'
        self.proposal_key_source = self.proposal_key + '_source_view'
        self.train = train

    def forward(self, pred_dict, label_dict):

        #return ((pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key_source].data / pred_dict[self.proposal_key_source].data) * torch.pow((pred_dict[self.key] - label_dict[self.key]), 2).mean(-1).mean(-1).mean(-1)).mean()
        #return ((pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data) * torch.pow((pred_dict[self.key] - label_dict[self.key]), 2).mean(-1).mean(-1).mean(-1)).mean()

        return ((pred_dict[self.confidence_key].data / pred_dict[self.proposal_key].data)  * torch.pow((pred_dict[self.key] - label_dict[self.key]), 2).mean(-1).mean(-1).mean(-1)).mean()
        # return (  (pred_dict[self.confidence_key]       / pred_dict[self.proposal_key].data)
        #         * (pred_dict[self.confidence_key_source]/ pred_dict[self.proposal_key_source].data)
        #         * torch.pow((pred_dict[self.key] - label_dict[self.key]), 2).mean(-1).mean(-1).mean(-1)).mean()

class BackgroundPixelLoss(torch.nn.Module):
    """
    Apply a loss 'single' on a single label 'key'    
    """
    def __init__(self, single, key='3D', aux_key='bg', confidence_key='confidence', proposal_key='importance', input_crop_key='input_crop_key', crop_key='inpainting_crop', crop_size_key='inpainting_size',  weight=1, train=True):
        super(BackgroundPixelLoss, self).__init__()
        print("ExpBG")

        self.single = single
        self.key = key
        self.aux_key = aux_key
        self.confidence_key = confidence_key
        self.proposal_key = proposal_key
        self.input_crop_key = input_crop_key
        self.crop_key = crop_key
        self.crop_size_key = crop_size_key
        self.weight = weight
        self.train = train
        # TODO : Create a model class and move this function there?

    def forward(self, pred_dict, label_dict):

        return self.weight * torch.exp(-1*torch.pow((pred_dict[self.aux_key]-pred_dict[self.key]), 2)).mean()

class BackgroundPixelLoss_NegSqr(torch.nn.Module):
    """
    Apply a loss 'single' on a single label 'key'    
    """
    def __init__(self, single, key='3D', aux_key='bg', confidence_key='confidence', proposal_key='importance', input_crop_key='input_crop_key', crop_key='inpainting_crop', crop_size_key='inpainting_size',  weight=1, train=True):
        super(BackgroundPixelLoss_NegSqr, self).__init__()
        print("Neg Square Loss BG!!!!!")
        self.single = single
        self.key = key
        self.aux_key = aux_key
        self.confidence_key = confidence_key
        self.proposal_key = proposal_key
        self.input_crop_key = input_crop_key
        self.crop_key = crop_key
        self.crop_size_key = crop_size_key
        self.weight = weight
        self.train = train
        # TODO : Create a model class and move this function there?

    def forward(self, pred_dict, label_dict):

        return self.weight * (-1*torch.pow((pred_dict[self.aux_key]-pred_dict[self.key]), 2)).mean()


class BackgroundPixelLoss_Sampling(torch.nn.Module):
    """
    Apply a loss 'single' on a single label 'key'    
    """
    def __init__(self, single, key='3D', aux_key='bg', confidence_key='confidence', proposal_key='importance', input_crop_key='input_crop_key', crop_key='inpainting_crop', crop_size_key='inpainting_size',  weight=1, train=True):
        super(BackgroundPixelLoss_Sampling, self).__init__()
        print("ExpBG")

        self.single = single
        self.key = key
        self.aux_key = aux_key
        self.confidence_key = confidence_key
        self.proposal_key = proposal_key
        self.confidence_key_source = self.confidence_key + '_source_view'
        self.proposal_key_source = self.proposal_key + '_source_view'
        self.input_crop_key = input_crop_key
        self.crop_key = crop_key
        self.crop_size_key = crop_size_key
        self.weight = weight
        self.train = train
        # TODO : Create a model class and move this function there?

    def forward(self, pred_dict, label_dict):
        #return self.weight * ((pred_dict[self.confidence_key] / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key_source] / pred_dict[self.proposal_key_source].data) * (((-1) * torch.pow((pred_dict[self.input_crop_key].data - pred_dict[self.crop_key].data), 2)).mean(1).sum(1).sum(1)) / pred_dict[self.crop_size_key]).mean()
        #return self.weight * ((pred_dict[self.confidence_key] / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key] / pred_dict[self.proposal_key].data) * (((-1) * torch.pow((pred_dict[self.input_crop_key].data - pred_dict[self.crop_key].data), 2)).mean(1).sum(1).sum(1)) / pred_dict[self.crop_size_key]).mean()

        return self.weight * ((pred_dict[self.confidence_key] / pred_dict[self.proposal_key].data)  * (((-1) * torch.pow((pred_dict[self.input_crop_key].data - pred_dict[self.crop_key].data), 2)).mean(1).sum(1).sum(1)) / pred_dict[self.crop_size_key]).mean()

        #l1 loss
        #return self.weight * (pred_dict[self.confidence_key] * pred_dict[self.proposal_key].data * (((-1) * torch.abs(pred_dict[self.input_crop_key].data - pred_dict[self.crop_key].data)).mean(1).sum(1).sum(1)) / pred_dict[self.crop_size_key]).mean()


class Bg_vs_Fg_PixelLoss_Sampling(torch.nn.Module):
    """
    Apply a loss 'single' on a single label 'key'    
    """
    def __init__(self, single, key='3D', aux_key='bg', confidence_key='confidence', proposal_key='importance', input_crop_key='input_crop_key', crop_key='inpainting_crop', crop_size_key='inpainting_size',  weight=1, train=True):
        super(Bg_vs_Fg_PixelLoss_Sampling, self).__init__()
        print("ExpBG")

        self.single = single
        self.key = key
        self.aux_key = aux_key
        self.confidence_key = confidence_key
        self.proposal_key = proposal_key
        self.confidence_key_source = self.confidence_key + '_source_view'
        self.proposal_key_source = self.proposal_key + '_source_view'
        self.input_crop_key = input_crop_key
        self.crop_key = crop_key
        self.crop_size_key = crop_size_key
        self.weight = weight
        self.train = train
        # TODO : Create a model class and move this function there?

    def forward(self, pred_dict, label_dict):
        # import pickle
        # img_dic = {'gt': label_dict[self.key][0].detach().cpu().numpy().transpose(1,2,0), 'pred': pred_dict[self.aux_key][0].detach().cpu().numpy().transpose(1,2,0)}
        # with open('images_new.pickle', 'wb') as handle:
        #     pickle.dump(img_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # import pdb
        # pdb.set_trace()
        #return self.weight * (pred_dict[self.confidence_key] * pred_dict[self.proposal_key].data * torch.exp(-1*torch.pow((pred_dict[self.key] - pred_dict[self.aux_key]), 2).mean(-1).mean(-1).mean(-1))).mean()
        #return self.weight * (pred_dict[self.confidence_key] * pred_dict[self.proposal_key].data * torch.exp(-1*torch.pow((pred_dict[self.key].data - pred_dict[self.aux_key].data), 2).mean(-1).mean(-1).mean(-1))).mean()
        # if self.train == False:
        #     import pdb
        #     pdb.set_trace()
        #return self.weight * (pred_dict[self.confidence_key] * pred_dict[self.proposal_key].data * torch.exp(-1*torch.pow((label_dict[self.key].data - pred_dict[self.aux_key].data), 2)).mean(-1).mean(-1).mean(-1)).mean()
        #return self.weight * (pred_dict[self.confidence_key] * pred_dict[self.proposal_key].data * torch.exp(-1*torch.pow((label_dict[self.key] - pred_dict[self.aux_key]), 2).mean(-1).mean(-1).mean(-1))).mean()

        #return self.weight * (pred_dict[self.confidence_key] * pred_dict[self.proposal_key].data * (((-1) * torch.pow((label_dict[self.key].data - pred_dict[self.aux_key].data), 2)).mean(-1).mean(-1).mean(-1)) / (pred_dict['scale_x']*pred_dict['scale_y'])).mean()

        #take mean in 3 channels and then take the spatial sum
        return self.weight * ((pred_dict[self.confidence_key] / pred_dict[self.proposal_key].data) * (pred_dict[self.confidence_key_source] / pred_dict[self.proposal_key_source].data) * (((-1) * torch.pow((pred_dict['output_img_crop'].data - pred_dict[self.crop_key].data), 2)).mean(1).sum(1).sum(1)) / pred_dict[self.crop_size_key]).mean()

        #l1 loss
        #return self.weight * (pred_dict[self.confidence_key] * pred_dict[self.proposal_key].data * (((-1) * torch.abs(pred_dict[self.input_crop_key].data - pred_dict[self.crop_key].data)).mean(1).sum(1).sum(1)) / pred_dict[self.crop_size_key]).mean()



class PreSplitPredToSubBatchList(torch.nn.Module):
    """
    Takes a batch of predictions from a network and splits them for multi task learning, e.g. 2D and 3D task or single and multi-view.
    Each sub-batch is passed on to the respective loss functions (e.g. 2D heatmap loss and 3D pose loss).
    """
    # TODO : Understand what this does
    def __init__(self, loss_list, loss_weights):
        super(PreSplitPredToSubBatchList, self).__init__()
        self.loss_list = loss_list
        self.loss_weights = loss_weights

    def forward(self, pred_dict, labels_subbatched):
        assert type(labels_subbatched) == dict
        # split up each of the networm outputs (labels are assumed to be splitted a-priori)
        subBatchOffset = 0
        pred_subbatched = []
        for labels_splitted_i, labels_splitted in labels_subbatched.items(): # first index of labels refers to the different loss types
            keys = list(labels_splitted.keys())
            subBatchSize = len(labels_splitted[keys[0]]) #  self.samplesPerConfig[labels_splitted_i]
            pred_subbatched_singles = {key : pred[subBatchOffset:subBatchOffset + subBatchSize] for key, pred in pred_dict.items()}
            pred_subbatched.append(pred_subbatched_singles)
            subBatchOffset += subBatchSize

        # apply the respective losses
        losses = []
        Label_keys = list(labels_subbatched.keys())
        for li, loss_single in enumerate(self.loss_list): # iterate over the different label types
            loss_ret = loss_single.forward(pred_subbatched[li], labels_subbatched[Label_keys[li]])
            if type(loss_ret) is list:
                loss_list_weighted = [l * self.loss_weights[li] for l in loss_ret]
                losses.extend(loss_list_weighted)
            else:
                losses.append(self.loss_weights[li]*loss_ret)

        return (losses)



class SubjectSelectiveLoss(torch.nn.Module):
    """
    only compute loss for selected subjects
    """
    def __init__(self, subjects, loss):
        super(SubjectSelectiveLoss, self).__init__()
        self.subjects = subjects
        self.loss = loss

    def forward(self, pred_dict, label_dict):
        frame_info = label_dict['frame_info']
        errors = [self.loss()] # TODO...
        return sum(errors) / len(errors)

class IntrinsicLoss(torch.nn.Module):
    """
    Squared error of predicted intrinsics and and GT crop-relative intrinsics
    """
    def __init__(self):
        super(IntrinsicLoss, self).__init__()

    def forward(self, pred_dict, label_dict):
        batch_size = label_dict['bounding_box'].size()[0]
        errors = []
        for bi in range(batch_size):
            bbox = label_dict['bounding_box'][bi].data.cpu().numpy()
            C_img_to_crop_np = transforms_aug.getC_img_to_crop(bbox)
            C_img_to_crop_inv = torch.autograd.Variable(torch.from_numpy(la.inv(C_img_to_crop_np)))
            diff = C_img_to_crop_inv @ pred_dict['intrinsic_crop'][bi].cpu().view(3,3) - label_dict['intrinsic'][bi].cpu().view(3,3) # HACK: noAug assumed
            mask = torch.autograd.Variable(torch.FloatTensor([1,0,1,0,1,1,0,0,0])) # only constrain the important ones, dont learn to predict constants
            diff_sq = mask*torch.mul(diff, diff).view(-1)
            diff_sq_mean = torch.mean(diff_sq)
            errors.append(diff_sq_mean)
        return sum(errors) / len(errors)

class HuberLossPair(torch.nn.Module):
    """
    """
    def __init__(self, delta):
        super(HuberLossPair, self).__init__()
        self.delta = delta
        self.smoothLoss = torch.nn.modules.loss.SmoothL1Loss()

    def forward(self, pred0, pred1):
        return self.smoothLoss(pred0/self.delta,pred1/self.delta)*self.delta
# this implementation caused too large memory consumption
#        diff = pred0 - pred1
#        diff_sq = torch.mul(diff, diff)
#        diff_abs = torch.abs(diff)
        
#        r1 = 0.5*diff_sq
#        r2 = self.delta*(diff_abs-0.5*self.delta)

#        mask_r1 = diff_abs < self.delta

#       errors = mask_r1.float()*r1 + (1-mask_r1).float()*r2
#        return torch.mean(errors)

class MSELossPair(torch.nn.Module):
    """
    Like the conventional MSE loss, but computes gradients with respect to both arguments
    (Conventional losses only with respect to the param, not the label)
    """
    def __init__(self):
        super(MSELossPair, self).__init__()

    def forward(self, pred0, pred1):
        diff = pred0 - pred1
        diff_sq = torch.mul(diff, diff)
        diff_sq_mean = torch.mean(diff_sq)
        return diff_sq_mean

class MSELossDict(torch.nn.Module):
    """
    """
    def __init__(self, key):
        super(MSELossDict, self).__init__()
        self.key = key
        
    def forward(self, pred_dict, label_dict):
        diff = pred_dict[self.key] - label_dict[self.key]
        diff_sq = torch.mul(diff, diff)
        diff_sq_mean = torch.mean(diff_sq)
        return diff_sq_mean

class MAELossPair(torch.nn.Module):
    def __init__(self):
        super(MAELossPair, self).__init__()

    def forward(self, pred0, pred1):
        diff = pred0 - pred1
        diff_abs = torch.abs(diff)
        diff_abs_mean = torch.mean(diff_abs)
        return diff_abs_mean

class MAELossDict(torch.nn.Module):
    def __init__(self, key):
        super(MAELossDict, self).__init__()
        self.key = key
        
    def forward(self, pred_dict, label_dict):
        diff = pred_dict[self.key] - label_dict[self.key]
        diff_abs = torch.abs(diff)
        diff_abs_mean = torch.mean(diff_abs)
        return diff_abs_mean

class MSELoss_3DTreshWeighted(torch.nn.Module):
    """
    Like the conventional MSE loss, but weighted and only computes error for 3D points which are below treshold
    (Conventional losses only with respect to the param, not the label)
    """
    def __init__(self, treshold):
        super(MSELoss_3DTreshWeighted, self).__init__()
        self.treshold_sq = treshold*treshold

    def forward(self, pred0, pred1, weight):
        diff = pred0 - pred1
        diff_sq = torch.mul(diff, diff)
        len_sq = torch.sum(diff_sq, dim=1) # TODO : Check if dim exists (not on the website but might exist on the installed version)
        #valid = torch.clamp(len_sq.data, min=-1, max=self.treshold_sq) # value can't be below 0, ignoring min treshold
        valid = torch.lt(len_sq.data, self.treshold_sq) # value can't be below 0, ignoring min treshold
        diff_sq_mean = torch.mean(diff_sq * torch.autograd.Variable( (valid.float()*weight).expand_as(diff_sq)) )
        return diff_sq_mean


class MAELoss_Weighted(torch.nn.Module):
    """
    Like the conventional MSE loss, but weighted and absolute norm
    """
    def __init__(self):
        super(MAELoss_Weighted, self).__init__()

    def forward(self, pred0, pred1, weight):
        diff = pred0 - pred1
        diff_abs = torch.abs(diff)
        diff_abs_mean = torch.mean(diff_abs * torch.autograd.Variable(weight).expand_as(diff_abs) )
        return diff_abs_mean

class MSELossLabelNormalized(torch.nn.Module):
    """
    """
    def __init__(self):
        super(MSELossLabelNormalized, self).__init__()

    def forward(self, pred, label):
        num_batches = pred.size()[0]
        diff = pred - label/label.view(num_batches, -1).norm(dim=1).expand_as(label)
        #diff = pred/pred.view(num_batches, -1).norm(dim=1).expand_as(pred)*label.view(num_batches, -1).norm(dim=1).expand_as(label) - label
        diff_sq = torch.mul(diff, diff)
        diff_sq_mean = torch.mean(diff_sq)
        return diff_sq_mean

class LossInstanceMeanStdFromLabel(torch.nn.Module):
    """
    Normalize the pose before applying the specified loss (done per batch element)
    """
    def __init__(self, loss_single):
        super(LossInstanceMeanStdFromLabel, self).__init__()
        self.loss_single = loss_single

    def forward(self, preds, labels):
        pred_pose = preds
        label_pose = labels
        batch_size = label_pose.shape[0]
        feature_size = label_pose.shape[1]
        eps = 0.00001 # to prevent division by 0
        # build mean and std across third and fourth dimension and restore afterwards again
        label_mean = torch.mean(label_pose.view([batch_size,feature_size,-1]),dim=2,keepdim=False).view([batch_size,-1,1,1])
        pose_mean  = torch.mean(pred_pose.view( [batch_size,feature_size,-1]),dim=2,keepdim=False).view([batch_size,-1,1,1])
        label_std  = torch.std( label_pose.view([batch_size,feature_size,-1]),dim=2,keepdim=False).view([batch_size,-1,1,1]) + eps
        pose_std   = torch.std( pred_pose.view( [batch_size,feature_size,-1]),dim=2,keepdim=False).view([batch_size,-1,1,1]) + eps

        pred_pose_norm = ((pred_pose - pose_mean)/pose_std)*label_std + label_mean
        if torch.isnan(pred_pose_norm).any():
            print('torch.isnan(pred_pose_norm)')
            IPython.embed()

        return self.loss_single.forward(pred_pose_norm,label_pose)


class LossLabelMeanStdNormalized(torch.nn.Module):
    """
    Normalize the label before applying the specified loss (could be normalized loss..)
    """
    def __init__(self, key, loss_single, subjects=False, weight=1):
        super(LossLabelMeanStdNormalized, self).__init__()
        self.key = key
        self.loss_single = loss_single
        self.subjects = subjects
        self.weight=weight

    def forward(self, preds, labels):
        pred_pose = preds[self.key]
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        label_pose_norm = (label_pose-label_mean)/label_std

        if self.subjects:
            info = labels['frame_info']
            subject = info.data.cpu()[:,3]
            errors = [self.loss_single.forward(pred_pose[i], label_pose_norm[i]) for i,x in enumerate(pred_pose) if subject[i] in self.subjects]
            #print('subject',subject,'errors',errors)
            if len(errors) == 0:
                return torch.autograd.Variable(torch.FloatTensor([0])).cuda()
            return self.weight * sum(errors) / len(errors)

        return self.weight * self.loss_single.forward(pred_pose,label_pose_norm)

class LossLabelMeanStdUnNormalized(torch.nn.Module):
    """
    UnNormalize the prediction before applying the specified loss (could be normalized loss..)
    """
    def __init__(self, key, loss_single, scale_normalized=False, weight=1):
        super(LossLabelMeanStdUnNormalized, self).__init__()
        self.key = key
        self.loss_single = loss_single
        self.scale_normalized = scale_normalized
        #self.subjects = subjects
        self.weight=weight
        assert scale_normalized==False # anything else is deprecated

    def forward(self, preds, labels):
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        pred_pose = preds[self.key]
        
        if self.scale_normalized:
            per_frame_norm_label = label_pose.norm(dim=1).expand_as(label_pose)
            per_frame_norm_pred  = pred_pose.norm(dim=1).expand_as(label_pose)
            pred_pose = pred_pose / per_frame_norm_pred * per_frame_norm_label

        pred_pose_norm = (pred_pose*label_std.view(pred_pose.shape)) + label_mean.view(pred_pose.shape)
        
        return self.weight*self.loss_single.forward(pred_pose_norm, label_pose)

class heightFromPose(torch.nn.Module):
    """
    Compute the subject height from the pose / bone-length
    """
    def __init__(self, loss):
        super(heightFromPose, self).__init__()
        self.loss = loss
        
        joint_indices_h36m=[    0,             1,          2,           3,            4,         5,           6,       7,     8,      9,        10,        11,            12,         13,         14,             15,         16 ]
        joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head','head-top','left-arm','left_forearm','left_hand','right_arm','right_forearm','right_hand']
        self.pose_mean = np.array([
                              [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                              [ -3.11228271e-04,  -7.11847551e-03,  -1.00847712e-03],
                              [ -5.70231131e-03,   3.19657718e-01,   7.19305775e-02],
                              [ -1.02318702e-02,   6.91174560e-01,   1.55397459e-01],
                              [  3.11227401e-04,   7.11841606e-03,   1.00846629e-03],
                              [ -5.04394087e-03,   3.27058249e-01,   7.22720117e-02],
                              [ -9.95563852e-03,   7.08293876e-01,   1.58103980e-01],
                              [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                              [  5.66454234e-03,  -4.35101240e-01,  -9.76913367e-02],
                              [  7.36173335e-03,  -5.83966828e-01,  -1.31184115e-01],
                              [  7.36173335e-03,  -5.83966828e-01,  -1.31184115e-01],
                              [  5.46302480e-03,  -3.83952022e-01,  -8.67773484e-02],
                              [  3.07465356e-03,  -1.87596050e-01,  -4.33811946e-02],
                              [  1.44451846e-03,  -1.20179073e-01,  -2.81799785e-02],
                              [  4.60202399e-03,  -3.83656530e-01,  -8.55249146e-02],
                              [  1.53727470e-03,  -1.97021215e-01,  -4.31663952e-02],
                              [  6.92175456e-04,  -1.68640338e-01,  -3.74556063e-02]])
        self.pose_std = np.array([
                             [ 0.        ,  0.        ,  0.        ],
                             [ 0.11072572,  0.02238527,  0.07245805],
                             [ 0.15855746,  0.18932306,  0.20878552],
                             [ 0.19178746,  0.24317334,  0.24754677],
                             [ 0.11072509,  0.02238515,  0.07245763],
                             [ 0.15879367,  0.1997508 ,  0.21469065],
                             [ 0.18000765,  0.25050014,  0.24851042],
                             [ 0.        ,  0.        ,  0.        ],
                             [ 0.09514143,  0.10131288,  0.12897807],
                             [ 0.1235873 ,  0.13082389,  0.16430713],
                             [ 0.1235873 ,  0.13082389,  0.16430713],
                             [ 0.14603792,  0.097062  ,  0.13950509],
                             [ 0.24351019,  0.12982746,  0.20226888],
                             [ 0.24477822,  0.21502671,  0.23936064],
                             [ 0.13874155,  0.1008755 ,  0.14243477],
                             [ 0.23686488,  0.14490233,  0.20981583],
                             [ 0.24405448,  0.23974177,  0.25520541]])
        self.height_mean = 171.8
        self.height_std = 10.0

        # HACK assuming h36m order
        self.bone_chain = [['neck', 'head-top'],['neck','hip'],['right_up_leg','right_leg'],['right_foot','right_leg']]  
        self.bone_chain = [ (joint_indices_h36m[joint_names_h36m.index(name1)],joint_indices_h36m[joint_names_h36m.index(name2)]) for name1,name2 in self.bone_chain]
        
    def forward(self, preds, labels):
        label_height_raw = labels['height']
        label_height = label_height_raw * self.height_std + self.height_mean
        pred_pose = preds['3D']
        batch_size = pred_pose.size()[0]
        pred_pose_3d = pred_pose.view(batch_size,-1,3)

        pred_pose_norm = (pred_pose_3d*torch.autograd.Variable(torch.from_numpy(self.pose_std).float().unsqueeze(0).expand_as(pred_pose_3d))) + torch.autograd.Variable(torch.from_numpy(self.pose_mean).float().unsqueeze(0).expand_as(pred_pose_3d))

        bone_vectors = [pred_pose_norm[:,b[0],:]-pred_pose_norm[:,b[1],:] for b in self.bone_chain]
        bone_distances = [torch.norm(vec,dim=1) for vec in bone_vectors]
        pred_height_m = sum(bone_distances)
        ankle_bobe_length_cm = 10
        constant_bias_train = 22.67
        constant_bias_test = 22.92
        constant_bias_test_inhous = 21.81
        
        pred_height_cm = pred_height_m * 100 + ankle_bobe_length_cm + constant_bias_train
        return self.loss(pred_height_cm, label_height)


class mAP(torch.nn.Module):
    """
    Mean absolute error
    """
    def __init__(self, key_pred='spatial_transformer', key_gt='bounding_box_yolo', iou_thres=0.5):
        super(mAP, self).__init__()
        self.key_pred = key_pred
        self.key_gt = key_gt
        self.iou_thres = iou_thres

    def forward(self, pred, label):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        import matplotlib.patches as patches
        from scipy import misc
        pred_bboxes = pred[self.key_pred]
        label_bboxes = label[self.key_gt]

        x_scale = pred_bboxes[:, 0, 0]
        y_scale = pred_bboxes[:, 1, 1]
        x_relative = pred_bboxes[:, 0, 2]
        y_relative = pred_bboxes[:, 1, 2]

        detections = label_bboxes.clone()
        detections[:, 0] = (x_relative + 1 - x_scale) / 2
        detections[:, 1] = (y_relative + 1 - y_scale) / 2
        detections[:, 2] = (x_relative + 1 + x_scale) / 2
        detections[:, 3] = (y_relative + 1 + y_scale) / 2

        labels = label_bboxes.clone()
        labels[:, 2] = labels[:, 0] + labels[:, 2]
        labels[:, 3] = labels[:, 1] + labels[:, 3]

        correct = []

        for det_ind in range(detections.shape[0]):

            pred_bbox = detections[det_ind].view(1, -1)
            label_bbox = labels[det_ind].view(1, -1)

            if 0:
                fig, ax = plt.subplots(1)

                import pdb
                pdb.set_trace()
                # cam, frame, trial = label['file_name_info'][det_ind].tolist()
                # base_folder = '/cvlabdata1/cvlab/datasets_ski/Kuetai_2011/'
                # image_template = base_folder + "Videos_Small/trial_{trial}_cam{cam}/frame_{frame:06d}.jpg"
                # file_name = image_template.format(cam=int(cam), frame=int(frame), trial=int(trial))
                # img = misc.imread(file_name)

                # print(cam, frame, trial)
                img = np.zeros((500, 500), dtype=np.uint8)
                ax.imshow(img)

                # rect1 = patches.Rectangle((int(pred_bbox[:,0].detach().cpu().numpy().item()*640), int(pred_bbox[:,1].detach().cpu().numpy().item()*360)),
                #                             int(pred_bbox[:,2].detach().cpu().numpy().item()*640)
                #                           - int(pred_bbox[:,0].detach().cpu().numpy().item()*640),
                #                             int(pred_bbox[:,3].detach().cpu().numpy().item()*360)
                #                           - int(pred_bbox[:,1].detach().cpu().numpy().item()*360),
                #                   linewidth=1, linestyle='dashed', edgecolor='r', facecolor='none')
                #
                # label_bbox = np.clip(label_bbox, 0, 1)
                # rect2 = patches.Rectangle((int(label_bbox[:, 0].detach().cpu().numpy().item() * 640),
                #                           int(label_bbox[:, 1].detach().cpu().numpy().item() * 360)),
                #                          int(label_bbox[:, 2].detach().cpu().numpy().item() * 640) - int(
                #                              label_bbox[:, 0].detach().cpu().numpy().item() * 640),
                #                          int(label_bbox[:, 3].detach().cpu().numpy().item() * 360) - int(
                #                              label_bbox[:, 1].detach().cpu().numpy().item() * 360),
                #                          linewidth=1, linestyle='dashed', edgecolor='g', facecolor='none')

                rect1 = patches.Rectangle((int(pred_bbox[:, 0].detach().cpu().numpy().item() * 500),
                                           int(pred_bbox[:, 1].detach().cpu().numpy().item() * 500)),
                                          int(pred_bbox[:, 2].detach().cpu().numpy().item() * 500)
                                          - int(pred_bbox[:, 0].detach().cpu().numpy().item() * 500),
                                          int(pred_bbox[:, 3].detach().cpu().numpy().item() * 500)
                                          - int(pred_bbox[:, 1].detach().cpu().numpy().item() * 500),
                                          linewidth=1, linestyle='dashed', edgecolor='r', facecolor='none')

                label_bbox = np.clip(label_bbox, 0, 1)
                rect2 = patches.Rectangle((int(label_bbox[:, 0].detach().cpu().numpy().item() * 500),
                                           int(label_bbox[:, 1].detach().cpu().numpy().item() * 500)),
                                          int(label_bbox[:, 2].detach().cpu().numpy().item() * 500) - int(
                                              label_bbox[:, 0].detach().cpu().numpy().item() * 500),
                                          int(label_bbox[:, 3].detach().cpu().numpy().item() * 500) - int(
                                              label_bbox[:, 1].detach().cpu().numpy().item() * 500),
                                          linewidth=1, linestyle='dashed', edgecolor='g', facecolor='none')

                ax.add_patch(rect1)
                ax.add_patch(rect2)
                plt.savefig('bbix_'+ str(det_ind) + '.png')


            # Compute iou with target boxes
            # Compute iou with target boxes
            iou = utils_generic.bbox_iou(pred_bbox, label_bbox)
            # Extract index of largest overlap
            # If overlap exceeds threshold and classification is correct mark as correct
            if iou > self.iou_thres:
                correct.append(1)
            else:
                correct.append(0)


        AP, R, P = utils_generic.ap(correct, detections, labels)

        # Compute mean AP across all classes in this image, and append to image list

        return torch.FloatTensor([AP.mean()])

class mAP_Localization(torch.nn.Module):
    """
    Mean absolute error
    """
    def __init__(self, key_pred='bbox', key_gt='bounding_box_yolo', iou_thres=0.5, img_size=(500,500)):
        super(mAP_Localization, self).__init__()
        self.key_pred = key_pred
        self.key_gt = key_gt
        self.iou_thres = iou_thres
        self.img_size = img_size

    def forward(self, pred_bboxes, label_bboxes):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        import matplotlib.patches as patches
        from scipy import misc

        img_file = pred_bboxes[0]
        bbox_id = pred_bboxes[1]
        pred_bboxes = pred_bboxes[-1]
        detections = pred_bboxes.clone()
        detections[:, 0] = pred_bboxes[:, 0] / self.img_size[0]
        detections[:, 1] = pred_bboxes[:, 1] / self.img_size[1]
        detections[:, 2] = pred_bboxes[:, 2] / self.img_size[0]
        detections[:, 3] = pred_bboxes[:, 3] / self.img_size[1]

        labels = label_bboxes.clone()
        labels[:, 2] = labels[:, 0] + labels[:, 2]
        labels[:, 3] = labels[:, 1] + labels[:, 3]

        correct = []

        for det_ind in range(detections.shape[0]):

            pred_bbox = detections[det_ind].view(1, -1)
            label_bbox = labels[det_ind].view(1, -1)

            # Compute iou with target boxes
            # Compute iou with target boxes
            iou = utils_generic.bbox_iou(pred_bbox, label_bbox)
            # Extract index of largest overlap
            # If overlap exceeds threshold and classification is correct mark as correct

            if iou > self.iou_thres:
                correct.append(1)
            else:
                correct.append(0)

            if 0:
                fig, ax = plt.subplots(1)

                # cam, frame, trial = label['file_name_info'][det_ind].tolist()
                # base_folder = '/cvlabdata1/cvlab/datasets_ski/Kuetai_2011/'
                # image_template = base_folder + "Videos_Small/trial_{trial}_cam{cam}/frame_{frame:06d}.jpg"
                # file_name = image_template.format(cam=int(cam), frame=int(frame), trial=int(trial))
                # img = misc.imread(file_name)

                # print(cam, frame, trial)

                img = scipy.misc.imread(img_file)
                img = scipy.misc.imresize(img, (500,500))
                #img = np.zeros((500, 500), dtype=np.uint8)
                ax.imshow(img)

                # rect1 = patches.Rectangle((int(pred_bbox[:,0].detach().cpu().numpy().item()*640), int(pred_bbox[:,1].detach().cpu().numpy().item()*360)),
                #                             int(pred_bbox[:,2].detach().cpu().numpy().item()*640)
                #                           - int(pred_bbox[:,0].detach().cpu().numpy().item()*640),
                #                             int(pred_bbox[:,3].detach().cpu().numpy().item()*360)
                #                           - int(pred_bbox[:,1].detach().cpu().numpy().item()*360),
                #                   linewidth=1, linestyle='dashed', edgecolor='r', facecolor='none')
                #
                # label_bbox = np.clip(label_bbox, 0, 1)
                # rect2 = patches.Rectangle((int(label_bbox[:, 0].detach().cpu().numpy().item() * 640),
                #                           int(label_bbox[:, 1].detach().cpu().numpy().item() * 360)),
                #                          int(label_bbox[:, 2].detach().cpu().numpy().item() * 640) - int(
                #                              label_bbox[:, 0].detach().cpu().numpy().item() * 640),
                #                          int(label_bbox[:, 3].detach().cpu().numpy().item() * 360) - int(
                #                              label_bbox[:, 1].detach().cpu().numpy().item() * 360),
                #                          linewidth=1, linestyle='dashed', edgecolor='g', facecolor='none')

                rect1 = patches.Rectangle((int(pred_bbox[:, 0].detach().cpu().numpy().item() * 500),
                                           int(pred_bbox[:, 1].detach().cpu().numpy().item() * 500)),
                                          int(pred_bbox[:, 2].detach().cpu().numpy().item() * 500)
                                          - int(pred_bbox[:, 0].detach().cpu().numpy().item() * 500),
                                          int(pred_bbox[:, 3].detach().cpu().numpy().item() * 500)
                                          - int(pred_bbox[:, 1].detach().cpu().numpy().item() * 500),
                                          linewidth=5, linestyle='dashed', edgecolor='r', facecolor='none')

                label_bbox = np.clip(label_bbox, 0, 1)
                rect2 = patches.Rectangle((int(label_bbox[:, 0].detach().cpu().numpy().item() * 500),
                                           int(label_bbox[:, 1].detach().cpu().numpy().item() * 500)),
                                          int(label_bbox[:, 2].detach().cpu().numpy().item() * 500) - int(
                                              label_bbox[:, 0].detach().cpu().numpy().item() * 500),
                                          int(label_bbox[:, 3].detach().cpu().numpy().item() * 500) - int(
                                              label_bbox[:, 1].detach().cpu().numpy().item() * 500),
                                          linewidth=5, linestyle='dashed', edgecolor='g', facecolor='none')

                ax.add_patch(rect1)
                ax.add_patch(rect2)




                plt.title(str(iou))
                plt.savefig('bbix_'+ os.path.basename(img_file).split('.jpg')[0] + '_bbox_' + str(bbox_id) + '.png')


        AP, R, P = utils_generic.ap(correct, detections, labels)

        # Compute mean AP across all classes in this image, and append to image list

        return torch.FloatTensor([AP.mean()]), torch.FloatTensor([iou])


class MAELoss(torch.nn.Module):
    """
    Mean absolute error
    """
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, pred, label):
        diff = pred - label
        diff_sq_mean = torch.mean(torch.abs(diff))
        return diff_sq_mean

class MeanErrorLoss(torch.nn.Module):
    """
    Mean error 
    """
    def __init__(self):
        super(MeanErrorLoss, self).__init__()

    def forward(self, pred, label):
        diff = pred - label
        diff_sq_mean = torch.mean(diff)
        return diff_sq_mean

class MSELossNormalized(torch.nn.Module):
    """
    Like the conventional MSE loss, but computes gradients with respect to both arguments. The inputs are normalized
    w.r.t. the norm.
    (Conventional losses only with respect to the param, not the label)
    """
    def __init__(self):
        super(MSELossNormalized, self).__init__()

    def forward(self, pred, label):
        num_batches = pred.size()[0]
        diff = pred/pred.view(num_batches, -1).norm(dim=1).expand_as(pred) - label/label.view(num_batches, -1).norm(dim=1).expand_as(label)
        diff_sq = torch.mul(diff, diff)
        diff_sq_mean = torch.mean(diff_sq)
        return diff_sq_mean

def testValue_containsLabels(labels):
    label_labels = labels[-1][0]
    if type(label_labels).__module__ != 'numpy' or label_labels.dtype.kind not in {'U', 'S'}:
        import pdb; pdb.set_trace()
        raise ValueError("LossCropRelative, no labels!!!!")

class LossDummyZero(torch.nn.Module):
    """
    Return a constant zero loss
    """
    def __init__(self):
        super(LossDummyZero, self).__init__()

    def forward(self, preds, labels):
        PCK_tensor = torch.Tensor(1,1)
        PCK_tensor[0,0] = 0
        return torch.autograd.Variable(PCK_tensor).cuda()

class LossCropRelative(torch.nn.Module):
    """
    Normalize to image crop size before comparing
    """
    def __init__(self, normalizedLoss, root_center, weak_projection, crop_domain=False, root_index=0):
        super(LossCropRelative, self).__init__()
        self.normalizedLoss = normalizedLoss
        self.root_center = root_center
        self.weak_projection = weak_projection
        self.root_index = root_index
        self.crop_domain = crop_domain

    def forward(self, preds, labels):
        testValue_containsLabels(labels)

        pred = preds[0]
        label_keys = labels[-1][0].tolist()
        #jointPositions_3D_crop, jointPositions_3D_weak = transforms.projective_to_crop_normalized(jointPositions_3D, K_crop[bi])
        num_batch_elements = pred.size()[0]
        jointPositions_3D_crop = pred.view(num_batch_elements,-1, 3)  # batches x Nrjoints x 3

        label_pose = labels[label_keys.index('3D_global')]
        K_crop     = labels[label_keys.index('intrinsic_crop')]
        losses = []
        # indirect loss on reconstructed projective space (projecting prediction forward). Note, was not possible to train
        if not self.crop_domain:
            for bi in range(0, num_batch_elements):
                label_pose_bi = label_pose[bi].view(-1,3)
                if self.weak_projection:
                    jointPositions_reconstructed = transforms_aug.crop_relative_weak_to_projective_tvar(jointPositions_3D_crop[bi], K_crop[bi])
                else:
                    jointPositions_reconstructed = transforms_aug.crop_relative_to_projective_tvar(jointPositions_3D_crop[bi], K_crop[bi])
                if self.root_center:
                    jointPositions_reconstructed = jointPositions_reconstructed - jointPositions_reconstructed[self.root_index,:].expand_as(jointPositions_reconstructed)
                    label_pose_bi                     = label_pose_bi - label_pose_bi[self.root_index,:].expand_as(label_pose_bi)
                losses.append(self.normalizedLoss.forward(jointPositions_reconstructed.view(1,-1), label_pose_bi.view(1,-1)))
        # direct loss in crop space (predicting label backwards)
        else:
            for bi in range(0, num_batch_elements):
                label_pose_bi = label_pose[bi].view(-1,3)
                if self.weak_projection:
                    labels_relative = torch.from_numpy(transforms_aug.projective_to_crop_relative_weak_np(label_pose_bi.data.cpu().numpy(), K_crop[bi].data.cpu().numpy())[0])
                else:
                    labels_relative = torch.from_numpy(transforms_aug.projective_to_crop_relative_np(label_pose_bi.data.cpu().numpy(), K_crop[bi].data.cpu().numpy())[0])
                predictions_relative = jointPositions_3D_crop[bi]
                if self.root_center:
                    predictions_relative = predictions_relative - predictions_relative[self.root_index,:].expand_as(predictions_relative)
                    labels_relative      = labels_relative - labels_relative[self.root_index,:].expand_as(label_pose_bi)
                losses.append(self.normalizedLoss.forward(predictions_relative.view(1,-1), torch.autograd.Variable(labels_relative.cuda()).view(1,-1)))

        return sum(losses) / len(losses)

class LossCropRelative_3DXYfrom2D(torch.nn.Module):
    def __init__(self, poseLoss, root_center, output_index_3d_2d=[0,1], root_index=0, unsupervised=False):
        super(LossCropRelative_3DXYfrom2D, self).__init__()

        self.poseLoss = poseLoss;
        self.root_center = root_center
        self.root_index = root_index
        self.unsupervised = unsupervised
        self.output_index_3d_2d = output_index_3d_2d

    def forward(self, preds, labels):
        label_keys = labels[-1][0].tolist() # transposing...

        pred_3d_raw = preds[self.output_index_3d_2d[0]]
        batch_size = pred_3d_raw.size()[0]
        pred_3d = pred_3d_raw.view(batch_size, -1, 3)

        if self.unsupervised:
            pred_2d_heat = preds[self.output_index_3d_2d[1]] # TODO
            heatmap_width = pred_2d_heat.size()[2]
            jointPositions_2D_crop, confidences = utils_generic.jointPositionsFromHeatmap_cropRelative_batch(pred_2d_heat.data)
            label_2D = torch.autograd.Variable(torch.from_numpy(jointPositions_2D_crop).float()).cuda()
            joint_visible = (torch.from_numpy(confidences) > 0.5).cuda()
        else:
            label_2D = labels[label_keys.index('2D')]
            joint_visible = labels[label_keys.index('joints_visible')].data == 1

        batch_size = joint_visible.size()[0]
        num_joints = joint_visible.size()[1]

        matching_joints_list = [True if j in utils_plt.joint_limbs else False for j in range(0,num_joints)]
        matching_joints_mask = torch.ByteTensor(matching_joints_list).view(1,-1).cuda().expand([*joint_visible.size()]) # expand over whole batch

        mask = (joint_visible*matching_joints_mask).view([*joint_visible.size(),1]).expand([*joint_visible.size(),2]) # duplicate last dimension
        masked_label  = label_2D[mask] # note, yields linearized view, i.e. no bone length possible anymore
        masked_output = pred_3d[:,:,:2][mask]
        # undo centering normalization
        output_crop_masked = masked_output + torch.autograd.Variable(torch.Tensor([0.5]).cuda()).expand_as(masked_output)
        return self.poseLoss(output_crop_masked, masked_label)

class LossBoneLengthSymmetry(torch.nn.Module):
    def __init__(self, bones, symmetric_bones):
        super(LossBoneLengthSymmetry, self).__init__()
        self.bones           = bones;
        self.symmetric_bones = symmetric_bones

    def forward(self, pred_raw, label_UNUSED):
        num_batch_elements = pred_raw.size()[0]
        pred_3d = pred_raw.view([num_batch_elements, -1, 3])
        losses = []
        for symmetry in self.symmetric_bones:
            bone_index_l = self.bones[symmetry[0]]
            bone_index_r = self.bones[symmetry[1]]
            bone_length_l = torch.norm(pred_3d[:,bone_index_l[1],:]-pred_3d[:,bone_index_l[0],:], 2)
            bone_length_r = torch.norm(pred_3d[:,bone_index_r[1],:]-pred_3d[:,bone_index_r[0],:], 2)
            loss = torch.pow(bone_length_l-bone_length_r, 2)
            losses.append(loss)

        return sum(losses) / len(losses)


def merge2Dand3Dprediction_crop(pred_2d, pred_3ds_crop):
    heatmap_width = pred_2d.size()[2]
    num_batch_elements = pred_3ds_crop.size()[0]
    jointPositions_3D_crop = pred_3ds_crop.view(num_batch_elements,-1, 3)  # batches x Nrjoints x 3
    num_joints = jointPositions_3D_crop.size()[1]

    # extract 2D
    jointPositions_2D_crop_batch = []
    for bi in range(0, num_batch_elements):
        jointPositions_2D, confidences, joints_confident = utils_generic.jointPositionsFromHeatmap(pred_2d[bi].data)
        jointPositions_2D_crop = jointPositions_2D / heatmap_width - 0.5 # normalize to 0..1, center around (0.5, 0.5)
        jointPositions_2D_crop_batch.append(jointPositions_2D_crop)
    jointPositions_2D_crop_batch = np.stack(jointPositions_2D_crop_batch)
    jointPositions_2D_crop_batch_var = torch.autograd.Variable(torch.from_numpy(jointPositions_2D_crop_batch)).cuda()

    # copy 2D to 3D (selectively)
    jointPositions_2And3D_crop = torch.autograd.Variable(torch.zeros(num_batch_elements, num_joints, 3)).cuda()
    jointPositions_2And3D_crop[:,:,2] = jointPositions_3D_crop[:,:,2]
    joints_from_2d = utils_plt.joint_limbs
    for bi in range(0, num_batch_elements):
        for j in range(0, num_joints):
            if j in joints_from_2d:
                jointPositions_2And3D_crop[bi,j,:2] = jointPositions_2D_crop_batch_var[bi,j,:]
            else:
                jointPositions_2And3D_crop[bi,j,:2] = jointPositions_3D_crop[bi,j,:2]
    jointPositions_3D_crop = jointPositions_2And3D_crop

    return jointPositions_3D_crop

class LossCropRelative2And3D(torch.nn.Module):
    """
    Use heatmap prediction for x,y part (non-differentiable) and depth regression for z
    """
    def __init__(self, normalizedLoss, root_center, weak_projection, crop_domain=False, root_index=0):
        super(LossCropRelative2And3D, self).__init__()

        self.lossRelative = LossCropRelative(normalizedLoss, root_center, weak_projection, crop_domain, root_index)

    """Assuming 3D pose is passed as a 1D vector at preds[0], and 2D pose as a heatmap at preds[1]"""
    def forward(self, preds, labels):
        # extract 3D
        pred_3d = preds[0]
        num_batch_elements = pred_3d.size()[0]
        #jointPositions_3D_crop = pred_3d.view(num_batch_elements,-1, 3)  # batches x Nrjoints x 3
        #num_joints = jointPositions_3D_crop.size()[1]

        # extract 2D
        pred_2d = preds[1]
        jointPositions_3D_crop = merge2Dand3Dprediction_crop(pred_2d, pred_3d)
#         heatmap_width = pred_2d.size()[2]
#         jointPositions_2D_crop_batch = []
#         for bi in range(0, num_batch_elements):
#             jointPositions_2D, confidences, joints_confident = utils_generic.jointPositionsFromHeatmap(pred_2d[bi].data)
#             jointPositions_2D_crop = jointPositions_2D / heatmap_width - 0.5 # normalize to 0..1, center around (0.5, 0.5)
#             jointPositions_2D_crop_batch.append(jointPositions_2D_crop)
#         jointPositions_2D_crop_batch = np.stack(jointPositions_2D_crop_batch)
#         jointPositions_2D_crop_batch_var = torch.autograd.Variable(torch.from_numpy(jointPositions_2D_crop_batch)).cuda()
#
#         # copy 2D to 3D (selectively)
#         jointPositions_2And3D_crop = torch.autograd.Variable(torch.zeros(num_batch_elements, num_joints, 3)).cuda()
#         jointPositions_2And3D_crop[:,:,2] = jointPositions_3D_crop[:,:,2]
#         joints_from_2d = utils_plt.joint_limbs
#         for bi in range(0, num_batch_elements):
#             for j in range(0, num_joints):
#                 if j in joints_from_2d:
#                     jointPositions_2And3D_crop[bi,j,:2] = jointPositions_2D_crop_batch_var[bi,j,:]
#                 else:
#                     jointPositions_2And3D_crop[bi,j,:2] = jointPositions_3D_crop[bi,j,:2]
#         jointPositions_3D_crop = jointPositions_2And3D_crop
        # jointPositions_3D_crop[:,:,:2] = jointPositions_2D_crop_batch_var does not work, "in-place operations can be only used on variables that don't share storage..."

        return self.lossRelative.forward([jointPositions_3D_crop.view(num_batch_elements,-1)],labels)

#         # crop to global pose
#         label_keys = np.array(labels[-1])[:,0].tolist() # transposing...
#         label_pose = labels[label_keys.index('3D_global')]
#         K_crop     = labels[label_keys.index('intrinsic_crop')]
#         losses = []
#         # indirect loss on reconstructed projective space (projecting prediction forward). Note, was not possible to train
#         if not self.crop_domain:
#             for bi in range(0, num_batch_elements):
#                 if self.weak_projection:
#                     jointPositions_weak_reconstructed = transforms_aug.crop_relative_weak_to_projective_tvar(jointPositions_3D_crop[bi], K_crop[bi])
#                 else:
#                     jointPositions_weak_reconstructed = transforms_aug.crop_relative_to_projective_tvar(jointPositions_3D_crop[bi], K_crop[bi])
#                 label_pose_bi = label_pose[bi].view(-1,3)
#                 if self.root_center:
#                     jointPositions_weak_reconstructed = jointPositions_weak_reconstructed - jointPositions_weak_reconstructed[self.root_index,:].expand_as(label_pose_bi)
#                     label_pose_bi                     = label_pose_bi - label_pose_bi[self.root_index,:].expand_as(label_pose_bi)
#                 losses.append(self.normalizedLoss.forward(jointPositions_weak_reconstructed.view(1,-1), label_pose_bi.view(1,-1)))
#         # direct loss in crop space (predicting label backwards)
#         else:
#             for bi in range(0, num_batch_elements):
#                 label_pose_bi = label_pose[bi].view(-1,3)
#                 if self.weak_projection:
#                     labels_relative = torch.from_numpy(transforms_aug.projective_to_crop_relative_weak_np(label_pose_bi.data.cpu().numpy(), K_crop[bi].data.cpu().numpy())[0])
#                 else:
#                     labels_relative = torch.from_numpy(transforms_aug.projective_to_crop_relative_np(label_pose_bi.data.cpu().numpy(), K_crop[bi].data.cpu().numpy())[0])
#                 predictions_relative = jointPositions_3D_crop[bi]
#                 if self.root_center:
#                     predictions_relative = predictions_relative - predictions_relative[self.root_index,:].expand_as(predictions_relative)
#                     labels_relative      = labels_relative - labels_relative[self.root_index,:].expand_as(label_pose_bi)
#                 losses.append(self.normalizedLoss.forward(predictions_relative.view(1,-1), torch.autograd.Variable(labels_relative.cuda()).view(1,-1)))
#
#         return sum(losses) / len(losses)
class PCKCriterion(torch.nn.Module):
    """
    Percentage of correct key point (wihtin treshold) for 3D pose
    """
    def __init__(self, treshold=0.150, weight=1.0, noSpine=False): # 150mm
        super(PCKCriterion, self).__init__()
        self.treshold = treshold
#        self.noSpine = noSpine
        self.weight = weight

    def forward(self, pred, label):
        size_orig = pred.size()
        batchSize = size_orig[0]
        diff = pred.view([batchSize,-1]) - label.view([batchSize,-1])

        diff_sq = torch.mul(diff,diff)
        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        num_joints = diff_sq.size()[1]
        diff_3d_len_sq = torch.sum(diff_sq, 2)
        #if self.noSpine:
        #    dummy_error = torch.autograd.Variable(diff_3d_len_sq[:,utils_plt.joint_names_h36m.index("neck")].data) /2
        #    diff_3d_len_sq[:,utils_plt.joint_names_h36m.index("spine1")] = dummy_error # spine as if it was the neck

        # treshold per joint distance
        diff_3d_len = torch.sqrt(diff_3d_len_sq)
        correct_points = diff_3d_len < self.treshold

        # build mean in a numerically stable fashion, also for big batches (double and separate mean across dimensions)
        correct_avg_joints = torch.mean(correct_points.double(),dim=1)
#        if self.noSpine:
#            correct_avg_joints = correct_avg_joints*17/16
        correct_avg_poses  = torch.mean(correct_avg_joints,dim=0).squeeze()
        #correct_avg_poses = torch.mean(correct_points.double()).squeeze()
        return -100*correct_avg_poses*self.weight    # mean across batch and joints, in percent (0...100), negative to facilitate minimization

class PCKCriterion_Normalized(torch.nn.Module):
    """
    Normalized mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, treshold=0.150):
        super(PCKCriterion_Normalized, self).__init__()
        self.pck_crit = PCKCriterion(treshold)

    def forward(self, pred, label):
        num_batches = pred.size()[0]
        per_frame_norm_label = label.view(num_batches, -1).norm(dim=1).expand_as(label)
        per_frame_norm_pred = pred.view(num_batches, -1).norm(dim=1).expand_as(pred)
        pred_norm = pred / per_frame_norm_pred * per_frame_norm_label

        return self.pck_crit.forward(pred_norm, label)

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

class Criterion3DPose_ProcrustesCorrected(torch.nn.Module):
    """
    Normalize translaion, scale and rotation in the least squares sense, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_ProcrustesCorrected, self).__init__()
        self.criterion = criterion

    def forward(self, pred_batch, label_batch):
        #Optimal scale transform
        preds_procrustes = []
        batch_size = pred_batch.size()[0]
        for i in range(batch_size):
            num_joints = label_batch[i].view(-1).shape[0]//3
            d, Z, tform = procrustes(label_batch[i].data.cpu().numpy().reshape(num_joints, 3), pred_batch[i].data.cpu().numpy().reshape(num_joints, 3))
            preds_procrustes.append(Z.reshape((num_joints*3)))
        pred_batch_aligned = torch.autograd.Variable(torch.FloatTensor(np.stack(preds_procrustes)))

        return self.criterion.forward(pred_batch_aligned, label_batch)

class Criterion3DPose_leastQuaresScaled(torch.nn.Module):
    """
    Normalize the scale in the least squares sense, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_leastQuaresScaled, self).__init__()
        self.criterion = criterion

    def forward(self, pred, label):
        #Optimal scale transform
        batch_size = pred.size()[0]
        pred_vec = pred.view(batch_size,-1)
        gt_vec = label.view(batch_size,-1)
        dot_pose_pose = torch.sum(torch.mul(pred_vec,pred_vec),1,keepdim=True)
        dot_pose_gt   = torch.sum(torch.mul(pred_vec,gt_vec),1,keepdim=True)

        s_opt = dot_pose_gt / dot_pose_pose

        return self.criterion.forward(s_opt.expand_as(pred)*pred, label)



class Criterion3DPose_normScaled(torch.nn.Module):
    """
    Normalize the scale through the respective norms, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_normScaled, self).__init__()
        self.criterion = criterion

    def forward(self, pred, label):
        num_batches = pred.size()[0]
        per_frame_norm_label = label.view(num_batches, -1).norm(dim=1, keepdim=True) #.expand_as(label)
        per_frame_norm_pred = pred.view(num_batches, -1).norm(dim=1, keepdim=True) #.expand_as(pred)
        print('per_frame_norm_label / per_frame_norm_pred')
        print(torch.mean(per_frame_norm_label/per_frame_norm_pred))
        pred_norm = pred.view(num_batches, -1) / per_frame_norm_pred * per_frame_norm_label

        return self.criterion.forward(pred_norm.view(pred.shape), label)

class Criterion3DPose_LabelBoneLengthScaled(torch.nn.Module):
    """
    Normalize the annotation bonelength scale, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_LabelBoneLengthScaled, self).__init__()
        self.criterion = criterion
        self.bones = utils_plt.bones_h36m # TODO
        
    def forward(self, pred, label):
        bone_length_sums = [sum(skeletons.computeBoneLengths(p,self.bones)) for p in label]
        label_scaled = torch.autograd.Variable(label.data)

        for i,length in enumerate(bone_length_sums):
            label_scaled[i] = label[i] / length.expand_as(label[i])
        
        return self.criterion.forward(pred, label_scaled)


#deprecated("Use Criterion3DPose_normScaled(MPJPECriterion())")
class MPJPECriterionNormalized(torch.nn.Module):
    """
    Normalized mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, weight=1):
        super(MPJPECriterionNormalized, self).__init__()
        self.mpjpe = MPJPECriterion(weight)

    def forward(self, pred, label):
        num_batches = pred.size()[0]
        per_frame_norm_label = label.view(num_batches, -1).norm(dim=1).expand_as(label)
        per_frame_norm_pred = pred.view(num_batches, -1).norm(dim=1).expand_as(pred)
        pred_norm = pred / per_frame_norm_pred * per_frame_norm_label

        return self.mpjpe.forward(pred_norm, label)

     
# class BoneLengthCriterion(torch.nn.Module):
#     """
#     Normalized mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
#     """
#     def __init__(self,  bones):
#         super(BoneLengthCriterion, self).__init__()
#         self.bones = bones
# 
#     def forward(self, p1, p2):
#         batchsize = p1.size()[0]
#         dists = []
#         for bi in range(batchsize):
#             l_pred = getBoneLengths(pose_pred[bi].view(-1,3), self.bones)
#             dists = sum([l_pred[i]-])
#         per_frame_norm_label = label.view(num_batches, -1).norm(dim=1).expand_as(label)
#         per_frame_norm_pred = pred.view(num_batches, -1).norm(dim=1).expand_as(pred)
#         pred_norm = pred / per_frame_norm_pred * per_frame_norm_label
# 
#         return self.mpjpe.forward(pred_norm, label)

# class MPJPECriterion_cropto3D(torch.nn.Module):
#     """
#     Normalizing 3d pose
#     """
#     def __init__(self, label_type_map, weight=1):
#         super(MPJPECriterion_cropto3D, self).__init__()
#         self.mpjpe = MPJPECriterion(weight)
#         self.label_type_map = label_type_map
#
#     def forward(self, preds, labels):
#         label_pose  = labels[self.label_type_map['3D']]
#         crop2global = labels[self.label_type_map['crop2global']]
#         pred = preds[0]
#         num_batches = pred.size()[0]
#         pose_3D = pred.view(num_batches,-1, 3)  # batches x Nrjoints x 3
#         pred_global = torch.bmm(pose_3D, crop2global.transpose(1,2))
#         return self.mpjpe.forward(pred_global.view(num_batches,-1), label_pose)

# class MPJPECriterion_cropto2D_depth1D(torch.nn.Module):
#     """
#     Normalizing 3d pose
#     """
#     def __init__(self, label_type_map):
#         super(MPJPECriterion_cropto3D, self).__init__()
#         self.mpjpe = MPJPECriterion()
#         self.label_type_map = label_type_map
#
#     def forward(self, preds, labels):
#         label_pose  = labels[self.label_type_map['3D']]
#         crop2global = labels[self.label_type_map['crop2global']]
#         pred = preds[0]
#         num_batches = pred.size()[0]
#         pose_3D = pred.view(num_batches,-1, 3)  # batches x Nrjoints x 3
#         pred_global = torch.bmm(pose_3D, crop2global.transpose(1,2))
#         return self.mpjpe.forward(pred_global.view(num_batches,-1), label_pose)

class MSELoss_perFrame(torch.nn.Module):
    def __init__(self):
        super(MSELoss_perFrame, self).__init__()

    def forward(self, pred, label):
        num_batches = pred.size()[0]
        diff = pred.view(num_batches, -1)-label.view(num_batches, -1)
        diff_sq = diff * diff
        mean = diff_sq.mean(dim=1)
        return mean


class MSECriterion_invNormWeighted(torch.nn.Module):
    """
    Weighting such that labels with larger norm get proportionally less weight,
    otherwise poses appearing larger in the image get a stronger weight when using crop centered prediction. Leads to a minor improvement?
    """
    def __init__(self):
        super(MSECriterion_invNormWeighted, self).__init__()
        self.mse_frame = MSELoss_perFrame()

    def forward(self, pred, label):
        num_batches = pred.size()[0]
        per_frame_err = self.mse_frame.forward(pred, label)
        per_frame_norm_label = label.view(num_batches, -1).norm(dim=1).expand_as(per_frame_err)
        per_frame_norm_avg = torch.mean(per_frame_norm_label) # mean across batch
        per_frame_err_normalized = per_frame_err / per_frame_norm_label * per_frame_norm_avg # times average norm to keep overall weight comparable to baseline
        return per_frame_err_normalized.mean()


class MPJPECriterion(torch.nn.Module):
    """
    Mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, weight=1, reduction='elementwise_mean'):
        super(MPJPECriterion, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, label):
        size_orig = pred.size()
        batchSize = size_orig[0]
        diff = pred.view(batchSize,-1) - label.view(batchSize,-1)
        diff_sq = torch.mul(diff,diff)

        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)

        #print('diff_3d_len_sq', diff_3d_len_sq.size())
        if self.reduction == 'sum':
            return self.weight*torch.sum(diff_3d_len);    # mean across batch and joints
        elif self.reduction == 'none':
            return self.weight*diff_3d_len;    # mean across batch and joints
        else: #if self.reduction == 'elementwise_mean':
            return self.weight*torch.mean(diff_3d_len);    # mean across batch and joints


class MPJPECriterionMultiPeople(torch.nn.Module):
    """
    Mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    Assumes that the prediction and lebel vectors are the concatenation of several people outputs or lables.
    """
    def __init__(self, num_coords=51):
        super(MPJPECriterionMultiPeople, self).__init__()
        self.num_coords = num_coords

    def forward(self, pred, label):
        pred1, pred2 = torch.split(pred, self.num_coords, dim=1)
        label1, label2 = torch.split(label, self.num_coords, dim=1)

        size_orig = pred1.size()
        batchSize = size_orig[0]
        diff = pred1 - label1
        diff_sq = torch.mul(diff, diff)

        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)
        mean1 = torch.mean(diff_3d_len)

        size_orig = pred2.size()
        batchSize = size_orig[0]
        diff = pred2 - label2
        diff_sq = torch.mul(diff, diff)

        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)
        mean2 = torch.mean(diff_3d_len)

        z = torch.cat([mean1, mean2])

        return torch.mean(z)


class HeatmapSimilarityCriterion(torch.nn.Module):
    """
    l2 error on heatmap, for those joints that are visible
    """
    def __init__(self):
        super(HeatmapSimilarityCriterion, self).__init__()

    def forward(self, pred, label):
        # pred = pred.data.cpu()  # cpu appears to be faster for the max evaluation
        # label = label.data.cpu()
        # print(label.size(), pred.size())

        numBatchElements = label.size()[0]
        pointDim = label.size()[2]  # 2 for 2D
        numJoints = label.size()[1]

        label_1D = label.view([numBatchElements, numJoints, -1])
        pred_1D = pred.view([numBatchElements, numJoints, -1])

        vis = torch.sum(label_1D, 2)
        diff = pred_1D - label_1D
        diff_sq = torch.mul(diff, diff)

        for bi in range(0, numBatchElements):
            for pi in range(0, numJoints):
                if not vis[bi, pi]:
                    diff_sq[bi, pi, :] = 0
        diff_sq_mean = torch.mean(diff_sq)

        return diff_sq_mean


class PCK2DCriterion(torch.nn.Module):
    """
    2D PCK error, where the threshold is half the head length, assuming predictions in form of heatmaps and labels in form of 2D joint locations, same order of joints
    """
    def __init__(self, headIndex, neckIndex):
        super(PCK2DCriterion, self).__init__()
        self.headIndex = headIndex
        self.neckIndex = neckIndex

    def forward(self, pred_dict, label_dict):
        # print('PCK2DCriterion:pred', pred.size())
        # print('PCK2DCriterion:label', label.size())

        # remove variable wrapper, this loss is not differentiable, only for validation
        pred = pred_dict['2D_heat'].data.cpu() # cpu appears to be faster for the max evaluation
        label = label_dict['2D'].data.cpu()
        numBatchElements = label.size()[0]
        numJoints = label.size()[1]//2
        heatmapWidth = pred.size()[2]

        label = label.view(numBatchElements,numJoints,-1) * heatmapWidth # make 2D array and scale up to heatmap width
        pointDim = label.size()[2] # 2 for 2D

        #print('numBatchElements', numBatchElements, 'pointDim',pointDim, 'numJoints',numJoints)

        loss = 0
        PCKs = []
        jointVisible = np.zeros( (numBatchElements,numJoints) )
        for bi in range(0, numBatchElements):
            correct = 0.
            jointPositions = torch.zeros(numJoints,2)
            headDist = (label[bi,self.headIndex,:]-label[bi,self.neckIndex,:]).norm()

            for j in range(0, numJoints):
                if label[bi,j,0] > 0 and label[bi,j,0]<heatmapWidth and label[bi,j,1]>0 and label[bi,j,1]<heatmapWidth:
                    jointVisible[bi,j] = 1

                mr, mr_ii = pred[bi,j].max(0)

                #print('mr:',mr.size(),'mr_ii',mr_ii.size())
                confidence, mc_i  = mr.max(1)
                mc_i = mc_i[0][0] # from 1x1 variable tensor to scalar, variable is not allowed as index, not differentiable..
                confidence = confidence[0][0]
                #print('mc_i:', mc_i, 'confidence',confidence)
                mr_i = mr_ii[0][mc_i]
                #print('mr_i:', mr_i, 'mc_i',mc_i)
                joint_position = torch.Tensor([[mc_i,mr_i]])
                #print('jp:', joint_position)
                jointPositions[j,:] = joint_position
                #print('l',label[bi,:,j].cpu())
                #print('jp',jointPositions[:,j])
                jointDist = (label[bi,j,:]-jointPositions[j,:]).norm()
                #print('headDist',headDist, 'jointDist',jointDist)
                if jointVisible[bi,j] and jointDist < headDist/2:
                    correct += 1.
            #print('a',label.cpu().numpy()[bi,:,utils_plt.mpii_to_h36m].shape)
            #print('a',label.cpu().numpy()[bi][:,utils_plt.mpii_to_h36m].shape)
            #print('a',label.cpu().numpy()[bi].shape)
            #print('a',label.cpu().numpy().shape)
            numVisibleJoints = sum(jointVisible[bi,:])
            if numVisibleJoints == 0:
                PCK = 1
            else:
                PCK = correct / float(numVisibleJoints)
            #print('PCK', PCK)
            PCKs.append(PCK)

            # img = misc.imread(imgName)
            #plt.imshow(img)

            if 0:
                plt.figure(1)
                ax = plt.subplot(111)
                jointVisible_h36m = jointVisible[bi, utils_plt.mpii_to_h36m]
                bones_visible = utils_plt.filterBones(utils_plt.bones_h36m, jointVisible_h36m)
                #bones_visible = utils_plt.bones_h36m
                utils_plt.plot_2Dpose(ax, jointPositions.cpu().numpy()[utils_plt.mpii_to_h36m,:].T, colormap='hsv',   bones=bones_visible, limits=[0,heatmapWidth,heatmapWidth,0])
                utils_plt.plot_2Dpose(ax,      label.cpu().numpy()[bi][utils_plt.mpii_to_h36m,:].T, colormap='winter', bones=bones_visible, limits=[0,heatmapWidth,heatmapWidth,0])
                utils_plt.plot_2Dpose(ax,      label.cpu().numpy()[bi][:,:].T, colormap='summer', bones=[[8,9]], limits=[0,0,heatmapWidth,heatmapWidth])
                plt.show()


        PCKmean = np.mean(PCKs)
        #print('batch mean PCK', PCKmean)
        # loss/nn.Module needs to return a Variable of Tensor
        PCK_tensor = torch.Tensor(1,1)
        PCK_tensor[0,0] = PCKmean
        PCK_tensor = torch.autograd.Variable(PCK_tensor).cuda()

        return -PCK_tensor; # Note, minus, to make it a loss (smaller value = better)



class Heatmap2D_DepthRegression_Criterion(torch.nn.Module):
    """
    2D PCK error, where the treshold is half the head length, assuming predictions in form of heatmaps and labels in form of 2D joint locations, same order of joints
    """
    def __init__(self,label_type_map,weight):
        super(Heatmap2D_DepthRegression_Criterion, self).__init__()
        self.label_type_map = label_type_map
        self.mpjpe_norm = MPJPECriterionNormalized(weight)

    def forward(self, preds, labels):
#        print('PCK2DCriterion:pred', pred.size())
#        print('PCK2DCriterion:label', label.size())

        crop2global = labels[self.label_type_map['crop2global'] ]

        label_3D  = labels[self.label_type_map['3D'] ].cpu()
        numBatchElements = label_3D.size()[0]
        label_3D = label_3D.view(numBatchElements,-1,3)

        pred_heat  = preds[1].data.cpu() # not differentiable with respect to heatmap
        pred_3d    = preds[0].cpu().view(numBatchElements,-1,3)
        pred_depth = pred_3d[:,:,2]

        numJoints_mpi  = label_3D.size()[1]
        numJoints_h36m = len(utils_plt.joint_names_h36m)
        heatmapWidth = pred_heat.size()[2]


 #       print('numBatchElements', numBatchElements, 'pointDim',pointDim, 'numJoints',numJoints)

        headIndex_mpi = 9 # assuming mpii annotation
        neckIndex_mpi = 8  # assuming mpii annotation
        headIndex_h36m = 9 # assuming mpii annotation
        neckIndex_h36m = 8  # assuming mpii annotation
        rootIndex_h36m = 0
        rootIndex_mpi = 6
        PCKs = []
        MPJPEs = []
        for bi in range(0, numBatchElements):
            correct = 0.
            headDist_2D = (label_3D[bi,headIndex_h36m,:2]-label_3D[bi,neckIndex_h36m,:2]).norm()
            jointPositions_2D = torch.zeros(numJoints_mpi,2)

            for j in range(0, numJoints_mpi):
                mr, mr_ii = pred_heat[bi,j].max(0)

                #print('mr:',mr.size(),'mr_ii',mr_ii.size())
                confidence, mc_i  = mr.max(1)
                mc_i = mc_i[0][0] # from 1x1 variable tensor to scalar, variable is not allowed as index, not differentiable..
                confidence = confidence[0][0]
                #print('mc_i:', mc_i, 'confidence',confidence)
                mr_i = mr_ii[0][mc_i]
                #print('mr_i:', mr_i, 'mc_i',mc_i)
                joint_position = torch.Tensor([[mc_i,mr_i]])
                #print('jp:', joint_position)
                jointPositions_2D[j,:] = joint_position



                #print('l',label[bi,:,j].cpu())
                #print('jp',jointPositions[:,j])

            # different joint order
            jointPositions_2D[:,:] = torch.from_numpy(jointPositions_2D.numpy()[utils_plt.mpii_to_h36m,:])
            root_position_2D = jointPositions_2D[rootIndex_h36m,:].clone()

            jointPositions_3D = Variable(torch.zeros(numJoints_h36m,3))

            for j in range(0, numJoints_h36m):
                # root center
                jointPositions_2D[j,:] = jointPositions_2D[j,:]-root_position_2D
                # normalize by map width
                jointPositions_2D[j,:] = jointPositions_2D[j,:]/heatmapWidth*2 # and magic factor 2
                jointDist = (label_3D[bi,j,:2].data.cpu()-jointPositions_2D[j,:]).norm()
                #print('headDist',headDist, 'jointDist',jointDist)
                if jointDist < headDist_2D/2:
                    correct += 1.

                # save 3D
                jointPositions_3D[j,0:2] = jointPositions_2D[j,:]
                jointPositions_3D[j,2:3] = pred_depth[bi,j]
                # crop to global frame
                jointPositions_3D[j,:] = torch.mm(jointPositions_3D[j:j+1,:], crop2global[bi].transpose(0,1).cpu())
                #pred_global = torch.bmm(pose_3D, crop2global.transpose(1,2))

            #print('a',label.cpu().numpy()[bi,:,utils_plt.mpii_to_h36m].shape)
            #print('a',label.cpu().numpy()[bi][:,utils_plt.mpii_to_h36m].shape)
            #print('a',label.cpu().numpy()[bi].shape)
            #print('a',label.cpu().numpy().shape)
            PCK = correct / float(numJoints_mpi)
            #print('PCK', PCK)
            PCKs.append(PCK)

            MPJPE_3D = self.mpjpe_norm.forward(jointPositions_3D, label_3D[bi])
            MPJPEs.append(MPJPE_3D)

            # img = misc.imread(imgName)
            #plt.imshow(img)

            if 1:
                #plt.switch_backend('Qt5Agg')
                plt.figure(1)
                ax = plt.subplot(111)
                #jointVisible_h36m = jointVisible[bi, utils_plt.mpii_to_h36m]
                #bones_visible = utils_plt.filterBones(utils_plt.bones_h36m, jointVisible_h36m)
                #bones_visible = utils_plt.bones_h36m
                utils_plt.plot_2Dpose(ax, jointPositions_3D.cpu().data.numpy()[:,:2].T,  colormap='hsv',    bones=utils_plt.bones_h36m, limits=[-1,1,1,-1])
                utils_plt.plot_2Dpose(ax, label_3D.cpu().data.numpy()[bi][:,:2].T, colormap='winter', bones=utils_plt.bones_h36m, limits=[-1,1,1,-1])
                utils_plt.plot_2Dpose(ax, label_3D.cpu().data.numpy()[bi][:,:2].T, colormap='summer', bones=[[headIndex_mpi, neckIndex_mpi]], limits=[-1,-1,1,1], linewidth=5)
                plt.show()

        PCKmean = np.mean(PCKs)
        #print('batch mean PCK', PCKmean)
        # loss/nn.Module needs to return a Variable of Tensor
        PCK_tensor = torch.Tensor(1,1)
        PCK_tensor[0,0] = PCKmean
        PCK_tensor = torch.autograd.Variable(PCK_tensor)

        #return -PCK_tensor; # Note, minus, to make it a loss (smaller value = better)
        return np.mean(MPJPEs); # Note, minus, to make it a loss (smaller value = better)

class MSELoss_TrainableFusion(torch.nn.Module):
    def __init__(self, reg_factor):
        super(MSELoss_TrainableFusion, self).__init__()
        self.reg_factor = reg_factor

    def forward(self, pred_list, label):
        loss_list = []
        pred = pred_list[0]
        alpha = pred_list[1]
        label1 = label[1][0]
        label2 = label[1][1]
        label = torch.cat((label1, label2), 1)
        size_orig = pred.size()
        batchSize = size_orig[0]

        diff = pred - label
        diff_sq = torch.mul(diff,diff)
        diff_sq_mean = torch.mean(diff_sq)

        reg = self.reg_factor/(alpha*alpha)

        loss_list.append(diff_sq_mean)
        loss_list.append(reg)
        return loss_list
        #return diff_sq_mean + reg;


class MSELoss_TrainableFusion_Normalized(torch.nn.Module):
    def __init__(self, reg_factor):
        super(MSELoss_TrainableFusion_Normalized, self).__init__()
        self.reg_factor = reg_factor

    def forward(self, pred_list, label):
        loss_list = []
        pred = pred_list[0]
        alpha = pred_list[1]
        label1 = label[1][0]
        label2 = label[1][1]
        #label = torch.cat((label1, label2), 1)
        size_orig = pred.size()
        batchSize = size_orig[0]

        pred1 = pred[:, 0:size_orig[1]//2]
        pred2 = pred[:, size_orig[1]//2:size_orig[1]]
        pred_norm1 = pred1/pred1.contiguous().view(batchSize, -1).norm(dim=1).expand_as(pred1)
        pred_norm2 = pred2/pred2.contiguous().view(batchSize, -1).norm(dim=1).expand_as(pred2)

        label_norm1 = label1/label1.view(batchSize, -1).norm(dim=1).expand_as(label1)
        label_norm2 = label2/label2.view(batchSize, -1).norm(dim=1).expand_as(label2)

        pred_norm = torch.cat((pred_norm1, pred_norm2), 1)
        label_norm = torch.cat((label_norm1, label_norm2), 1)
        #diff = pred - label
        diff = pred_norm - label_norm
        diff_sq = torch.mul(diff,diff)
        diff_sq_mean = torch.mean(diff_sq)

        reg = self.reg_factor * 1.0/(alpha*alpha)

        loss_list.append(diff_sq_mean)
        loss_list.append(reg)
        return loss_list
        #return diff_sq_mean + reg;


class MSELoss_TrainableFusion_OneStream(torch.nn.Module):
    def __init__(self):
        super(MSELoss_TrainableFusion_OneStream, self).__init__()

    def forward(self, pred, label):
        label1 = label[0]
        label2 = label[1]
        label = torch.cat((label1, label2), 1)
        size_orig = pred.size()
        batchSize = size_orig[0]
        diff = pred - label
        diff_sq = torch.mul(diff,diff)
        diff_sq_mean = torch.mean(diff_sq)

        return diff_sq_mean

class MPJPECriterion_TrainableFusion(torch.nn.Module):
    """
    Mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, weight=1):
        super(MPJPECriterion_TrainableFusion, self).__init__()
        self.weight = weight

    def forward(self, pred_list, label):
        pred = pred_list[0]
        alpha = pred_list[1]
        label1 = label[0]
        label2 = label[1]
        label = torch.cat((label1, label2), 0)
        size_orig = pred.size()
        batchSize = size_orig[0]
        pred1 = pred[:, 0:size_orig[1]//2]
        pred2 = pred[:, size_orig[1]//2: size_orig[1]]
        pred = torch.cat((pred1, pred2), 0)
        diff = pred - label
        diff_sq = torch.mul(diff,diff)

        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)

        #print('diff_3d_len_sq', diff_3d_len_sq.size())

        return self.weight*torch.mean(diff_3d_len);    # mean across batch and joints

class MPJPECriterion_TrainableFusion_Normalized(torch.nn.Module):
    """
    Mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, weight=1):
        super(MPJPECriterion_TrainableFusion_Normalized, self).__init__()
        self.weight = weight

    def forward(self, pred_list, label):
        pred = pred_list[0]
        alpha = pred_list[1]
        label1 = label[0]
        label2 = label[1]
        label = torch.cat((label1, label2), 0)
        size_orig = pred.size()
        batchSize = size_orig[0]
        pred1 = pred[:, 0:size_orig[1]//2]
        pred2 = pred[:, size_orig[1]//2: size_orig[1]]
        pred = torch.cat((pred1, pred2), 0)
        per_frame_norm_pred  = pred.view(batchSize*2, -1).norm(dim=1).expand_as(pred)
        per_frame_norm_label = label.view(batchSize*2, -1).norm(dim=1).expand_as(label)
        pred_norm = pred / per_frame_norm_pred * per_frame_norm_label
        diff = pred_norm - label
        diff_sq = torch.mul(diff,diff)

        diff_sq = diff_sq.view((batchSize*2, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)

        #print('diff_3d_len_sq', diff_3d_len_sq.size())

        return self.weight*torch.mean(diff_3d_len);    # mean across batch and joints


class MPJPECriterion_TrainableFusion_OneStream(torch.nn.Module):
    """
    Mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, weight=1):
        super(MPJPECriterion_TrainableFusion_OneStream, self).__init__()
        self.weight = weight

    def forward(self, pred, label):
        label1 = label[0]
        label2 = label[1]
        label = torch.cat((label1, label2), 0)
        size_orig = pred.size()
        batchSize = size_orig[0]
        pred1 = pred[:, 0:size_orig[1]//2]
        pred2 = pred[:, size_orig[1]//2: size_orig[1]]
        pred = torch.cat((pred1, pred2), 0)
        diff = pred - label
        diff_sq = torch.mul(diff,diff)

        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)

        #print('diff_3d_len_sq', diff_3d_len_sq.size())

        return self.weight*torch.mean(diff_3d_len);    # mean across batch and joints


class MSELoss_Iterative(torch.nn.Module):
    def __init__(self, weight=1):
        super(MSELoss_Iterative, self).__init__()
        self.weight = weight

    def forward(self, pred, label):

        loss_list = []
        pose_loss = 0
        mask_loss = 0
        pred_poses = pred[1]
        pred_masks = pred[2]
        label_poses = label[0]

        for i in range(len(pred_poses)):

            diff = pred_poses[i] - label_poses
            diff_sq = torch.mul(diff, diff)
            diff_sq_mean = torch.mean(diff_sq)
            pose_loss += diff_sq_mean

            diff2 = pred_masks[i+1] - pred_masks[i]
            diff_sq2 = torch.mul(diff2, diff2)
            diff_sq_mean2 = torch.mean(diff_sq2)
            mask_loss += diff_sq_mean2
            loss_list.append(diff_sq_mean2)

        loss_list.append(pose_loss)
        loss_list.append(mask_loss)

        return loss_list


class MPJPECriterion_Iterative(torch.nn.Module):
    """
    Mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, weight=1):
        super(MPJPECriterion_Iterative, self).__init__()
        self.weight = weight

    def forward(self, pred, label):
        pred_poses = pred[0]
        label_poses = label[0]

        size_orig = pred_poses.size()
        batchSize = size_orig[0]
        diff = pred_poses - label_poses
        diff_sq = torch.mul(diff,diff)

        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)

        return self.weight*torch.mean(diff_3d_len);    # mean across batch and joints


class MSELoss_TrainableFusion_Iterative(torch.nn.Module):
    def __init__(self, reg_factor):
        super(MSELoss_TrainableFusion_Iterative, self).__init__()
        self.reg_factor = reg_factor

    def forward(self, pred_list, label):
        pose_loss = 0
        pose_reg = 0
        mask_loss = 0
        mask_reg = 0
        loss_list = []
        pred = pred_list[1]
        pred_masks = pred_list[5]
        alpha = pred_list[2]
        alpha_mask = pred_list[6]
        label1 = label[1][0]
        label2 = label[1][1]
        label = torch.cat((label1, label2), 1)

        for i in range(len(pred)):

            diff = pred[i] - label
            diff_sq = torch.mul(diff, diff)
            diff_sq_mean = torch.mean(diff_sq)

            reg = self.reg_factor / (alpha[i] * alpha[i])

            pose_loss += diff_sq_mean
            pose_reg += reg

            diff2 = pred_masks[i + 1] - pred_masks[i]
            diff_sq2 = torch.mul(diff2, diff2)
            diff_sq_mean2 = torch.mean(diff_sq2)
            reg_mask = self.reg_factor / (alpha_mask[i] * alpha_mask[i])

            mask_loss += diff_sq_mean2
            mask_reg += reg_mask

        loss_list.append(pose_loss)
        loss_list.append(mask_loss)
        loss_list.append(pose_reg)
        loss_list.append(mask_reg)
        return loss_list
        # return diff_sq_mean + reg;

class MPJPECriterion_TrainableFusion_Iterative(torch.nn.Module):
    """
    Mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, weight=1):
        super(MPJPECriterion_TrainableFusion_Iterative, self).__init__()
        self.weight = weight

    def forward(self, pred_list, label):
        pred = pred_list[0]
        alpha = pred_list[1]
        label1 = label[0]
        label2 = label[1]
        label = torch.cat((label1, label2), 0)
        size_orig = pred.size()
        batchSize = size_orig[0]
        pred1 = pred[:, 0:size_orig[1]//2]
        pred2 = pred[:, size_orig[1]//2: size_orig[1]]
        pred = torch.cat((pred1, pred2), 0)
        diff = pred - label
        diff_sq = torch.mul(diff,diff)

        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)

        #print('diff_3d_len_sq', diff_3d_len_sq.size())

        return self.weight*torch.mean(diff_3d_len);    # mean across batch and joints

class MinPairwiseLineDistance(torch.nn.Module):
    """
    """
    def __init__(self, key='pairwise_dist', weight=0.1):
        super(MinPairwiseLineDistance, self).__init__()
        self.key = key
        self.weight = weight

    def forward(self, input_dict, label_dict_unused):
        pairwise_dist = input_dict[self.key]
        reference_prob = 0.0
        diffs = torch.mean(torch.pow((pairwise_dist - reference_prob), 2))

        return self.weight*diffs


class ConfidencePrior(torch.nn.Module):
    """
    """
    def __init__(self, key='confidence', weight=0.1):
        super(ConfidencePrior, self).__init__()
        self.key = key
        self.weight = weight

    def forward(self, input_dict, label_dict_unused):
        confidence = input_dict[self.key]
        reference_prob = 0.0
        diffs = torch.mean(torch.pow((confidence- reference_prob), 2))

        return self.weight*diffs

class ConfidenceAllPrior(torch.nn.Module):
    """
    """
    def __init__(self, key='confidence_all', weight=0.1):
        super(ConfidenceAllPrior, self).__init__()
        self.key = key
        self.weight = weight

    def forward(self, input_dict, label_dict_unused):
        confidence = input_dict[self.key]
        reference_prob = 0.0
        diffs = torch.mean(torch.pow((confidence- reference_prob), 2))

        return self.weight*diffs

class ConfidenceBeforeSoftmaxPrior(torch.nn.Module):
    """
    """
    def __init__(self, key='confidence_before_softmax', weight=0.1):
        super(ConfidenceBeforeSoftmaxPrior, self).__init__()
        self.key = key
        self.weight = weight

    def forward(self, input_dict, label_dict_unused):
        confidence = input_dict[self.key]
        reference_prob = 0.0
        diffs = torch.mean(torch.pow((confidence- reference_prob), 2))

        return self.weight*diffs

class AffineCropPositionPrior(torch.nn.Module):
    """
    """
    def __init__(self, fullFrameResolution, weight=0.1):
        super(AffineCropPositionPrior, self).__init__()
        self.key = 'spatial_transformer'
        self.weight = weight
        # without this aspect ratio the side that is longer in the image will also be longer in the crop
        # (becasue the x and y coordinates are normalized 0..1 irrespective of their true pixel length.
        # But we desire an equal aspect ratio:
        self.scale_aspectRatio = torch.FloatTensor(np.array(fullFrameResolution)/min(fullFrameResolution)).cuda()
        self.scale_aspectRatio[0] *= 1.5 # makes x dimension smaller (1.5 as wide as y), to prefere tall crops for upright poses

    def forward(self, input_dict, label_dict_unused):
        affine_params = input_dict[self.key]
        scale_mean = 0.4
        trans_mean = 0
        diffs = 0

        translations = affine_params[:, :, :, 2]
        scales = torch.stack([affine_params[:, :, 0, 0],affine_params[:, :, 1,1]], dim=-1)

        # average position across batch (which is a sample of the whole dataset) should be the image center
        # take mean across batch (dim=1), number of transformers per image (dim=0) will be averaged after taking the difference.
        # Otherwise it is too easy to fulfill the prior with opposing positions and scales
        diffs += torch.mean((torch.mean(translations,dim=1)*self.scale_aspectRatio.unsqueeze(0).unsqueeze(0) - trans_mean)**2)
        # put a slight penality on scale, towards small scales
        diffs += torch.mean((torch.mean(scales,dim=1)*self.scale_aspectRatio.unsqueeze(0).unsqueeze(0) - scale_mean)**2)
        #diffs += torch.mean((torch.mean(scales[:, :, 1],dim=1)*self.scale_aspectRatio[1] - scale_mean)**2)
        return self.weight*diffs

class MaskPrior(torch.nn.Module):
    """
    """
    def __init__(self, weight=0.1):
        super(MaskPrior, self).__init__()
        self.key = 'mask_prior_loss'
        self.weight = weight


    def forward(self, input_dict, label_dict_unused):
        mask_prior = input_dict[self.key]
        return self.weight*mask_prior

class FGPrior(torch.nn.Module):
    """
    """
    def __init__(self, key, weight=0.1):
        super(FGPrior, self).__init__()
        self.key = key
        self.weight = weight


    def forward(self, input_dict, label_dict_unused):

        output_mask = input_dict[self.key]
        #output_mask = input_dict['blend_mask_crop']
        return self.weight * torch.mean(torch.abs(output_mask))
        #return self.weight*torch.mean(torch.pow(output_mask, 2))

class RadianceNormalizedPrior(torch.nn.Module):
    """
    """
    def __init__(self, key, weight=0.1):
        super(RadianceNormalizedPrior, self).__init__()
        self.key = key
        self.weight = weight


    def forward(self, input_dict, label_dict_unused):

        output_mask = input_dict[self.key]
        #output_mask = input_dict['blend_mask_crop']
        return self.weight * torch.mean(torch.abs(output_mask))
        #return self.weight*torch.mean(torch.pow(output_mask, 2))

class RadianceNormalizedPriorBinary(torch.nn.Module):
    """
    """
    def __init__(self, key, weight=0.1):
        super(RadianceNormalizedPriorBinary, self).__init__()
        self.key = key
        self.weight = weight

    def forward(self, input_dict, label_dict_unused):
        output_mask = input_dict[self.key]
        return self.weight * torch.mean((torch.mul(output_mask , (1 - output_mask)))) #p(1-p)
        #return self.weight * torch.mean((0.2 * torch.exp(- torch.pow((output_mask - 0.5), 2) / 0.05))) # 0.2*exp(-(p-0.5)^2/0.05)

class AffineCropLocalPrior(torch.nn.Module):
    """
    """
    def __init__(self, fullFrameResolution, weight=0.1):
        super(AffineCropLocalPrior, self).__init__()
        self.key = 'spatial_transformer'
        self.weight = weight
        # without this aspect ratio the side that is longer in the image will also be longer in the crop
        # (becasue the x and y coordinates are normalized 0..1 irrespective of their true pixel length.
        # But we desire an equal aspect ratio:
        self.scale_aspectRatio = torch.FloatTensor(np.array(fullFrameResolution)/min(fullFrameResolution)).cuda()
        self.scale_aspectRatio[0] *= 1.5 # makes x dimension smaller (1.5 as wide as y), to prefere tall crops for upright poses

    def forward(self, input_dict, label_dict_unused):
        affine_params = input_dict[self.key]
        scale_mean = 0.4
        trans_mean = 0
        diffs = 0

        assert len(input_dict['scale_x'].shape) ==1
        scales = torch.stack([input_dict['scale_x'], input_dict['scale_y']], dim=-1).unsqueeze(0)
        translations = torch.stack([input_dict['offset_x'], input_dict['offset_y']], dim=-1).unsqueeze(0)

        # average position across batch (which is a sample of the whole dataset) should be the image center
        # take mean across batch (dim=1), number of transformers per image (dim=0) will be averaged after taking the difference.
        # Otherwise it is too easy to fulfill the prior with opposing positions and scales
        diffs += torch.mean((torch.mean(translations,dim=1)*self.scale_aspectRatio.unsqueeze(0).unsqueeze(0) - trans_mean)**2)
        # put a slight penality on scale, towards small scales
        #diffs += torch.mean((torch.mean(scales,dim=1)*self.scale_aspectRatio.unsqueeze(0).unsqueeze(0) - scale_mean)**2)
        #diffs += torch.mean((torch.mean(scales[:, :, 1],dim=1)*self.scale_aspectRatio[1] - scale_mean)**2)
        return self.weight*diffs


class AffineCropRelativeSize(torch.nn.Module):
    """
    """
    def __init__(self, fullFrameResolution, weight=0.1):
        super(AffineCropRelativeSize, self).__init__()
        self.key = 'spatial_transformer'
        self.weight = weight
        # without this aspect ratio the side that is longer in the image will also be longer in the crop. But we desire an equal aspect ratio
        self.scale_aspectRatio = torch.FloatTensor(np.array(fullFrameResolution)/min(fullFrameResolution)).cuda()

    def forward(self, input_dict, label_dict_unused):
        affine_params = input_dict[self.key]
        diffs = 0

        # relative size between crops. Using the mean as reference works also for more/less than two crops.
        scales = torch.stack([affine_params[:, :, 0, 0],affine_params[:, :, 1,1]], dim=-1)
        mean_size_across_crops = torch.mean(scales,dim=0)
        sq_dist_to_mean = (scales - mean_size_across_crops.unsqueeze(0).expand_as(scales))**2
        diffs += torch.mean(sq_dist_to_mean)

        return self.weight*diffs

class EntropyPrior(torch.nn.Module):
    """
    make foregrounds of both spatial transformers as different as possible
    """
    def __init__(self, weight=0.1):
        super(EntropyPrior, self).__init__()
        self.key = 'latent_fg'
        self.weight = weight
        # without this aspect ratio the side that is longer in the image will also be longer in the crop. But we desire an equal aspect ratio

    def forward(self, input_dict, label_dict_unused):
        latent_fg = input_dict[self.key]

        eps = 0.0001
        dot_product = torch.sum(latent_fg[0]*latent_fg[1],dim=-1)
        cos_angle = dot_product / (eps + torch.norm(latent_fg[0],dim=-1) * torch.norm(latent_fg[1],dim=-1) )

        return self.weight*cos_angle


class RunningMomentModule(torch.nn.Module):
    """
    Base class for computing the running mean and std
    """
    def __init__(self):
        super(RunningMomentModule, self).__init__()
        self.count = 0
        self.mean = 0
        self.M2 = 0
        self.eps = 0.001

    # From wikipedia, https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # Original from B. P. Welford (1962)."Note on a method for calculating corrected sums of squares and products"
    # for a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def updateRunningMoments(self, newValue):
        self.count += 1
        delta = newValue - self.mean
        self.mean += delta / self.count
        delta2 = newValue - self.mean
        self.M2 += delta * delta2

    # to be able to propagate through
    def runningMomentsWithoutUpdate(self, newValue):
        count_ = self.count + 1
        delta = newValue - self.mean
        mean_ = self.mean + delta / count_
        delta2 = newValue - mean_
        M2_ = self.M2 + delta * delta2
        return mean_, self.eps+torch.sqrt(M2_) / count_

    # retrieve the mean, variance and sample variance from an aggregate
    def getMeanStdRegularized(self):
        variance = self.M2 / self.count
        # +eps to prevent division by zero
        return self.mean, self.eps+torch.sqrt(variance)

class StaticNormalizedLoss():
    """
    Normalize by std and mean before loss computation
    """
    def __init__(self, key, loss_single):
        super(StaticNormalizedLoss, self).__init__()
        self.key = key
        self.loss_single = loss_single

    def forward(self, preds, labels):
        pred_pose = preds[self.key]
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']

        label_pose_norm = (label_pose-label_mean)/label_std
        pred_pose_norm = (pred_pose-label_mean)/label_std
        return self.loss_single.forward(pred_pose_norm,label_pose_norm)

class StaticDenormalizedLoss():
    """
    Denormalize by std and mean before loss computation. Should improve output statistics to be unit variance, but not alter loss
    """
    def __init__(self, key, loss_single):
        super(StaticDenormalizedLoss, self).__init__()
        self.key = key
        self.loss_single = loss_single

    def forward(self, preds, labels):
        pred_pose_norm = preds[self.key]
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        pred_pose = pred_pose_norm*label_std+label_mean
        return self.loss_single.forward(pred_pose, label_pose)

class RunningMomentNormalizedLoss(RunningMomentModule):
    """
    Compute running mean and std and normalize by it before loss computation
    """
    def __init__(self, loss):
        super(RunningMomentNormalizedLoss, self).__init__()
        self.loss = loss

    def forward(self, pred, label):
        self.updateRunningMoments(label.data)
        mean, std = self.getMeanStdRegularized()

        label_norm = (label-mean)/std
        pred_norm  = (pred-mean)/std

        return self.loss(pred_norm, label_norm)

class RunningMomentUpscaling(torch.nn.Module):
    """
    Compute running mean and std and normalize by it before loss computation
    """
    def __init__(self, loss):
        super(RunningMomentUpscaling, self).__init__()

    def forward(self, x_norm):
        mean, std = self.getMeanStdRegularized()
        x_unnorm = x_norm * std + mean
        if not self.train:
            return x_unnorm
        else: # update the mean and std
            mean_, std_ = self.runningMomentsWithoutUpdate(x_unnorm)
            x_unnorm_ = x_norm * std_ + mean_
            # now update really, but without saving gradients somehow (.data)
            self.updateRunningMoments(x_unnorm.data)

            # debugging
            mean__, std__ = self.getMeanStdRegularized(x_norm.data)
            assert torch.norm(mean_.data-mean__) == 0
            assert torch.nrm(std_.data-std__) == 0

            return x_unnorm_ # return the unnormalized


class TwoPersonDictMinLoss(torch.nn.Module):
    """
    Compute running mean and std and normalize by it before loss computation
    """
    def __init__(self, pose_loss, key='3D'):
        super(TwoPersonDictMinLoss, self).__init__()
        self.pose_loss = pose_loss
        self.pose_key = key

    def forward(self, pred_dict, label_dict):
        label = label_dict[self.pose_key]
        label_flipped = label_dict[self.pose_key][:,(1,0),:,:]
        # get rid of spatial transformer dim
        batch_size  = label.shape[0]
        num_persons = label.shape[1]
        num_joints  = label.shape[2]
        point_dim   = label.shape[3]
        label = label.view(batch_size*num_persons,num_joints*point_dim)
        label_flipped = label_flipped.view(label.shape)
        pred  = pred_dict[self.pose_key].view(label.shape)

        label_dict_s0 = label_dict.copy()
        label_dict_s0[self.pose_key] = label
        label_dict_s1 = label_dict.copy()
        label_dict_s1[self.pose_key] = label_flipped


        # compute per element error, then take min
        err0 = torch.mean(self.pose_loss(pred_dict, label_dict_s0),dim=1)
        err1 = torch.mean(self.pose_loss(pred_dict, label_dict_s1),dim=1)
        err_min, arg_min = torch.min(torch.stack([err0,err1]), dim=0)
        # mask was needed to handle missing labels
        #mask = (torch.sum(label,dim=1) != 0).float()
        #err = err_min * mask
        #err = torch.sum(err) / torch.sum(mask)
        err = torch.mean(err)

        # old loss, where pose_loss averages over all elements
        #diffs = []
        #for i in range(batch_size//2):
        #    diff0 = self.pose_loss(pred[2*i:2*i+2],label[2*i:2*i+2])
        #    diff1 = self.pose_loss(pred[2*i:2*i+2],label_flipped[2*i:2*i+2])
        #    diffs.append(min(diff0, diff1) * mask[2*i] * mask[2*i+1])
        #err = sum(diffs) / torch.sum(mask)
        return err

class TwoPersonMinLoss(torch.nn.Module):
    """
    Compute running mean and std and normalize by it before loss computation
    """
    def __init__(self, pose_loss):
        super(TwoPersonMinLoss, self).__init__()
        self.pose_loss = pose_loss

    def forward(self, pred, label):
        # get rid of spatial transformer dim
        batch_size_times_ST  = label.shape[0]
        #num_persons = label.shape[1]
        num_joints  = label.shape[1]
        point_dim   = label.shape[2]
        label_flipped = label.view([batch_size_times_ST//2,-1,num_joints,point_dim])[:,(1,0),:,:]
        label = label.view(batch_size_times_ST,num_joints*point_dim)
        label_flipped = label_flipped.view(label.shape)

        # compute per element error, then take min
        err0 = torch.mean(self.pose_loss(pred, label),dim=1)
        err1 = torch.mean(self.pose_loss(pred, label_flipped),dim=1)
        err_min, arg_min = torch.min(torch.stack([err0,err1]), dim=0)
        err = torch.mean(err_min)

        return err


class MultiPersonSimpleLoss(torch.nn.Module):
    """
    Compute running mean and std and normalize by it before loss computation
    """
    def __init__(self, pose_loss, key='3D'):
        super(MultiPersonSimpleLoss, self).__init__()
        self.pose_loss = pose_loss
        self.pose_key = key

    def forward(self, pred_dict, label_dict):
        # reshape the labels to match with the prediction
        label = label_dict[self.pose_key]
        batch_size  = label.shape[0]
        num_persons = label.shape[1]
        num_joints  = label.shape[2]
        point_dim   = label.shape[3]
        label = label.view(batch_size*num_persons,num_joints*point_dim)
        label_dict_flattened = label_dict.copy()
        label_dict_flattened[self.pose_key] = label
        if 'pose_mean' in label_dict:
            label_dict_flattened['pose_mean'] = label_dict_flattened['pose_mean'].view(label.shape)
        if 'pose_std' in label_dict:
            label_dict_flattened['pose_std'] = label_dict_flattened['pose_std'].view(label.shape)

        # compute per element error, then take mean
        return self.pose_loss.forward(pred_dict, label_dict_flattened)#) ,dim=1)
        # mask is needed to handle missing labels
        #mask = (torch.sum(label,dim=1) != 0).float()
        #err = err * mask
        #err = torch.sum(err) / torch.sum(mask)

class CriterionNMPJPEBelowTreshold(torch.nn.Module):
    """
    Compute detection rate, number of poses that are below a treshold
    """
    def __init__(self, treshold):
        super(CriterionNMPJPEBelowTreshold, self).__init__()
        self.pose_loss = MPJPECriterion(weight=1, reduction='none')
        self.treshold = treshold

    def forward(self, pred_dict, label_dict):
        loss = torch.mean(self.pose_loss.forward(pred_dict, label_dict),dim=1)
        assert len(loss.shape) == 1
        detection_rate = torch.sum(loss<self.treshold).float()/loss.shape[0]
        return detection_rate

