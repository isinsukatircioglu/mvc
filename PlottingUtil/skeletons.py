import numpy as np
import numpy.linalg as la
import torch

def computeBoneLengths(pose_tensor, bones):
    pose_tensor_3d = pose_tensor.view(-1,3)
    length_list = [torch.norm(pose_tensor_3d[bone[0]] - pose_tensor_3d[bone[1]])
                     for bone in bones]
    return length_list

def computeBoneLengths_np(pose_tensor, bones):
    pose_tensor_3d = pose_tensor.reshape(-1,3)
    length_list = [la.norm(pose_tensor_3d[bone[0]] - pose_tensor_3d[bone[1]])
                     for bone in bones]
    return length_list
