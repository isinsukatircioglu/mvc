import matplotlib.pyplot as plt

import csv
import scipy.misc
import torch
import torch.nn.functional as F

import IPython

import camera_transformer

original_image_resolution = torch.FloatTensor([2704, 1520])

K_px = torch.FloatTensor([
     [2704*0.456584543,  0.000000000,  2704*0.498595864],
     [0.000000000,  2704*0.456602871,  2704*0.281708330],
     [0.000000000,  0.000000000,  1.000000000]])

# make intrinsic matrix operate on 0..1 coordinates in both dimensions
K[1,:] = K[1,:]*original_image_resolution[0]/original_image_resolution[1]
print("K\n",K)

cop_pixel_size = torch.Size([3, 3])
positions_px = torch.FloatTensor([[0,0],
                                 [2704, 1520],
#                                 [2704*0.498595864, 1520*0.281708330],
#                                 [18,214]
                                  ])
scale = torch.FloatTensor([0.1,0.1 * original_image_resolution[0]/original_image_resolution[1]])

for position_px in positions_px:
    print("########################################################")
    position_px_unit = position_px / original_image_resolution

    PCL = camera_transformer.PerspectiveCropLayer(cop_pixel_size, "cpu")
    P_virt2orig, R_virt2orig, K_virt = PCL.forward(position_px_unit.unsqueeze(0), scale.unsqueeze(0), K.unsqueeze(0))
    R_orig2virt = R_virt2orig.squeeze().inverse().unsqueeze(0)

    position_px_homo = torch.cat([position_px_unit, torch.FloatTensor([1])])
    position_c_homo = K.inverse() @ position_px_homo
    position_v_homo = R_orig2virt[0] @ position_c_homo
    position_v_homo = position_v_homo / position_v_homo[2]
    position_c_homo_rec = R_virt2orig[0] @ position_v_homo #torch.FloatTensor([0,0,1])
    position_c_homo_rec = position_c_homo_rec / position_c_homo_rec[2]
    print("p_px = ", position_px_homo)
    print("p_c = ", position_c_homo)
    print("p_v = ", position_v_homo)
    print("p_c_rec = ", position_c_homo_rec)
    print("R_orig2virt = ", R_orig2virt)


grid = PCL.perspective_grid(P_virt2orig)
grid = 2*grid-1 # pytorch image coordinates go from -1..1 instead of 0..1

exit()