#import numpy as np
#import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import projection
import csv
import os

#from skimage.transform import warp 
import scipy.misc
import torch
import torch.nn.functional as F

import IPython

import pcl_complex

import matplotlib.patches


path = './data/'
out = './output/'

imgFileName  = path+'images/frame_{:04d}.png'
csvFileName  = path+'annotation/annotation_p0002_c03_f{:04d}.csv'

#    intrinsics
K_px = torch.FloatTensor([
     [2704*0.456584543,  0.000000000,  2704*0.498595864],
     [0.000000000,  2704*0.456602871,  2704*0.281708330],
     [0.000000000,  0.000000000,  1.000000000]])

image_resolution_px = torch.FloatTensor([2704, 1520])

frame_start = 2184
frame_end = 2230

scale = torch.FloatTensor([0.05*image_resolution_px[0], 0.08*2*image_resolution_px[1]])
crop_resolution_px = torch.Size([400, 600])

PCL = pcl_complex.PerspectiveCropLayer(crop_resolution_px, "cpu")

positions = torch.FloatTensor([
[18,1520-214], # 81
[52,1520-224],
[90,1520-240],
[88,1520-244],
[130,1520-250],
[168,1520-266],
[204,1520-280],
[238,1520-290],
[282,1520-304],
[322,1520-318],# 90
[362,1520-330],
[400,1520-348],
[446,1520-364],
[490,1520-378],
[534,1520-390],
[576,1520-408],
[620,1520-422],
[664,1520-436],
[710,1520-448],
[756,1520-466], #100
[804,1520-488],
[854,1520-496],
[900,1520-512],
[952,1520-530],
[996,1520-554],
[1054,1520-560],
[1106,1520-578],
[1159,1520-592],
[1210,1520-614],
[1268,1520-634], # 110
[1322,1520-650],
[1384,1520-664],
[1444,1520-690],
[1504,1520-706],
[1568,1520-726],
[1634,1520-742],
[1698,1520-764],
[1762,1520-788],
[1838,1520-800],
[1908,1520-826],# 220
[1985,1520-850],
[2066,1520-876],
[2142,1520-896],
[2224,1520-928],
[2312,1520-952],
[2396,1520-976],
[2494,1520-1002],
[2592,1520-1036],
[2690,1520-1060],
[2842,1520-1114] #230
]) #120

convention = 'pytorch'
#convention = 'unit'
convention = 'px'

def plotPerspectiveCrop(ax_img, positions, scales, width, height, Ks, overlay_both):
    linewidth=1
    if Ks is not None:
        PCL_sparse = pcl_complex.PerspectiveCropLayer(torch.Size([3, 3]), "cpu")
        if convention == 'pytorch':
            P_virt2orig, R_virt2orig, K_virt = PCL_sparse.forward(positions, 2*scales, Ks)
        else:
            P_virt2orig, R_virt2orig, K_virt = PCL_sparse.forward(positions, scales, Ks)

        grid_sparse = PCL_sparse.perspective_grid(P_virt2orig)

        if convention == 'pytorch':
            xv = (grid_sparse[0,:,:,0].numpy()+1)/2 * width
            # 1- due to image y axis going downwards
            yv = (grid_sparse[0,:,:,1].numpy()+1)/2 * height
        elif convention == 'unit':
            xv = grid_sparse[0, :, :, 0].numpy()*width
            yv = grid_sparse[0, :, :, 1].numpy()*height
        else:
            xv = grid_sparse[0,:,:,0].numpy()
            yv = height-grid_sparse[0,:,:,1].numpy()

        # plot grid (thereby twice, once horizontal once vertical)
        color="green"
        ax_img.plot(xv,yv,'-',linewidth=linewidth,color=color)
        ax_img.plot(xv.transpose(),yv.transpose(),'-',linewidth=linewidth,color=color)
        if 1:
            ax_img.plot(xv,yv,'.',markersize=3,color=color)

    if Ks is None or overlay_both:
        color = "red"
        if convention == 'pytorch':
            rect = matplotlib.patches.Rectangle(
                (width  * ((positions[0,0]+1)/2-scales[0,0]/2), height * ((positions[0,1]+1)/2-scales[0,1]/2)),
                scales[0,0] * width, scales[0,1] * height,
                linewidth=linewidth, linestyle='dashed', edgecolor=color, facecolor='none')
        elif convention == 'unit':
            rect = matplotlib.patches.Rectangle(
                (width  * (positions[0,0]-scales[0,0]/2), height * (positions[0,1]-scales[0,1]/2)),
                scales[0,0] * width, scales[0,1] * height,
                linewidth=linewidth, linestyle='dashed', edgecolor=color, facecolor='none')
        else:
            rect = matplotlib.patches.Rectangle(
                (positions[0,0]-scales[0,0]/2, height-positions[0,1]-scales[0,1]/2),
                scales[0,0], scales[0,1],
                linewidth=linewidth, linestyle='dashed', edgecolor=color, facecolor='none')
        ax_img.add_patch(rect)
        # plot crop center
        if 1:
            if convention == 'pytorch':
                ax_img.plot([width*(positions[0,0]+1)/2], [height*(positions[0,1]+1)/2], marker='o', ms=3, color=color)
            elif convention == 'unit':
                ax_img.plot([width * positions[0, 0]], [height * positions[0, 1]], marker='o', ms=3, color=color)
            else:
                ax_img.plot([positions[0,0]], [(height-positions[0,1])], marker='o', ms=3, color=color)

for f in range(frame_start, frame_end, 1):
    relative_duration = (f-frame_start) / (frame_end-frame_start)
    #position = ((1-relative_duration)*position_start + relative_duration*position_end) / image_resolution_px
    position = positions[f-2181]

    # load image
    img_orig = torch.FloatTensor(scipy.misc.imread(imgFileName.format(f)))/256
    img_torch = img_orig.permute([2,0,1])  # note the flipping to place origin to bottom-left

    # make batches
    Ks = K_px.unsqueeze(0)
    positions = position.unsqueeze(0)
    scales = scale.unsqueeze(0)

    # pytorch convention as input (BROKEN: scale is in different coordinates then position (factor of two)
    positions_torch = PCL.point_px2torch(positions, image_resolution_px)
    Ks_torch = PCL.K_px2K_torch(Ks, image_resolution_px)
    scales_torch = PCL.scale_px2torch(scales, image_resolution_px)

    # parameterize image from 0..1 in both directions, y pointing downwards
    positions_unit = PCL.point_px2unit(positions, image_resolution_px)
    Ks_unit = PCL.K_px2K_unit(Ks, image_resolution_px)
    scales_unit = PCL.scale_px2unit(scales, image_resolution_px)

    # Note, unsqueeze to mimick batch in first dimension
    #position_projective = torch.cat([postion,torch.FloatTensor([1])])
    if convention == 'pytorch':
        P_virt2orig, R_virt2orig, K_virt = PCL.forward(positions_torch, 2*scales_torch, Ks_torch)
    elif convention == 'unit':
        P_virt2orig, R_virt2orig, K_virt = PCL.forward(positions_unit, scales_unit, Ks_unit)
    else:
        P_virt2orig, R_virt2orig, K_virt = PCL.forward(positions, scales, Ks)

    grid = PCL.perspective_grid(P_virt2orig)

    # Testing two kinds of coordinates
    # 1) pixel coordinates with the origin in the bottom left
    # 2) pytorch coordinates with the origin in the center and y-axis pointing downwards
    if convention == 'px': # pytorch convention as input
        grid_torch = grid.clone()
        grid_torch[:,:,:,0] =   2*grid[:,:,:,0]/image_resolution_px[0]-1
        grid_torch[:,:,:,1] = -(2*grid[:,:,:,1]/image_resolution_px[1]-1)
        grid_px = grid
    elif convention == 'pytorch':
        grid_torch = grid # no conversion needed?
        grid_px = grid.clone()
        grid_px[:,:,:,0] = ( grid[:,:,:,0]+1)/2*image_resolution_px[0]
        grid_px[:,:,:,1] = (-grid[:,:,:,1]+1)/2*image_resolution_px[1]
    elif convention == 'unit':
        grid_torch = grid.clone()
        grid_torch[:,:,:,0] = 2*grid[:,:,:,0]-1
        grid_torch[:,:,:,1] = 2*grid[:,:,:,1]-1
        grid_px = grid.clone()
        grid_px[:,:,:,0] = ( grid[:,:,:,0])*image_resolution_px[0]
        grid_px[:,:,:,1] = (-grid[:,:,:,1])*image_resolution_px[1]
    #IPython.embed()

    img_crop_torch = F.grid_sample(img_torch.unsqueeze(0), grid_torch)
    
    # back to python ordering of dimensions
    img_crop_perspective = img_crop_torch.squeeze().permute([1,2,0])

    # 1-to flip y axis from upwards (camera coordinates) to downwards pointing (image coordinates)
    s = (torch.clamp(torch.FloatTensor([positions_torch[0,0]-scales_torch[0,0]/2, 1-(positions_torch[0,1]+scales_torch[0,1]/2)]),0,1) * image_resolution_px).int().tolist()
    e = (torch.clamp(torch.FloatTensor([positions_torch[0,0]+scales_torch[0,0]/2, 1-(positions_torch[0,1]-scales_torch[0,1]/2)]),0,1) * image_resolution_px).int().tolist()
    img_crop_rect = img_orig[s[1]:e[1],s[0]:e[0],:]

    fig = plt.figure()
    plt.imshow(img_crop_perspective) #, origin='lower')
    plt.show()
    plt.close(fig)
    scipy.misc.imsave(out+"crop_perspective_{:06d}.png".format(f), img_crop_perspective)
    if 0:
        plt.imshow(img_crop_rect)#, origin='lower')
        plt.show()
        plt.close(fig)
        scipy.misc.imsave(out+"crop_rectangular_{:06d}.png".format(f), img_crop_rect)

    if 1:
        fig, ax = plt.subplots(frameon=False)
        #ax.set_axis_off()
        ax.imshow(img_orig)#, origin='lower')
        height, width = img_orig.shape[0], img_orig.shape[1]
        if convention == 'pytorch':
            plotPerspectiveCrop(ax, positions_torch, scales_torch, height=img_orig.shape[0], width=img_orig.shape[1], Ks=Ks_torch, overlay_both=True)
        elif convention == 'unit':
            plotPerspectiveCrop(ax, positions_unit, scales_unit, height=img_orig.shape[0], width=img_orig.shape[1],
                                Ks=Ks_unit, overlay_both=True)
        else:
            plotPerspectiveCrop(ax, positions, scales, height=img_orig.shape[0], width=img_orig.shape[1], Ks=Ks,
                                overlay_both=True)
        plt.show()
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Note, first argument is height, otherwise the image is flipped...
        fig.savefig(out+"img_overlay_{:06d}.png".format(f), bbox_inches=extent)
        plt.close(fig)



    exit()


with open(csvFileName, 'r') as ground_truth_file:
    csv_reader = csv.reader(ground_truth_file)
    for line in csv_reader:
        pose_3d_in = [float(elem) for elem in line[0:]]

    pose_3d_in = np.mat(pose_3d_in)
    pose_3d_in = pose_3d_in.reshape([-1,3]).T
    pose_3d_in = np.multiply(pose_3d_in,np.mat([1,-1,1]).T)
    pose_3d    = pose_3d_in #pose_3d_in[:,util.cpm2h36m]
    print('pose_3d',pose_3d)

    pose_3d_center = pose_3d[:,0]
    pose_3d_centered = pose_3d - pose_3d_center
    print('pose_3d_centered\n',pose_3d_centered)

    # plot results
    fig = plt.figure(0)
    ax_img  = fig.add_subplot(1,2,1)
    plt.xlim([0.0,img_full.shape[1]])
    plt.ylim([0.0,img_full.shape[0]])
    ax_img.set_axis_off()
    ax_img.imshow(img_full)

    ax_3d   = fig.add_subplot(122, projection='3d')
    ax_3d.xaxis.set_visible(False)
    ax_3d.yaxis.set_visible(False)
    ax_3d.set_axis_off()
    ax_3d.axis('equal')
    util.plot_3Dpose(ax_3d, pose_3d_centered, bones)

    #plt.show()
    #plt.show(block=False)
    dpi=600
    #fig.savefig('{:s}correction_{:04d}.png'.format(path,fi), dpi=dpi)
