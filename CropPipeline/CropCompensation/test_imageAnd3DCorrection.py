import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')

from PlottingUtil import util
from PlottingUtil import skeletons
from CropCompensation import projection

import csv
import os

from skimage.transform import warp 
from skimage import data 
from scipy import misc
#from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

bones_dashed = []
bones_dotted = []

bone_radius = 20

path = '/Users/rhodin/Documents/Paper/FirstOrderCropping/skiExample/'
inToInternalSkeleton = skeletons.cpm2h36m
bones = skeletons.bones_h36m
for fi in range(2184,2224):
#for fi in range(2184,2185):
    imgFileName  = '{:s}cam_3/frame_{:04d}.jpg'.format(path,fi)
    cropFileName = '{:s}annotation_p0002_c03_f{:06d}.jpg'.format(path,fi)
    csvFileName  = '{:s}annotation_p0002_c03_f{:06d}.jpg.csv'.format(path,fi)
    
#inToInternalSkeleton = range(0,8)
#bones_cube  = [[0,1],[2,3],[0,3],[1,2], [4,5],[6,7],[5,6],[4,7], [-0,7],[1,6],[2,5],[3,4]]
#bones_cube_front = [[0,1],[2,3],[0,3],[1,2]]
#bones_cube_side = [[-0,7],[1,6],[2,5],[3,4]]
#bones_cube_back  = [[4,5],[6,7],[5,6],[4,7]]
#bones = bones_cube_front
#bones_dashed = bones_cube_side
#boned_dotted = bones_cube_back
#path = './toyExample/'
#for fi in range(0,100):
#    imgFileName  = '{:s}global_persp_{:04d}.png'.format(path,fi)
#    cropFileName = '{:s}crop_persp_{:04d}.png'.format(path,fi)
#    csvFileName  = '{:s}global_persp_{:04d}.csv'.format(path,fi)


    if not os.path.exists(csvFileName):
        print("No csv file for '{}'".format(csvFileName))
        continue
        
    img_full = np.flipud(misc.imread(imgFileName))  # note the flipping to place origin to bottom-left
    img_crop = np.flipud(misc.imread(cropFileName)) # note the flipping

    with open(csvFileName, 'r') as ground_truth_file:
        csv_reader = csv.reader(ground_truth_file)
        for line in csv_reader:
            pose_3d_in = [float(elem) for elem in line[0:]]
        
    pose_3d_in = np.mat(pose_3d_in)
#    pose_3d_in = pose_3d_in.reshape([3,-1])
    pose_3d_in = pose_3d_in.reshape([-1,3]).T
    pose_3d_in = np.multiply(pose_3d_in,np.mat([1,-1,1]).T)
    pose_3d    = pose_3d_in[:,inToInternalSkeleton]
    print('pose_3d',pose_3d)

    pose_3d_mean = np.mean(pose_3d,1)
    print('pose_3d_mean', pose_3d_mean)
    pose_3d_center = pose_3d_mean #pose_3d[:,0]
    pose_3d_centered = pose_3d - pose_3d_center
    print('pose_3d_centered\n',pose_3d_centered)

    # compute compensation from crop location (projection of global coords center)
    K_canonical = np.mat(np.eye(4))
    pose_2d_center = pose_3d_center / pose_3d_center[2]
    K_crop = [[1, 0, 0, img_crop.shape[0]/2],
              [0, 1, 0, img_crop.shape[1]/2],
              [0, 0, 1, 0  ],
              [0, 0, 0, 1  ]]
    #[R_world, _, S_crop_local] = projection.getImageAnd3DCorrection(np.mat(pose_2d_center), K_canonical, K_cropLocal=K_crop)
    #[R_world, _, S_crop_local] = projection.getImageAnd3DCorrection(np.mat(pose_2d_center), K_canonical, decomposition=projection.para_getIsometricDecomposition, K_cropLocal=K_crop)
    [R_world, _, S_crop_local] = projection.getImageAnd3DCorrection(np.mat(pose_2d_center), K_canonical, decomposition=projection.persp_getIsometricDecomposition, K_cropLocal=K_crop)
    
    pose_3d_centered_corr = np.matmul(R_world[0:3,0:3], pose_3d_centered)
    img_crop_corrected    = warp(img_crop, inv(S_crop_local), order=3)
    
    # compensate skeleton with shearing (WARNING, non-isometric, not applicable to kinematic skeleton)
    [P, S_world] = projection.para_getShearDecomposition(np.mat(pose_2d_center))
    pose_3d_centered_sheared = np.matmul(S_world[0:3,0:3], pose_3d_centered)
    
    # plot results
    fig = plt.figure(0)
    ax_img  = fig.add_subplot(2,3,1)
    plt.xlim([0.0,img_full.shape[1]])
    plt.ylim([0.0,img_full.shape[0]])
    ax_img.set_axis_off()
    ax_img.imshow(img_full)

    ax_3d   = fig.add_subplot(232, projection='3d')
    ax_3d.xaxis.set_visible(False)
    ax_3d.yaxis.set_visible(False)
    ax_3d.set_axis_off()
    ax_3d.axis('equal')
    util.plot_3Dpose(ax_3d, pose_3d_centered, bones=bones, radius=bone_radius)#, bones_lines=bones, bones_dashed=bones_dotted, bones_dashdot=bones_dashed)

    ax_crop = fig.add_subplot(2,3,3)
    plt.xlim([0.0,img_crop.shape[0]])
    plt.ylim([0.0,img_crop.shape[1]])
    ax_crop.set_axis_off()
    ax_crop.imshow(img_crop)
    
    ax_3d_s   = fig.add_subplot(2,3,4, projection='3d')
    ax_3d_s.xaxis.set_visible(False)
    ax_3d_s.yaxis.set_visible(False)
    ax_3d_s.set_axis_off()
    ax_3d_s.axis('equal')
    util.plot_3Dpose(ax_3d_s, pose_3d_centered_sheared, bones=bones, radius=bone_radius)#[], bones_lines=bones, bones_dashed=bones_dotted, bones_dashdot=bones_dashed)

    ax_3d_c   = fig.add_subplot(2,3,5, projection='3d')
    ax_3d_c.xaxis.set_visible(False)
    ax_3d_c.yaxis.set_visible(False)
    ax_3d_c.set_axis_off()
    ax_3d_c.axis('equal')
    util.plot_3Dpose(ax_3d_c, pose_3d_centered_corr, bones=bones, radius=bone_radius)#[], bones_lines=bones, bones_dashed=bones_dotted, bones_dashdot=bones_dashed)

    ax_crop_c = fig.add_subplot(2,3,6)
    plt.xlim([0.0,img_crop_corrected.shape[0]])
    plt.ylim([0.0,img_crop_corrected.shape[1]])
    ax_crop_c.set_axis_off()
    ax_crop_c.imshow(img_crop_corrected)

    #plt.show()
    plt.show(block=False)
    dpi=600
    fig.savefig('{:s}correction_persp{:04d}.png'.format(path,fi), dpi=dpi)
    #break
exit()




