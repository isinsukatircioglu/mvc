import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import warp 
from skimage import data 
from scipy import misc
from numpy.linalg import inv

import random
random.seed(1)

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from PlottingUtil import util
from CropCompensation import projection


def figureWithoutBorders(id, resolution_x, resolution_y):
    fig = plt.figure(id)
    plt.clf()    
    #axes = plt.axes([0,0,1,1])
    #axes.xaxis.set_visible(True)
    #axes.yaxis.set_visible(True)
    #axes.set_axis_on()
    #axes.axis('equal')
    #fig.add_axes(axes)
    plt.xlim([0.0,resolution_x])
    plt.ylim([0.0,resolution_y])
    return fig

def saveFigureWithoutBorders(fig,figName=None,resolution=512):
    dpi=128
    width_inch = resolution/dpi
    #print('width_inch',width_inch)


    fig.set_size_inches(width_inch,width_inch,forward=True)
    #plt.tight_layout()
    if figName is not None:
        plt.show(block=False)
        fig.savefig(figName, bbox_inches=0, pad_inches=0, dpi=dpi)
    
img = data.coffee()
img = misc.imread('/Users/rhodin/Documents/Paper/FirstOrderCropping/toyExample/global_persp_0001.png')
#img = misc.imread('./toyExample/global_para_0001.png')
img = np.flipud(img) # to make the origin in the bottom left, as when plotting values
#img = img * 0.9      # to make boundaries visible

#img = data.astronaut()
#img = data.checkerboard()
#img = data.chelsea()

resolution_x = img.shape[0]
resolution_y = img.shape[1]
f = 500
# note K is in R3 homogeneous coordinates
K = np.mat([[f, 0, 0, 0.5*resolution_x],
            [0, f, 0, 0.5*resolution_y],
            [0, 0, 1, 0  ],
            [0, 0, 0, 1  ]])
print('f:',f)
print('K:\n',K)
     
#crop_corner_px = np.array([200,440, 1])
#crop_width_px  = np.array([100,100, 0])
crop_corner_px = np.array([20,300, 1])
crop_width_px  = np.array([150,150, 0])
crop_center_px = crop_corner_px+crop_width_px/2.
box = np.zeros((5,3))
box[0,:] = crop_corner_px
box[1,:] = crop_corner_px+crop_width_px*np.array([1,0,0])
box[2,:] = crop_corner_px+crop_width_px
box[3,:] = crop_corner_px+crop_width_px*np.array([0,1,0])
box[4,:] = box[0,:]
print('box:\n',box)

points_crop   = np.mat([[random.randint(0,100),random.randint(0,100),0] for p in range(0,5)])
points_global = crop_corner_px + points_crop

# display original figure and crop mark
fig = figureWithoutBorders(0, resolution_x,resolution_y)
plt.imshow(img)
plt.plot(crop_center_px[0], crop_center_px[1], 'o', linewidth=1, color='red')
plt.plot(box[:,0], box[:,1], '-', linewidth=1, color='red')
plt.plot(points_global[:,0], points_global[:,1], '.', color='yellow')
#plt.show()

# display corrected figure and crop mark
fig = figureWithoutBorders(1, resolution_x,resolution_y)
#crop_center_px_flipped = crop_center_px[[1,0,2]]
#[R_world, S_img, _] = projection.getImageAnd3DCorrection(np.mat(crop_center_px_flipped).T, K)
[R_world, S_img, _] = projection.getImageAnd3DCorrection(np.mat(crop_center_px).T, K)
#S_img[[0,1],:] = S_img[[1,0],:] # because of different axis convention of images
#S_img = np.mat([[1, 0, 100],
#                [0, 1, 0],
#                [0,0,1]])

#print('S_img:\n',S_img)
#print('S_img2:\n',S_img2)
#print('points_global:\n',points_global)
#S_img2 = np.mat(np.eye(3))
#S_img2[0:3,0:3] = S_img
#S_img2[0:2,0:2] = (S_img[0:2,0:2])

# note, the warp function assumes the inverse matrix as argument
img_corrected = warp(img, inv(S_img), order=3)
points_global_corr = np.matmul(S_img,points_global.T).T
print('S_img:\n',S_img)

plt.imshow(img_corrected)
plt.plot(crop_center_px[0], crop_center_px[1], 'o', linewidth=1, color='red')
plt.plot(box[:,0], box[:,1], '-', linewidth=1, color='red')
plt.plot(points_global_corr[:,0], points_global_corr[:,1], '.', color='yellow')
#plt.show()

# display original crop
fig = figureWithoutBorders(2, crop_width_px[0],crop_width_px[1])
img_crop_orig      = img          [crop_corner_px[1]:crop_corner_px[1]+crop_width_px[1], 
                                   crop_corner_px[0]:crop_corner_px[0]+crop_width_px[0]]
plt.imshow(img_crop_orig)

# display corrected crop, computed from original image
fig = figureWithoutBorders(3, crop_width_px[0],crop_width_px[1])
img_crop_corrected = img_corrected[crop_corner_px[1]:crop_corner_px[1]+crop_width_px[1], 
                                   crop_corner_px[0]:crop_corner_px[0]+crop_width_px[0]]
plt.imshow(img_crop_corrected)

# display corrected crop, computed from original crop
fig = figureWithoutBorders(4, crop_width_px[0],crop_width_px[1])
crop_center_px_cropRelative = crop_center_px - crop_corner_px
crop_center_px_cropRelative[2] = crop_center_px[2]
Shift_crop = np.mat([[1, 0, 0, -crop_corner_px[0]],
                     [0, 1, 0, -crop_corner_px[1]], 
                     [0, 0, 1,  0  ],
                     [0, 0, 0,  1  ]])
K_crop = Shift_crop*K
[R_world_crop,S_crop,_] = projection.getImageAnd3DCorrection(np.mat(crop_center_px_cropRelative).T, K_crop)
img_crop_corrected2 = warp(img_crop_orig, inv(S_crop), order=3)
plt.imshow(img_crop_corrected2)

# display corrected crop, without knowing exact intrinsics
fig = figureWithoutBorders(5, crop_width_px[0],crop_width_px[1])
K_cropLocal = [[1, 0, 0, crop_width_px[0]/2],
               [0, 1, 0, crop_width_px[1]/2],
               [0, 0, 1, 0  ],
               [0, 0, 0, 1  ]]

#print('T_crop:\n',Shift_crop)
#K_cropLocal = Shift_crop*K
#print('K_crop:\n',K_crop)
[R_world_crop,_,S_crop_local] = projection.getImageAnd3DCorrection(np.mat(crop_center_px).T, K, K_cropLocal=K_cropLocal)
img_crop_corrected2           = warp(img_crop_orig, inv(S_crop_local), order=3)
plt.imshow(img_crop_corrected2)



# these are supposed to be the same
print('R_world:\n',R_world)
print('R_world_crop:\n',R_world_crop)
plt.show()


