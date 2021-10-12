import numpy as np
import matplotlib.pyplot as plt
import csv

from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from PlottingUtil import util
from CropCompensation import projection

def figureWithoutBorders():
    fig = plt.figure(100)
    plt.clf()    
    axes = plt.axes([0,0,1,1])
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    axes.set_axis_off()
    axes.axis('equal')
    fig.add_axes(axes)
    return fig
    
def saveFigureWithoutBorders(fig,figName,resolution=512):
    dpi=128
    width_inch = resolution/dpi
    #print('width_inch',width_inch)

    plt.plot(0,0,'or')
    plt.plot(0,resolution,'or')
    plt.plot(resolution,0,'or')
    plt.plot(resolution,resolution,'or')
    plt.xlim([0.0,resolution])
    plt.ylim([0.0,resolution])

    fig.set_size_inches(width_inch,width_inch,forward=True)
    #plt.tight_layout()
    plt.show(block=False)
    
    fig.savefig(figName, bbox_inches=0, pad_inches=0, dpi=dpi)
    
width = 1

# box example
points = np.array([[-width,-width,-width],[width,-width,-width],[width,width,-width],[-width,width,-width],[-width,width,width],[width,width,width],[width,-width,width],[-width,-width,width]]).T
bones  = [[0,1],[2,3],[0,3],[1,2], [4,5],[6,7],[5,6],[4,7], [-0,7],[1,6],[2,5],[3,4]]
bones_front = [[0,1],[2,3],[0,3],[1,2]]
bones_side = [[-0,7],[1,6],[2,5],[3,4]]
bones_back  = [[4,5],[6,7],[5,6],[4,7]]

# intrinsics
resolution = 512
f = 500 # focal length
K = [[f, 0, 0, 0.5*f],
     [0, f, 0, 0.5*f],
     [0, 0, 1, 0  ],
     [0, 0, 0, 1  ]]

path='/Users/rhodin/Documents/Paper/FirstOrderCropping/'

numSamples = 100
turn = numSamples/3
for fi in range(0,100):
    
    print('Processing iteration {:d} of {:d}'.format(fi+1,numSamples))
    
    # move box
    z = 10
    if fi < turn:
        x = (min(fi,turn) / turn - 0.5) * 6
        y = 2.5
    elif fi < 1.5*turn:
        x = 0.5 * 6
        y = 2.5 - (max(fi-turn,0) / turn) * 10
    elif fi < 2*turn:
        x = 0.5*6 - (max(fi-1.5*turn,0) / turn) * 6
        y = -2.5  + (max(fi-1.5*turn,0) / turn) * 5
    else:
        x = 0       
        y = 0
        z = z - (max(fi-2*turn,0) / turn) * 20
        
    offset = np.array([x, y, z])
    points_3d_w = points + offset.reshape(3,1);

    center_3d_w = np.mean(points_3d_w,1)
    center_2d_w = projection.perspective(np.matrix(center_3d_w).T, K=K)

    points_2d_w_persp    = projection.perspective(points_3d_w, K=K)
    points_2d_w_weak     = projection.weak       (points_3d_w, K=K)
    points_2d_w_para     = projection.para       (points_3d_w, K=K)
    points_2d_w_para_rot = projection.para_rot   (points_3d_w, K=K)
    
    resolution_crop = resolution*0.35
    points_2d_w_persp_c    = points_2d_w_persp    - center_2d_w + np.array([[resolution_crop*0.5,resolution_crop*0.5,0]]).T
    points_2d_w_weak_c     = points_2d_w_weak     - center_2d_w + np.array([[resolution_crop*0.5,resolution_crop*0.5,0]]).T
    points_2d_w_para_c     = points_2d_w_para     - center_2d_w + np.array([[resolution_crop*0.5,resolution_crop*0.5,0]]).T
    points_2d_w_para_rot_c = points_2d_w_para_rot - center_2d_w + np.array([[resolution_crop*0.5,resolution_crop*0.5,0]]).T
    
    csvName = '{:s}toyExample/global_persp_{:04d}.csv'.format(path,fi)
    with open(csvName, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        vectorOfPositions = np.squeeze(np.array(np.reshape(points_3d_w,(-1,1))))
        csv_writer.writerow(vectorOfPositions)
    
    #print('pose_3d',points_3d_w)
    #print('center_3d_w',center_3d_w)
    
    fig = figureWithoutBorders()
    util.plot_2Dpose(points_2d_w_persp.T,    bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='black')
    figName = '{:s}toyExample/global_persp_{:04d}.png'.format(path,fi)
    saveFigureWithoutBorders(fig,figName,resolution)

    fig = figureWithoutBorders()
    util.plot_2Dpose(points_2d_w_persp.T,    bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='black')
    util.plot_2Dpose(points_2d_w_weak.T,     bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='red')
    util.plot_2Dpose(points_2d_w_para.T,     bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='green')
    util.plot_2Dpose(points_2d_w_para_rot.T, bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='blue')
    figName = '{:s}toyExample/global_all_{:04d}.png'.format(path,fi)
    saveFigureWithoutBorders(fig,figName,resolution)
    
    fig = figureWithoutBorders()
    util.plot_2Dpose(points_2d_w_weak.T,     bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='red')
    figName = '{:s}toyExample/global_weak_{:04d}.png'.format(path,fi)
    saveFigureWithoutBorders(fig,figName,resolution)

    fig = figureWithoutBorders()
    util.plot_2Dpose(points_2d_w_para.T,     bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='green')
    util.plot_2Dpose(points_2d_w_para_rot.T, bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='blue')
    figName = '{:s}toyExample/global_para_{:04d}.png'.format(path,fi)
    saveFigureWithoutBorders(fig,figName,resolution)

    # and now again for the cropped region
    fig = figureWithoutBorders()
    util.plot_2Dpose(points_2d_w_persp_c.T,    bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='black', linewidth=2)
    figName = '{:s}toyExample/crop_persp_{:04d}.png'.format(path,fi)
    saveFigureWithoutBorders(fig,figName,resolution_crop)
    
    fig = figureWithoutBorders()
    util.plot_2Dpose(points_2d_w_weak_c.T,     bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='red', linewidth=3)
    figName = '{:s}toyExample/crop_weak_{:04d}.png'.format(path,fi)
    saveFigureWithoutBorders(fig,figName,resolution_crop)
    
    fig = figureWithoutBorders()
    util.plot_2Dpose(points_2d_w_para_c.T,     bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='green', linewidth=2)
    util.plot_2Dpose(points_2d_w_para_rot_c.T, bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='blue', linewidth=2)
    figName = '{:s}toyExample/crop_para_{:04d}.png'.format(path,fi)
    saveFigureWithoutBorders(fig,figName,resolution_crop)
    
    fig = figureWithoutBorders()
    util.plot_2Dpose(points_2d_w_persp_c.T,    bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='black', linewidth=2)
    util.plot_2Dpose(points_2d_w_weak_c.T,     bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='red', linewidth=3)
    util.plot_2Dpose(points_2d_w_para_c.T,     bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='green', linewidth=2)
    util.plot_2Dpose(points_2d_w_para_rot_c.T, bones=bones_front, bones_dashed=bones_back, bones_dashdot=bones_side, color='blue', linewidth=2)
    figName = '{:s}toyExample/crop_all_{:04d}.png'.format(path,fi)
    saveFigureWithoutBorders(fig,figName,resolution_crop)
    
    #break
    
