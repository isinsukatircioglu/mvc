import csv
import cv2
from subprocess import call
import sys
inputFrameFile = "/cvlabdata1/cvlab/dataset_ski_drone_goPro/Glacier_02_15/rough_sync/ex_0003_extended/selected_frames.csv"
inputImages = "/cvlabdata1/cvlab/dataset_ski_drone_goPro/Glacier_02_15/rough_sync/ex_0003_extended/undistorted_images_jpg/cam_{}/Images/frame_{}.jpg"
outputImages = "/cvlabdata1/cvlab/dataset_ski_drone_goPro/Glacier_02_15/rough_sync/ex_0003_extended/undistorted_images_jpg_selected/cam_{}/Images/frame_{}.jpg" 

frame_list = []
with open(inputFrameFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        frame_list.append(int(line[0]))

frameCountNew = 0
for fi in frame_list:
    frameCountNew = frameCountNew + 1
    for ci in range(0,4):
        source = inputImages.format(ci,fi)
        dest   = outputImages.format(ci,frameCountNew)
        print("mv"+source+" "+dest);
        call(["mv", source, dest])

#    sys.exit("Stop")

