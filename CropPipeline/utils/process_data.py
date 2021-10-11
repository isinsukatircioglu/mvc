import numpy
import os
import sys
import glob
import h5py
import matplotlib.image as mpimg
import scipy.misc
import subprocess
import itertools
import pickle

base_folder = '/cvlabdata1/cvlab/dataset_boxing/Step3_Data/PickleTest/'

seq = 'Sequence' + sys.argv[1]
actor = sys.argv[2]
cam = 'cam_' + sys.argv[3]

data_folder = base_folder + seq + '/' + actor + '/' + cam + '/'

################
#1)Resize images to 256*256 for stacked hourglass heatmap generation

files = glob.glob(data_folder + '*.jpg')
newpath = ''
for f in files:
    image_name = os.path.basename(f)
    newpath = data_folder + '/Resized_Images/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    img = mpimg.imread(f)
    img = scipy.misc.imresize(img, (256, 256))
    scipy.misc.imsave(newpath + image_name, img)

#################
#2)Create individual image h5 after resizing the images back to 128*128

files = glob.glob(newpath + '*.jpg')

for f_ind, f in enumerate(files):
    frame_no = numpy.zeros(1)
    img = mpimg.imread(f)
    imgname = os.path.splitext(os.path.basename(f))[0]
    frame_no[0] = imgname.split('frame_')[1]
    newf = newpath + os.path.splitext(os.path.basename(f))[0] + '_img.h5'
    img = scipy.misc.imresize(img, (128, 128))
    img = numpy.transpose(img, (2, 0, 1))
    hfile = h5py.File(newf, 'w')
    hfile.create_dataset('imgs', data=img)
    hfile.create_dataset('index', data=frame_no)
    hfile.close()

#################
#3)H5Concat Images

files = glob.glob(newpath + '*_img.h5')
allimg = numpy.zeros((len(files), 3 * 128 * 128))
allimgind = numpy.zeros(len(files))

for i, f in enumerate(files):
    img = h5py.File(f)
    imgarr = numpy.array(img['imgs']) / 255.0
    allimg[i, :] = imgarr.reshape((3 * 128 * 128))[...]
    allimgind[i] =  numpy.array(img['index'])[0]
    img.close()


allimgh5 = h5py.File(data_folder + seq + '_' + actor + '_' + cam  +'_images.h5', 'w')
allimgh5.create_dataset('imgs', data=allimg)
allimgh5.create_dataset('index', data=allimgind)
allimgh5.close()

#################
#4)Remove individual image h5 files
for filename in glob.glob(newpath + '*_img.h5'):
    os.remove(filename)

#################
#5)Call stacked hourglass
cwd = os.getcwd()
os.chdir("/cvlabdata1/home/katircio/heatmap_code/ski_images_hourglass/")
subprocess.call("CUDA_VISIBLE_DEVICES=1 th run-hg.lua " + newpath + " jpg", shell=True)
os.chdir(cwd)

#################
#6)Concatenate heatmaps
heatmap_filenames = glob.glob(newpath + '*.h5')

allheatmaps = numpy.zeros((len(heatmap_filenames), 16 * 64 * 64))
heatmap_no = numpy.zeros(len(heatmap_filenames))
for i, h in enumerate(heatmap_filenames):
    hm = h5py.File(h)
    hmarr = numpy.array(hm['heatmap'])
    allheatmaps[i, :] = hmarr.reshape((16 * 64 * 64))[...]
    heatmapname = os.path.splitext(os.path.basename(h))[0]
    heatmap_no[i] = heatmapname.split('frame_')[1]
    hm.close()

allheath5 = h5py.File(data_folder + seq + '_' + actor + '_' + cam +'_hourglass_heatmaps.h5', 'w')
allheath5.create_dataset('heatmap', data=allheatmaps)
allheath5.create_dataset('index', data=heatmap_no)
allheath5.close()

#################
#7)Remove individual heatmap h5 files
#for filename in glob.glob(newpath + '*.h5'):
#    os.remove(filename)
#################
#7_v2)Save heatmap paths
os.chdir(data_folder) 
os.chdir('../..')
data = pickle.load(open('test_pickle.pkl', 'rb'))
for filename in glob.glob(newpath + '*.h5'):
    hm_path = filename.split(seq)[1]
    hm_path = hm_path.replace("//","/")
    hm_frame_name = filename.split('frame_')[1].split('.')[0]
    data['actors'][actor][hm_frame_name][sys.argv[3]]['heatmap_path'] = hm_path


#################
#8)Change groundtruth pose order & hip coordinates
#os.chdir(data_folder)
#os.chdir('../..')

#data = pickle.load(open('test_pickle.pkl', 'rb'))
total_frame = len(data['actors'][actor])
gt = numpy.zeros((total_frame, 51))
frame_arr = []

#pkl_file.close()

ind = -1
for i, frm in enumerate(data['actors'][actor].keys()):
    cam_list = data['actors'][actor][frm].keys()
    if sys.argv[3] in cam_list:
        ind = ind + 1
        annotation = data['actors'][actor][frm][sys.argv[3]]['annotations_3D']
        annotation = list(itertools.chain(*annotation))
        gt[ind,:] = numpy.array(annotation)
        frame_arr.append(frm)

gt = gt[0:len(frame_arr)]

lhip = 4;
rhip = 1;
pelvis = 0;

annot_3d = gt
correct_annot_3d = annot_3d;

#Vertical correction for hips
correction3d = annot_3d[:,(3*pelvis):(3*pelvis+3)] -(annot_3d[:,(3*lhip):(3*lhip+3)] + annot_3d[:,(3*rhip):(3*rhip+3)])/2;


#Horizontal correction for hips
h_correction_3d = 0.6*(annot_3d[:,(3*lhip):(3*lhip+3)] - annot_3d[:,(3*rhip):(3*rhip+3)])/2;


#Left Hip
hip = 4;
correct_annot_3d[:,(3*hip):(3*hip+3)] = annot_3d[:,(3*hip):(3*hip+3)] + correction3d # + h_correction_3d;


#Right Hip
hip = 1;
correct_annot_3d[:,(3*hip):(3*hip+3)] = annot_3d[:,(3*hip):(3*hip+3)] + correction3d #- h_correction_3d;

gt = correct_annot_3d

print(gt)

for r in range(0, gt.shape[0]):
    rootx = gt[r, 0]
    rooty = gt[r, 1]
    rootz = gt[r, 2]
    for j in range(0, 17):
        gt[r, 3*j] = gt[r, 3*j] - rootx
        gt[r, 3*j+1] = gt[r, 3*j+1] - rooty
        gt[r, 3*j+2] = gt[r, 3*j+2] - rootz
print(gt)

#scale
gt = gt*1.83

print(gt)

for i, frm in enumerate(frame_arr):
        data['actors'][actor][frm][sys.argv[3]]['annotation_3D_h36m'] = gt[i, :]

with open('test_pickle.pkl', 'wb') as newfile:
    pickle.dump(data, newfile)








