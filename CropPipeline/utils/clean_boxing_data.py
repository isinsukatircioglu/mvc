import os
import sys
import glob
import pickle


base_folder = '/Users/victor/workspace/data/BoxingDataV2/'

seq = 'Sequence' + sys.argv[1]
actor = sys.argv[2]
cams = ['cam_0', 'cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_6']
ref_frame = sys.argv[3]

data = pickle.load(open(os.path.join(base_folder, seq, 'data_pickle.pkl'), 'rb'))

for cam in cams:
    data_folder = os.path.join(base_folder, seq, actor, cam)
    all_images = glob.glob(data_folder+'*.jpg')
    for f in all_images:
        imgname = os.path.splitext(os.path.basename(f))[0]
        curr_frame = imgname.split('frame_')[1]
        if curr_frame >= ref_frame:
            os.remove(f)


mydict = {k: v for k, v in data['actors'][actor].items() if k < ref_frame}
data['actors'][actor] = mydict

with open(os.path.join(base_folder, seq, 'data_pickle.pkl'), 'wb') as newfile:
    pickle.dump(data, newfile, protocol=2)
