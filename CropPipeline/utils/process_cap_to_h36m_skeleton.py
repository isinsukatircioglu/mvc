import numpy
import sys
import itertools
import pickle


# Open database file
scale_factor = 1.83
pickle_file_in = sys.argv[1]
pickle_file_out = sys.argv[2]
print("Opening pickle file : {}".format(pickle_file_in))
data = pickle.load(open(pickle_file_in, 'rb'))

# # fix naming issue
# print("Renaming data['camera_parameters'][cam]['intrinsic'] to data['camera_parameters'][cam]['intrisic']")
# for cam in data['camera_parameters'].keys():
#     data['camera_parameters'][cam]['intrinsic'] = data['camera_parameters'][cam]['intrisic']

#################
# 8)Change groundtruth pose order & hip coordinates
print("Fixing hip position and scale")
for actor in data['actors']:
    total_frame = len(data['actors'][actor])

    for i, frm in enumerate(data['actors'][actor].keys()):
        cam_list = data['actors'][actor][frm].keys()
        for cam in cam_list:
            annotation = data['actors'][actor][frm][cam]['annotations_3D_cap']
            # data['actors'][actor][frm][cam]['annotations_3D_cap'] = annotation # save original annotation
            annotation = list(itertools.chain(*annotation))
            annot_3d = numpy.array(annotation).copy()

            lhip = 4
            rhip = 1
            pelvis = 0

            # Vertical correction for hips
            v_correction_3d = annot_3d[(3*pelvis):(3*pelvis+3)] -(annot_3d[(3*lhip):(3*lhip+3)] + annot_3d[(3*rhip):(3*rhip+3)])/2;

            # Horizontal correction for hips
            h_correction_3d = 0.6*(annot_3d[(3*lhip):(3*lhip+3)] - annot_3d[(3*rhip):(3*rhip+3)])/2;

            # Left Hip
            hip = 4
            correct_annot_3d = annot_3d.copy()
            correct_annot_3d[(3*hip):(3*hip+3)] = annot_3d[(3*hip):(3*hip+3)] + v_correction_3d # + h_correction_3d;

            # Right Hip
            hip = 1
            correct_annot_3d[(3*hip):(3*hip+3)] = annot_3d[(3*hip):(3*hip+3)] + v_correction_3d #- h_correction_3d;

            # scale
            correct_annot_3d = correct_annot_3d * scale_factor
            data['actors'][actor][frm][cam]['annotations_3D'] = correct_annot_3d

# import IPython; IPython.embed()

print("Saving in pickle file : {}".format(pickle_file_out))
with open(pickle_file_out, 'wb') as newfile:
    pickle.dump(data, newfile, protocol=2)








