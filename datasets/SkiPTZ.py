import os
import csv
from PIL import Image
import numpy as np
import numpy.linalg as la
import torch
import torch.utils.data as data
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import warp
from scipy import misc
import imageio
from random import shuffle
import IPython
import scipy
import pdb
import math
import sys
import os.path

sys.path.insert(0, '../')
import datasets.utils as utils_data
from PlottingUtil import util as utils_plt

import CropPipeline.utils as utils_crop
from CropPipeline.BoundingBoxes.square_bounding_box import BoundingBox
from CropPipeline.CropCompensation import projection
from datasets import transforms as transforms_aug
from tqdm import tqdm

class Sequence_DLT:
    def __init__(self, base_folder, input_types, label_types,
                 input_img_width, map_width, input_transform,
                 augmentation, randomize,
                 active_cameras,
                 trial, subject, start_frame, end_frame, offset_3d_params,
                 camera_indices_db,
                 joint_transformation, root_index, bbox_margin, check_score):
        print(
        "Loading trial {} from frame {} to frame {}, from folder {}".format(trial, start_frame, end_frame, base_folder))
        self.image_shape = None
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.trial = trial
        self.subject = subject
        self.offset_3d_params = offset_3d_params

        self.input_img_width = input_img_width
        self.map_width = map_width
        self.input_transform = input_transform

        self.augmentation = augmentation
        self.randomize = randomize
        self.label_types = label_types
        self.input_types = input_types
        #Offline detection
        self.detection_dict = {}
        self.detection_offline_available = False
        #self.image_template = base_folder + "Videos/trial_{trial}_cam{cam}/frame_{frame:06d}.jpg"
        self.image_template = base_folder + "Videos_Small/trial_{trial}_cam{cam}/frame_{frame:06d}.jpg"
        self.image_template_original = base_folder + "Videos/trial_{trial}_cam{cam}/frame_{frame:06d}.jpg"
        self.validation_masks = base_folder + "Segmentation_masks/trial_{trial}_cam{cam}/frame_{frame:06d}.jpg"
        self.check_score = check_score

        self.camera_DLT_template = base_folder + "Data/DLT_Parameters/PP_Trial_{trial}_CAM{cam}.DLT"
        self.pose_annotation_template = base_folder + "Data/3D_Position_Data/Trial_{trial}_AllMarker_Raumkoor.mat.csv"
        self.camera_indices_db = camera_indices_db
        self.margin = bbox_margin  # 0.4 is default
        self.joint_transformation = joint_transformation
        self.root_index = root_index

        # load skier and reference positions
        trial = self.trial
        key_points_file = self.pose_annotation_template.format(trial=trial)
        with open(key_points_file, 'r') as f:
            content = f.readlines()
            content = [x.strip().split(",") for x in content]

        header = content[0]
        points_str = content[1:]
        points_flt = [[float(s) for s in s_row] for s_row in points_str]
        position_mat = np.array(points_flt)
        position_tensor = position_mat.reshape([position_mat.shape[0], -1, 3]) / 1000  # from mm to meters
        self.position_ski_tensor = position_tensor[:, :-2, :]  # remove center of mass
        self.COM_tensor = position_tensor[:, -2:, :]

        # undo rotation of coordinate system to ground plane
        psi = -0.193200777
        for pose in self.position_ski_tensor:
            Az = np.linalg.inv(np.matrix(
                [[math.cos(psi), math.sin(psi), 0],
                 [-math.sin(psi), math.cos(psi), 0],
                 [0, 0, 1], ]))
            pose[:, :] = pose @ Az.T

        # load DLT parameters
        camera_DLTs = []
        camera_extrinsics = []
        camera_intrinsics = []
        camera_extrinsics_inverse = []
        camera_intrinsics_inverse = []

        for ci in self.camera_indices_db:
            trial = self.trial
            camera_DLT_filename = self.camera_DLT_template.format(trial=trial, cam=ci)
            with open(camera_DLT_filename, 'r') as f:
                content = f.readlines()
                content = [x.strip() for x in content]

                DLT_single_cam = []
                extrinsics_single_cam = []
                intrinsics_single_cam = []
                extrinsics_inverse_single_cam = []
                intrinsics_inverse_single_cam = []
                for row in content:
                    points_float = []
                    row_words = list(filter(None, row.split(" ")))  # filtering out empty elements
                    for element in row_words:
                        try:
                            number = float(element)
                            points_float.append(number)
                        except:
                            # print('Warning, skipping element: {}'.format(element))
                            points_float.append(0)
                    # print(row, points_float)
                    if len(points_float) < 12:
                        points_mat = np.zeros((12))
                    else:
                        points_mat = np.array(points_float).reshape((-1))

                    error_value_UNUSED = points_mat[-1]
                    DLT_mat = points_mat.reshape([3, 4]).copy()  # n
                    DLT_mat[2, 3] = 1

                    # reconstruct intrinsics and extrinsics from Projection matrix
                    # P = [M | -MC] = K[R|-RC]
                    # => C = - M^-1 MC
                    # and R and K can be decomposed by rq decomposition KR = rq(M)
                    DLT_single_cam.append(DLT_mat)
                    M = DLT_mat[:3, :3]
                    if np.sum(M) == 0:
                        extrinsics_single_cam.append(np.eye(3))
                        intrinsics_single_cam.append(np.eye(3))
                        extrinsics_inverse_single_cam.append(np.eye(3))
                        intrinsics_inverse_single_cam.append(np.eye(3))
                    else:
                        MC = DLT_mat[:, -1]
                        C = - np.linalg.inv(M) @ MC
                        r, q = scipy.linalg.rq(M)
                        T = np.diag(np.sign(np.diag(r)))  # appears to be the identity in our case
                        K = r @ T
                        K = K / K[2, 2]
                        R = T @ q  # (T is its own inverse)

                        assert np.linalg.det(R) > 0
                        R_mRC = np.concatenate([R, - R @ C.reshape(3, 1)], 1)

                        extrinsics_single_cam.append(R_mRC)
                        intrinsics_single_cam.append(K)
                        extrinsics_inverse_single_cam.append(np.linalg.inv(np.concatenate([R_mRC.reshape(3,4), np.array([0,0,0,1]).reshape(1,4)], 0)))
                        intrinsics_inverse_single_cam.append(np.linalg.inv(K))
                camera_DLTs.append(DLT_single_cam)
                camera_extrinsics.append(extrinsics_single_cam)
                camera_intrinsics.append(intrinsics_single_cam)
                camera_extrinsics_inverse.append(extrinsics_inverse_single_cam)
                camera_intrinsics_inverse.append(intrinsics_inverse_single_cam)

        self.camera_extrinsics = camera_extrinsics
        self.camera_intrinsics = camera_intrinsics
        self.camera_extrinsics_inverse = camera_extrinsics_inverse
        self.camera_intrinsics_inverse = camera_intrinsics_inverse
        self.camera_DLTs = camera_DLTs
        self.start_frame = start_frame

        last_data_point = min(end_frame, len(self.camera_extrinsics[0]) + self.offset_3d_params,
                              len(self.position_ski_tensor) + self.offset_3d_params)
        self.last_data_pointXXX = last_data_point
        first_data_point = start_frame

        self.camera_intrinsics = [
            mat_list[first_data_point + self.offset_3d_params:last_data_point + self.offset_3d_params] for mat_list in
            self.camera_intrinsics]
        self.camera_extrinsics = [
            mat_list[first_data_point + self.offset_3d_params:last_data_point + self.offset_3d_params] for mat_list in
            self.camera_extrinsics]
        self.camera_intrinsics_inverse = [
            mat_list[first_data_point + self.offset_3d_params:last_data_point + self.offset_3d_params] for mat_list in
            self.camera_intrinsics_inverse]
        self.camera_extrinsics_inverse = [
            mat_list[first_data_point + self.offset_3d_params:last_data_point + self.offset_3d_params] for mat_list in
            self.camera_extrinsics_inverse]
        self.camera_DLTs = [mat_list[first_data_point + self.offset_3d_params:last_data_point + self.offset_3d_params]
                            for mat_list in self.camera_DLTs]
        self.position_ski_tensor = self.position_ski_tensor[
                                   first_data_point + self.offset_3d_params:last_data_point + self.offset_3d_params]

        self.pose_3d_std = np.array(
            [[1., 1., 1.],
             [0.05826372, 0.04978976, 0.05134333],
             [0.27340364, 0.09691441, 0.29028127],
             [0.35750586, 0.16221502, 0.36814487],
             [0.05826372, 0.04978976, 0.05134333],
             [0.24525689, 0.10482119, 0.32860646],
             [0.37285304, 0.16775107, 0.39257801],
             [0.16011892, 0.03951239, 0.12028801],
             [0.32023785, 0.07902478, 0.24057603],
             [0.40029722, 0.09878097, 0.30071992],
             [0.40029722, 0.09878097, 0.30071992],
             [0.30249023, 0.09055768, 0.25097767],
             [0.34041852, 0.19714764, 0.36417514],
             [0.43341705, 0.274544, 0.50092417],
             [0.32783848, 0.108885, 0.22390638],
             [0.38636398, 0.20898452, 0.30639204],
             [0.48139435, 0.27329221, 0.43275994]])

        self.pose_3d_mean = np.array(
            [[0., 0., 0.],
             [-0.01267349, 0.00287692, 0.00595046],
             [-0.06182071, 0.20187752, - 0.05194594],
             [-0.06079669, 0.41750649, - 0.05093238],
             [0.01267349, - 0.00287692, - 0.00595046],
             [0.00381442, 0.18486036, - 0.09098279],
             [0.01250531, 0.40546718, - 0.09612962],
             [-0.01068636, - 0.16846988, - 0.01268945],
             [-0.02137272, - 0.33693975, - 0.0253789],
             [-0.02671592, - 0.42117494, - 0.03172364],
             [-0.02671592, - 0.42117494, - 0.03172364],
             [0.00504989, - 0.33490506, - 0.028424],
             [0.0321653, - 0.15484168, - 0.07342236],
             [0.0194478, - 0.03149849, - 0.11704593],
             [-0.0379229, - 0.32253978, - 0.00925214],
             [-0.07470388, - 0.11401245, - 0.01218812],
             [-0.09565282, 0.00567211, - 0.04976835]])

    def getNumberOfFrames(self):
        num_DLT_frames = min(min(map(len, self.camera_extrinsics)),
                             min(map(len, self.camera_intrinsics)),
                             min(map(len, self.camera_DLTs)))
        num_pose_frames = len(self.position_ski_tensor)
        return min(num_DLT_frames, num_pose_frames)

    def retrieveCameraRepresentation(self, cam_id, frame_id):
        K = self.camera_intrinsics[cam_id][frame_id]
        K_inv = self.camera_intrinsics_inverse[cam_id][frame_id]
        RT = self.camera_extrinsics[cam_id][frame_id]
        RT_inv = self.camera_extrinsics_inverse[cam_id][frame_id]
        R_world_2_cam = RT[:, :3]
        R_cam_2_world = np.linalg.inv(R_world_2_cam)
        cam_center = - (R_cam_2_world @ RT[:, -1])

        return K, R_world_2_cam, R_cam_2_world, cam_center, RT, K_inv, RT_inv

    def projectPoseToCam(self, frame_id, K, cam_center, R_world_2_cam):
        pose_3d_w_orig = self.position_ski_tensor[frame_id, :, :]
        pose_3d_w = self.joint_transformation @ pose_3d_w_orig
        pose_3d_c = ((pose_3d_w - cam_center) @ R_world_2_cam.T)
        if type(self.root_index) == list:  # average of multiple joints?
            pose_3d_center_c = np.mean(pose_3d_c[self.root_index, :], axis=0)
        else:
            pose_3d_center_c = pose_3d_c[self.root_index, :]

        pose_3d_centered_c = pose_3d_c - pose_3d_center_c

        pose_2d_cam = pose_3d_c / pose_3d_c[:, 2, np.newaxis]
        pose_2d_px = pose_2d_cam @ K.T

        return pose_3d_w, pose_3d_c, pose_3d_centered_c, pose_2d_px

    def isPersonInView(self, frame_id, cam_id):
        cam_index_db = self.camera_indices_db[cam_id]

        if not self.image_shape:
            file_name = self.image_template_original.format(cam=cam_index_db, frame=self.start_frame + frame_id,
                                                   trial=self.trial)
            self.image_shape = imageio.imread(file_name).shape


        K, R_world_2_cam, R_cam_2_world, cam_center, cam_ext, cam_int_inv, cam_ext_inv = self.retrieveCameraRepresentation(cam_id, frame_id)

        pose_3d_w, pose_3d_c, pose_3d_centered_c, pose_2d_px = self.projectPoseToCam(frame_id, K, cam_center,
                                                                                     R_world_2_cam)

        any_negative_entry = np.any(pose_2d_px < 0)

        # compute bounding box
        bbox_generator = BoundingBox(0)  # 0 instead of self.margin to also use partially visible frames
        bbox = bbox_generator._compute_bounding_box(pose_2d_px[:15, :2].tolist())
        too_small_bbox = bbox[3] < 56 or bbox[2] < 56
        exceeding_bounds = bbox[0] + bbox[2] > self.image_shape[1] or bbox[1] + bbox[3] > self.image_shape[0]

        return not (any_negative_entry or too_small_bbox or exceeding_bounds)

    def getitem_single(self, frame_id, cam_id, index):
        # print("Get single image, cam_id={}, frame_id={}".format(cam_id,frame_id))
        cam_index_db = self.camera_indices_db[cam_id]
        # Note, frame id starts from zero, the image name not
        image_frame_index = self.start_frame + frame_id
        file_name = self.image_template.format(cam=cam_index_db, frame=image_frame_index, trial=self.trial)

        #read val masks:
        if self.check_score:
            mask_name = self.validation_masks.format(cam=cam_index_db, frame=image_frame_index, trial=self.trial)
            msk_read = (cv2.imread(mask_name, 0) > 200).astype('float32')
            msk_read = cv2.resize(msk_read, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        # load and crop image
        img_read = imageio.imread(file_name)
        img_raw = img_read.copy()
        orig_image_size = (720,1280)
        K, R_world_2_cam, R_cam_2_world, cam_center, cam_ext, cam_int_inv, cam_ext_inv = self.retrieveCameraRepresentation(cam_id, frame_id)
        pose_3d_w, pose_3d_c, pose_3d_centered_c, pose_2d_px = self.projectPoseToCam(frame_id, K, cam_center,
                                                                                     R_world_2_cam)
        K_resized = K.copy()
        K_resized[0, 0] = K[0, 0] / 2
        K_resized[1, 1] = K[1, 1] / 2
        K_resized[0, 2] = K[0, 2] / 2
        K_resized[1, 2] = K[1, 2] / 2
        # 2D locations
        #bbox_generator_without = BoundingBox(self.margin)
        #bbox_generator_ski = BoundingBox(0)
        #bbox_without_ski = bbox_generator_without._compute_bounding_box(pose_2d_px[:15,
        #                                                                :2].tolist())  # note, :15 to explude ski from bb computation, person gets too small otherwise. Was experimented by Mirela
        #bbox_with_ski = bbox_generator_ski._compute_bounding_box(pose_2d_px.tolist())
        #if self.margin == 0:
        #    bbox = bbox_with_ski
        #else:
        #    bbox = (np.array(bbox_with_ski) + np.array(bbox_without_ski)) // 2  # // to round to full iinteger number
        #bbox_gen = BoundingBox(self.margin)
        bbox_gen = BoundingBox(0.6)
        bbox = bbox_gen._compute_bounding_box(pose_2d_px.tolist())# HACK

        img = utils_crop.crop(img_raw, bbox)

        # transform 2D coordinates to crop
        K_img_to_crop_o = transforms_aug.getC_img_to_crop(bbox)
        pose_2d_crop = pose_2d_px @ K_img_to_crop_o.T

        pose_2d_crop = pose_2d_crop[:, :2]
        combined_dict = {}
        for key in self.label_types + self.input_types:
            if key == 'img_crop':
                combined_dict[key] = np.array(img, dtype='float32')
            elif key in ['img', 'bg']:
                combined_dict[key] = img_raw
            elif key == 'bg_crop':
                combined_dict[key] = np.array(img * 0, dtype='float32')  # TODO HACK
            elif key == '3D':
                combined_dict[key] = np.array(pose_3d_centered_c, dtype='float32').reshape(-1)
            elif key == '3D_global':
                combined_dict[key] = np.array(pose_3d_c, dtype='float32').reshape(-1)
            elif key == '3D_world':
                combined_dict[key] = np.array(pose_3d_w, dtype='float32').reshape(-1)
            elif key == '2D':
                combined_dict[key] = np.array(pose_2d_crop, dtype='float32').reshape(-1)
            # elif key == '2D_orig':
            #    combined_dict[key] = np.array(pose_2d_orig_crop, dtype='float32').reshape(-1)
            elif key == '2D_fullSize':
                combined_dict[key] = np.array(pose_2d_px[:, :2] / np.flipud(orig_image_size),
                                              dtype='float32').reshape(-1)
            elif key == 'bounding_box':
                combined_dict[key] = np.array(bbox, dtype='float32')
            elif key == 'bounding_box_cam':
                combined_dict[key] = transforms_aug.getBoundingBoxInCamSpace(bbox, K)
            elif key == 'bounding_box_yolo':
                bbox_01 = np.array(bbox.copy(), dtype='float32')
                bbox_01[:2] = bbox_01[:2] / np.flipud(orig_image_size)
                bbox_01[2:] = bbox_01[2:] / np.flipud(orig_image_size)
                combined_dict[key] = np.array(bbox_01, dtype='float32')
            elif key == 'extrinsic_rot_inv':
                R_cam_2_world = np.linalg.inv(R_world_2_cam)
                combined_dict[key] = np.array(R_cam_2_world, dtype='float32')
            elif key == 'extrinsic_rot':
                combined_dict[key] = np.array(R_world_2_cam, dtype='float32')
            elif key == 'extrinsic_pos':
                combined_dict[key] = np.array(cam_center, dtype='float32')
            elif key == 'camera_extrinsics':
                combined_dict[key] = np.array(cam_ext, dtype='float32')
            elif key == 'inverse_camera_extrinsics':
                combined_dict[key] = np.array(cam_ext_inv, dtype='float32')
            elif key == 'intrinsic_crop':
                K_crop_orig = K_img_to_crop_o @ K
                combined_dict[key] = np.array(K_crop_orig, dtype='float32')
            elif key == 'intrinsic':
                #combined_dict[key] = np.array(K, dtype='float32')
                combined_dict[key] = np.array(K_resized, dtype='float32')
            elif key == 'inverse_intrinsic':
                combined_dict[key] = np.array(np.linalg.inv(K_resized), dtype='float32')
            elif key == 'frame_info':
                combined_dict[key] = np.array([cam_id, frame_id, self.trial, self.subject, index], dtype='float32')
            elif key == 'file_name_info':
                combined_dict[key] = np.array([cam_index_db, image_frame_index, self.trial],dtype='float32')
            elif key == 'trial':
                combined_dict[key] = np.array([self.trial], dtype='float32')
            elif key == 'image_frame':
                combined_dict[key] = np.array([image_frame_index], dtype='float32')
            elif key == 'camera':
                combined_dict[key] = np.array([cam_index_db], dtype='float32')
            elif key == 'label_labels':
                combined_dict[key] = np.array(self.label_types)
            elif key == 'input_labels':
                combined_dict[key] = np.array(self.input_types)
            elif key == 'subject':
                combined_dict[key] = np.array(self.subject)
            elif key in ['pose_mean']:
                combined_dict[key] = np.array(self.pose_3d_mean, dtype='float32').reshape([-1])
            elif key in ['pose_std']:
                combined_dict[key] = np.array(self.pose_3d_std, dtype='float32').reshape([-1])
            else:
                raise ValueError('WARNING, label {} not found, setting to -1'.format(key))

        # augmentation and compensation
        if self.augmentation is not None:
            transformation_instances = self.augmentation.parametrize_and_randomize(label_dict=combined_dict,
                                                                                   batch_index=0)  # TODO
            self.augmentation.apply(transformation_instances, combined_dict,
                                    [360, 640]) 

            combined_dict['trans_2d'] = np.array(transformation_instances['trans2D'], dtype='float32')
            combined_dict['trans_2d_inv'] = np.array(np.linalg.inv(transformation_instances['trans2D']),
                                                     dtype='float32')

        if self.input_transform is not None:
            for key in ['img_crop', 'bg_crop', 'img', 'bg']:
                if key in combined_dict:
                    combined_dict[key] = self.input_transform(combined_dict[key])

        input_dict = {}
        label_dict = {}
        for key in self.input_types:
            input_dict[key] = combined_dict[key]
        for key in self.label_types:
            label_dict[key] = combined_dict[key]

        if self.check_score:
            label_dict['annotated'] = msk_read
        return input_dict, label_dict


class SkiPanTiltDataset_DLT(data.Dataset):
    def __init__(self, base_folder, input_types, label_types,
                 input_img_width, map_width,
                 subjects=None,
                 active_cameras=False,
                 study_id=1,
                 input_transform=None,
                 file_name_template=None, frame_indices=None,
                 augmentation=None,
                 use_multi_scale=False,
                 bbox_from_annotation=True,
                 bbox_margin=0.4,
                 useCamBatches=0,
                 every_nth_frame=1, randomize=False, useSequentialFrames=False,
                 joint_transformation=utils_plt.ski_spoerri_to_h36m,
                 root_index=utils_plt.root_index_h36m, check_score=False):

        self.useCamBatches = useCamBatches
        self.randomize = randomize
        self.check_score = check_score
        self.camera_indices_db = [1, 2, 3, 4, 5, 6]
        trials_study1_A = [103, 124, 202, 221, 302, 412]
        trials_study3_A = [111, 114, 208, 217, 304, 406]
        trials_study1_B = [115, 110, 214, 207, 309, 405]  # [110,115,207,214,309,405]
        trials_study3_B = [119, 102, 224, 205, 315, 401]  # [102,119,205,224,315,401]

        trials_all = trials_study1_A + trials_study3_A + trials_study1_B + trials_study3_B

        # note, this list requires trials_all to be in the right order (trials_study1_A + trials_study3_A + trials_study1_B + trials_study3_B)
        subject_id = dict(zip(trials_all,
                              [0, 1, 2, 3, 4, 5,
                               0, 1, 2, 3, 4, 5,
                               0, 1, 2, 3, 4, 5,
                               0, 1, 2, 3, 4, 5]))

        ski_type = dict(zip(trials_all,
                            [1, 1, 1, 1, 1, 1,
                             2, 2, 2, 2, 2, 2,
                             3, 3, 3, 3, 3, 3,
                             4, 4, 4, 4, 4, 4]))

        # offset due to index of 1 in matlab??
        offset_3d_params = dict(zip(trials_all,
                                    [-1, -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1, ]))

        first_datapoint = dict(zip(trials_all, np.repeat(36, 6 * 4)))
        gate_2 = dict(zip(trials_all, np.repeat(51, 6 * 4)))
        gate_3 = dict(zip(trials_all,
                          [137, 138, 135, 135, 132, 132,
                           138, 139, 134, 133, 130, 132,
                           138, 140, 134, 134, 129, 136,
                           135, 141, 136, 133, 133, 136]
                          ))
        gate_4 = dict(zip(trials_all,
                          [213, 218, 211, 211, 203, 206,
                           214, 218, 212, 210, 203, 209,
                           218, 218, 211, 208, 202, 214,
                           216, 222, 216, 212, 206, 212]))
        last_datapoint = dict(zip(trials_all,
                                  [218, 223, 216, 216, 208, 211,
                                   219, 223, 217, 215, 208, 214,
                                   223, 223, 216, 213, 207, 219,
                                   221, 227, 221, 217, 211, 217]))

        crossingpoint_a = dict(zip(trials_all,
                                   [80, 85, 79, 89, 77, 83,
                                    81, 84, 88, 86, 89, 83,
                                    84, 82, 87, 87, 83, 89,
                                    84, 88, 84, 93, 88, 89]))
        crossingpoint_b = dict(zip(trials_all,
                                   [166, 165, 160, 164, 157, 160,
                                    165, 165, 166, 164, 158, 161,
                                    165, 165, 163, 164, 159, 169,
                                    165, 174, 169, 171, 162, 167]))

        # apply respective temporal offsets
        # offset of 3D coordinates, has already been applied on mat files when transforming to h5. Needs to be applied on frame annotations too
        temporal_offset = dict(zip(trials_all, [
            32, 31, 31, 33, 27, 31,
            33, 35, 30, 30, 24, 30,
            36, 37, 35, 29, 29, 31,
            35, 34, 32, 32, 31, 34]))
        for key in temporal_offset.keys():
            off_by_one = 1  # due to matlab notation statign at 1 not 0, I guess
            first_datapoint[key] += temporal_offset[key] + off_by_one
            gate_2[key] += temporal_offset[key] + off_by_one
            gate_3[key] += temporal_offset[key] + off_by_one
            gate_4[key] += temporal_offset[key] + off_by_one
            last_datapoint[key] += temporal_offset[key] + off_by_one
            crossingpoint_a[key] += temporal_offset[key] + off_by_one
            crossingpoint_b[key] += temporal_offset[key] + off_by_one

        # the 'first_datapoints' originally marked are not reliable, start at the gate. Same for 'last_datapoint'
        first_datapoint = gate_2
        last_datapoint = {k: min(gate_4[k], last_datapoint[k]) for k in last_datapoint}

        print("Offset normalized frame numbers: (equals to image frame number = dataset index + 51 (start frame)")
        print("first_datapoint:", first_datapoint)
        print("gate_2:", gate_2)
        print("gate_3:", gate_3)
        print("gate_4:", gate_4)
        print("last_datapoint:", last_datapoint)
        print("crossingpoint_a:", crossingpoint_a)
        print("crossingpoint_b:", crossingpoint_b)

        # select a subset of trials to train on
        self.trials = trials_all  # trials_study1_A #+ trials_study1_B
        # self.trials = [trials_all[1]] #trials_study1_A #+ trials_study1_B
        if subjects is not None:
            self.trials = [trial for trial in self.trials if subject_id[trial] in subjects]
        if study_id in [1, 2]:
            self.trials = [trial for trial in self.trials if trial in (trials_study1_A + trials_study1_B)]
        elif study_id in [3, 4]:
            self.trials = [trial for trial in self.trials if
                           trial in (trials_study1_A + trials_study1_B + trials_study3_A + trials_study3_B)]
        else:  # specified a specific trial
            self.trials = [int(study_id)]
            first_datapoint = crossingpoint_a
            last_datapoint = crossingpoint_b

        if study_id in [4]:
            first_datapoint = crossingpoint_a
            last_datapoint = crossingpoint_b

        trial_data = {}
        randomize = False
        for trial in self.trials:
            trial_data[trial] = Sequence_DLT(base_folder,
                                             input_types, label_types,
                                             input_img_width, map_width, input_transform,
                                             augmentation, randomize,
                                             active_cameras,
                                             trial, subject_id[trial], first_datapoint[trial], last_datapoint[trial],
                                             offset_3d_params[trial],
                                             self.camera_indices_db,
                                             joint_transformation,
                                             root_index, bbox_margin, self.check_score)

        # createSequentialIndexLists
        self.index_to_seq_frame_cam_sub = []
        self.index_to_seq_frame_sub = []
        for trial in self.trials:
            if subjects == [4] and self.check_score:
                if trial == 302:
                    frames = [21, 71, 111]
                elif trial == 309:
                    frames = [19, 69, 109]
            elif subjects == [5] and self.check_score:
                if trial == 405:
                    frames = [  3,   7,  13,  17,  22,  27,  32,  37,  42,  47,  52,  59,  66, 72,  77,  82,  87,  92,  97, 102, 107, 111, 117, 122, 129]
                elif trial == 412:
                    frames = [1,   7,  14,  22,  27,  31,  37,  42,  48,  54,  59,  66,  71, 75,  79,  86,  90,  95,  98, 102, 105, 108, 117, 122, 127]
            else:
                frames = range(trial_data[trial].getNumberOfFrames())

            for fi in frames:
                # Change here ! if fi in [100, 150, 190]
                inView = []
                for ci in range(len(self.camera_indices_db)):
                    if trial_data[trial].isPersonInView(fi, ci) and ((not active_cameras) or ci in active_cameras):
                        inView.append(ci)
                if len(inView) >= useCamBatches:
                    self.index_to_seq_frame_sub.append([trial, fi, inView])
                    for ci in inView:
                        self.index_to_seq_frame_cam_sub.append([trial, fi, ci])


        if useSequentialFrames:
            s = self.index_to_seq_frame_cam_sub
            s = sorted(s, key=lambda t: t[1])  # frames
            s = sorted(s, key=lambda t: t[2])  # cams
            s = sorted(s, key=lambda t: t[0])  # seq
            self.index_to_seq_frame_cam_sub = s

        self.trial_data = trial_data

        print("SkiPanTiltDataset_DLT: Done initializing, listed {} frames".format(self.__len__()))

    def __len__(self):
        if self.useCamBatches > 0:
            return len(self.index_to_seq_frame_sub)
        else:
            return len(
                self.index_to_seq_frame_cam_sub)  # num_cams * max(0, min(num_DLT_frames, num_pose_frames)-self.start_frame)

    def __getitem__(self, index):
        if self.useCamBatches > 0:
            # get input (image) and annotation for each camera
            trial, frame_id, cam_list = self.index_to_seq_frame_sub[index]
            cam_list = cam_list[:]  # copy
            #THIS CHANGE IS DONE FOR THE MULTICAM APPROACH:
            wrong_combination_of_cams = True
            if self.randomize:
                if self.useCamBatches == 2:
                    while wrong_combination_of_cams:
                        shuffle(cam_list)
                        # 1 corresponds to cam 2, 4 corresponds to cam 5 and 5 corresponds to cam6
                        if ((1 in cam_list[:self.useCamBatches]) and (4 in cam_list[:self.useCamBatches])) or ((1 in cam_list[:self.useCamBatches]) and (5 in cam_list[:self.useCamBatches])) or ((2 in cam_list[:self.useCamBatches]) and (5 in cam_list[:self.useCamBatches])) or ((0 in cam_list[:self.useCamBatches]) and (4 in cam_list[:self.useCamBatches])):
                            continue
                        else:
                            wrong_combination_of_cams = False
                else:
                    shuffle(cam_list)
            assert self.useCamBatches <= len(cam_list), pdb.set_trace()
            cam_keys_shuffled = cam_list[:self.useCamBatches]

            single_examples = [self.trial_data[trial].getitem_single(frame_id, cam_i, index) for cam_i in
                               cam_keys_shuffled]
            collated_examples = utils_data.default_collate_with_string(
                single_examples)  # accumulate list of single frame results
            return collated_examples
        else:
            trial, frame_id, cam_id = self.index_to_seq_frame_cam_sub[index]
            return self.trial_data[trial].getitem_single(frame_id, cam_id, index)

