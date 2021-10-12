import numpy as np
from numpy import linalg as la
import scipy 

import torch
import torch.utils.data
import IPython
import pdb
import random
import skimage
from CropPipeline.CropCompensation import projection
from PlottingUtil import util as utils_plt

import math
from math import sin,cos
from numpy.ma.extras import mr_

def getBoundingBoxInCamSpace(bbox,K):
    K_inv = np.linalg.inv(K)
    origin = np.dot(K_inv, np.array([[bbox[0],         bbox[1],        1]]).T)
    corner = np.dot(K_inv, np.array([[bbox[0]+bbox[2], bbox[1]+bbox[3],1]]).T)
    bounding_box_cam = np.reshape(np.array([origin[0], origin[1], corner[0]-origin[0], corner[1]-origin[1]]), (-1)).astype('float32')
    return np.array(bounding_box_cam, dtype='float32')

# get the cropping deformation
def getC_img_to_crop(bbox, crop_width=1):
    s = crop_width / bbox[2] # change of focal length (zoom)
    C_img_to_crop = np.array([[s, 0, -bbox[0]*s],
                              [0, s, -bbox[1]*s],
                              [0, 0,  1]],dtype='float32')
    
    #np.array([[1/bbox[2], 0, -bbox[0]/bbox[2]],
    #                            [0, 1/bbox[3], -bbox[1]/bbox[3]],
    #                            [0, 0,  1]])
    return C_img_to_crop

def getK_crop(bbox, K):
    C_img_to_crop = getC_img_to_crop(bbox)
    K_crop   = C_img_to_crop @ K
    return K_crop

def normalize_mean_std_np(pose_tensor, pose_mean, pose_std):
    return (pose_tensor-pose_mean)/pose_std

def denormalize_mean_std_np(pose_tensor, pose_mean, pose_std):
    return pose_tensor*pose_std + pose_mean

def normalize_mean_std_tensor(pose_tensor, label_dict):
    pose_mean = label_dict["pose_mean"]
    pose_std  = label_dict["pose_std"]
    return (pose_tensor-pose_mean)/pose_std

def denormalize_mean_std_tensor(pose_tensor, label_dict):
    pose_mean = label_dict["pose_mean"]
    pose_std  = label_dict["pose_std"]
    return pose_tensor*pose_std + pose_mean

class normalize_cropRect(torch.nn.Module):
    """
    Normalize the deformation imposed due to cropping
    """
    def __init__(self, child_criterion, key='3D'):
        super(normalize_cropRect, self).__init__()
        self.child_criterion = child_criterion
        self.key = key

    def forward(self, preds, labels):
        pred_tensor = preds[self.key] # pred_keys.index(self.key)]
        batch_size = pred_tensor.size()[0]
        pred_tensor = pred_tensor.view(batch_size,-1,3)
        D_aug = labels["aug_3D_inv"]
        #R_cam_to_world = label_cam_to_world_raw.view( (batch_size,3,  3) ).float() # batches x R_rows x R_cols
        pred_tensor_norm = torch.bmm(pred_tensor, D_aug.transpose(1,2)).view(batch_size,-1) # matrix multiplication, separate for each batch (first dim)

        return self.child_criterion.forward(pred_tensor_norm, labels[self.key])


class ApplyLinearTransformation:
    '''Combined application of a list of linear transformations on input (image, heatmap, depthmap),
        2D and 3D labels, as well as camera parameters (intrinsics and extrinsics)'''

    def __init__(self):
        self.transformations = []

    def add(self, trans):
        self.transformations.append(trans)

    def parametrize_and_randomize(self, label_dict, batch_index):
        transformation_instances = []
        for trans in self.transformations:
            transformation_instances.append(trans.parametrize_and_randomize(label_dict, batch_index))
            
        # image operations (2D)
        trans_2d_comb = np.eye(3, 3)
        for trans in transformation_instances:
            trans_2d = trans.get('trans2D', np.eye(3, 3))
            trans_2d_comb = np.dot(trans_2d, trans_2d_comb)

        # extrinsic (3D)
        trans_3d_comb = np.eye(4, 4)
        for trans in transformation_instances:
            trans_3d = trans.get('trans3D', np.eye(4, 4))
            trans_3d_comb = np.dot(trans_3d, trans_3d_comb)

        # intrinsics (focal length, crop location)
        trans_intr_comb = np.eye(3, 3)
        for trans in transformation_instances:
            trans_intr = trans.get('transIntr', np.eye(3, 3))
            trans_intr_comb = np.dot(trans_intr, trans_intr_comb)

        # row operations (e.g. flipping labels for mirroring)
        label_transformation_comb = None
        for trans in transformation_instances:
            label_transformation = trans.get('transLabel', None)
            if label_transformation_comb is None:
                label_transformation_comb = label_transformation
            elif label_transformation is not None:
                label_transformation_comb = np.dot(label_transformation_comb, label_transformation)
        return {'trans2D': trans_2d_comb, 'trans3D': trans_3d_comb, 'transIntr':trans_intr_comb, 'transLabel': label_transformation_comb}

    # apply a chain of linear transformation on images and heatmaps
    @staticmethod
    def apply2DImage(trans_2d_comb, img, output_shape):
        ##################################################
        # operations are in normalized pixel coordinates, transform to and back again
        # pixel to normalized coordinates (0..1)
        pixel2normalized = np.diagflat(1/np.array([img.shape[0], img.shape[1], 1]))
        trans_2d_comb = np.dot(trans_2d_comb, pixel2normalized)
        # normalized to pixel (is usually different to the input dimension)
        normalized2pixelout = np.diagflat(np.array([output_shape[0], output_shape[1], 1]))
        trans_2d_comb = np.dot(normalized2pixelout, trans_2d_comb)

        # other relations should work as well, but are not tested
        #assert output_shape[0] == output_shape[1]
        #assert img.shape[0] == img.shape[1]
        
        if 0:
            IPython.embed()
            import matplotlib.pyplot as plt
            plt.switch_backend('Qt5Agg')
            fig = plt.figure(0)
            
            img_float = img.astype('float32') / 255
            # input image
            ax   = fig.add_subplot(131)
            ax.imshow(img_float)


            #img_trans = skimage.transform.warp(img_float, np.linalg.inv(trans_2d_comb),  order=3, mode='constant', cval=0) # TODO: output_shape=(output_shape[0],output_shape[1])
            img_trans = skimage.transform.warp(img_float, np.linalg.inv(trans_2d_comb),  order=3, mode='symmetric') # TODO: output_shape=(output_shape[0],output_shape[1])
            ax   = fig.add_subplot(132)
            ax.imshow(img_trans)

            ax   = fig.add_subplot(133)
            ax.imshow(img_trans)

            plt.show()

        # mode='constant' sets a constant value for points outside the original image boundary
        # cval, the constant value to set
        # order: the order of interpolation
        # 0: nn
        # 1: Bi-linear (default)
        # 2: Bi-quadratic
        # 3: Bi-cubic
        # 4: Bi-quartic
        # 5: Bi-quintic

        # import IPython
        # IPython.embed()
        
        affine_inv = np.linalg.inv(trans_2d_comb)
        #def affine_norm(v):
        #    return v[:-1]/v[-1]
        
#        output_shape[0] = output_shape[1] = 224
#        warp_field = np.zeros([2,output_shape[0],output_shape[1]])
        
        homography = skimage.transform.ProjectiveTransform(matrix=affine_inv)
        
        #def comput_warp_field(warp_field):
        #    for r in range(output_shape[0]):
        #         for c in range(output_shape[1]):
        #             field_float = affine_norm(np.array(affine_inv) @ np.array([r,c,1]))
        #             warp_field[:,r,c] = np.array(field_float,dtype=int)
                 
        #img_trans = skimage.transform.warp(img, homography, preserve_range=True, order=1, mode='constant', cval=0,  output_shape=output_shape)
        #img_trans = skimage.transform.warp(img, affine_inv, preserve_range=True, order=1, mode='constant', cval=0,  output_shape=output_shape)
        img_trans = skimage.transform.warp(img, affine_inv, preserve_range=True, order=1, mode='edge', output_shape=output_shape)
        img_trans = img_trans.astype(img.dtype) # somehow the warping transforms it back to double, undo this for float32
        # img_trans = np.array(img_trans * 255).astype(img.dtype)
        
        # old separate transformation
        #img_trans = skimage.transform.warp(img.astype(float), np.linalg.inv(trans_2d_comb),  order=3, mode='constant', cval=0) # TODO: output_shape=(output_shape[0],output_shape[1])
        #img_trans = skimage.transform.resize(img_trans, output_shape, order=3)
        #img_trans = np.array(img_trans).astype(img.dtype) # without array it becomes matrix, which somehow is iterable and causes an infinite recursion loop in the dataloader
        return img_trans

    # row transformations, TODO: test again
    @staticmethod
    def applyLableTransformation(label_transformation_comb, points_Njoints_X):
        if label_transformation_comb is None:
            return points_Njoints_X
        # reshape to be applicable to points(1D array) as well as heatmaps (2D array)
        numJoints = points_Njoints_X.shape[0]
        pose_trans = np.reshape(points_Njoints_X, (numJoints,-1))
        pose_trans = np.dot(label_transformation_comb.T, pose_trans)
        return np.reshape(pose_trans, points_Njoints_X.shape)

    @staticmethod
    def apply2D(trans_2d_comb, pose_2d_Njoints_x_2):
        return projection.applyHomoTransform(trans_2d_comb, pose_2d_Njoints_x_2.T).T
    @staticmethod
    def apply3D(trans_3d_comb, pose_3d_Njoints_x_3):
        # TODO: also projective transformations?
        return np.dot(pose_3d_Njoints_x_3, trans_3d_comb[:3,:3].T)
         
    def apply(self, transformations, data_dict, output_shape_pixel):
        #(trans_2d_comb, trans_3d_comb, trans_intr_comb, label_transformation_comb) = transformations
        trans_2d_comb = transformations['trans2D']
        trans_3d_comb = transformations['trans3D']
        trans_intr_comb = transformations['transIntr']
        label_transformation_comb = transformations['transLabel']

        for key, label in data_dict.items():
#            if label == []: # or (type(label).__module__ == np.__name__ and label.size==0): # ignore empty entries
#                data_dict[key] = label
#                continue
            if label is None:
                print("Got None", key)
                assert False;
                
            if key in ['img', 'img_crop', 'depth_map', 'bg_crop']:
                data_dict[key] = self.apply2DImage(trans_2d_comb, label, output_shape_pixel)
            elif key in ['2D_heat']:
                maps_trans = self.apply2DImage(trans_2d_comb, label, output_shape_pixel)
                maps_trans_T = maps_trans.transpose((2, 1, 0)) # since flipping assumed the first dimension to be the joint index
                maps_trans_T = self.applyLableTransformation(label_transformation_comb, maps_trans_T)
                maps_trans =  maps_trans_T.transpose((2, 1, 0)) # TODO: more efficient version? Give dimension across which to apply the mapping?
                data_dict[key] = maps_trans.astype('float32')
            elif key in ['2D','2D_orig']:
                pose_2d = np.reshape(label, (-1,2))
                pose_2d_trans = self.apply2D(trans_2d_comb, pose_2d)
                pose_2d_trans = self.applyLableTransformation(label_transformation_comb, pose_2d_trans)
                data_dict[key] = np.reshape(np.array(pose_2d_trans, dtype='float32'), (-1))
            elif key in ['3D', '3D_global','3D_h36m', '3D_cap', '3D_crop_coord','latent_3d']:
                pose_3d = np.reshape(label, (-1,3))
                pose_3d_trans = self.apply3D(trans_3d_comb, pose_3d)
                pose_3d_trans = self.applyLableTransformation(label_transformation_comb, pose_3d_trans)
                data_dict[key] = np.reshape(np.array(pose_3d_trans, dtype='float32'), (-1)) #pose_3d_trans, dtype='float32')
            elif key in ['extrinsic','extrinsic_rot','aug_3D']:
                matrix = label
                rows = int(np.sqrt(matrix.size))
                matrix = matrix.reshape( (rows,rows))
                mat_trans = np.dot(trans_3d_comb[:rows,:rows], matrix)
                data_dict[key] = np.array(mat_trans).astype('float32')
            elif key in ['extrinsic_inv','extrinsic_rot_inv','aug_3D_inv']:
                matrix = label
                rows = int(np.sqrt(matrix.size))
                matrix = matrix.reshape( (rows,rows))
                mat_trans = np.dot(matrix, np.linalg.inv(trans_3d_comb)[:rows,:rows])
                data_dict[key] = np.array(mat_trans).astype('float32')
            elif key in ['intrinsic_crop']:
                matrix = label
                mat_trans = np.dot(trans_intr_comb[:3,:3], matrix)
                data_dict[key] = np.array(mat_trans).astype('float32')
            else: # no transformation for this type
                pass
                #data_dict[key] = label
        return

def axisAngleFromRMat(R):
    angle = np.arccos((np.trace(R)-1)/2)
    sin_angle_2 = 2*np.sin(angle)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / sin_angle_2
    return axis, angle
####################################################
# The following classes implement transformations on the image, the 3D pose, and intrinsics in a coherent way.
# I.e. after the transformation the 3D pose should still project perfectly onto the image/heatmap/2Dpose
# intrinsic matrix: from normalized cam coordinates to normalized (0..1) crop coordinates

class LinearPerspectiveCorrection(object):
    ''' Correct perspective effect during cropping, by rotating the optical axis to the crop center and zooming to maintain the original focal length.
         This compensates artifacts occurring with large FOV cameras '''

    def __init__(self, shear_augmentation=0):
        #self.normalize_focal_length = normalize_focal_length
        self.shear_augmentation = shear_augmentation

    def parametrize_and_randomize(self, label_dict, batch_index):
        assert 'bounding_box_cam' in label_dict
        assert 'intrinsic_crop' in label_dict
        #bbox     = label_dict['bounding_box']
        bbox_cam    = label_dict['bounding_box_cam']
        K_crop_orig = label_dict['intrinsic_crop']
        
        # get rotation to new optical center
        pose_2d_center_cam = np.array([bbox_cam[0]+bbox_cam[2]/2, bbox_cam[1]+bbox_cam[3]/2, 1])
        if len(bbox_cam.shape) > 1:
            print("len(bbox_cam.shape) > 1:")
            IPython.embed()
        theta, phi = projection.getAizmuthAndEvaluation(pose_2d_center_cam)
        R_virtualCamToOrigCam_ = projection.rotationMatrix(theta, phi)
        R_origCamToVirtualCam = np.linalg.inv(R_virtualCamToOrigCam_)

        # Get perceived scaling due to perspective change, the virtual focal length is larger due to rotating the new view ray to the center
        # Without adaption the crop would shrink due to the additional distance after rotation. The adaption compensates this.
        center_new = pose_2d_center_cam @ R_origCamToVirtualCam.T
        focal_length_factor = center_new[0,2]
        
        if self.shear_augmentation > 0:
            rand_2d_pose = np.random.uniform(-self.shear_augmentation,self.shear_augmentation,2)
            center_aug = np.append(rand_2d_pose, 1)
            theta_aug, phi_aug = projection.getAizmuthAndEvaluation(center_aug)
            R_aug = projection.rotationMatrix(theta_aug, phi_aug) # no inverse needed, since we anyways want to rotate in all directions equally, instead we negate C
            #c = R_aug @ np.array([0,0,1])
            #c = c / c[0,2]
            # center the original center
            C_aug = np.eye(3)
            C_aug[:3,2] = -center_aug
            
            # crop again around the projected 2D points, to have the same margin as before
            if '2D' in label_dict: 
                trans_to_normalized = R_aug @ R_origCamToVirtualCam @ la.inv(K_crop_orig)
                pose_2d_cam_orig = label_dict['2D'].reshape(-1, 2)
                width_orig = np.max(np.max(pose_2d_cam_orig, axis=0)-np.min(pose_2d_cam_orig, axis=0))
                pose_2d_cam_def  = projection.applyHomoTransform(trans_to_normalized, pose_2d_cam_orig.T).T
                min_v = np.min(pose_2d_cam_def, axis=0)
                max_v = np.max(pose_2d_cam_def, axis=0)
                max_width = np.max(max_v - min_v)
                center = (min_v + max_v)/2
                
                # compensate for the focal length we will set later on
                focal_length_factor_aug = (pose_2d_center_cam @ R_origCamToVirtualCam.T)[0,2]
                focal_length_virtual = focal_length_factor_aug*K_crop_orig[0,0]
                C_aug = np.eye(3)
                C_aug[0,0]  = C_aug[1,1] = max_width/width_orig * focal_length_virtual
                C_aug[:2,2] = center
                
                focal_length_factor = focal_length_factor_aug # good thing to do?
            
            # combined rotation(3d rot==2D projective) and shear(2D translate=3D shear)
            D_aug = la.inv(C_aug) @ R_aug
            
            R_origCamToVirtualCam = D_aug @ R_origCamToVirtualCam

            #print("center_aug={}, c={}".format(center_aug))

#             print(A)
#             V = K_crop @ A @ la.inv(K_crop_orig)
#        D = A @ la.inv(K_crop) @ K_crop_orig

        # we define the optical center to be in the center of the crop, and to have equal focal length
        #if self.normalize_focal_length:
        f_avg = K_crop_orig[0,0]
        f_compensated = f_avg * focal_length_factor
        K_crop = np.diagflat([f_compensated,f_compensated,1])
        #else:
        #    K_crop = np.diagflat([focal_length_factor*K_crop_orig[0,0], focal_length_factor*K_crop_orig[1,1], 1])
        K_crop[:2,2]  = [0.5, 0.5]

        # get transformation from original crop (0..1, not centered) to new crop(0..1, centered) given the new rotation
        S_crop = K_crop @ R_origCamToVirtualCam @ np.linalg.inv(K_crop_orig)  # crop relative -> cam -> transform to virtual cam -> virtual crop relative
        R_homo = np.mat(np.eye(4))
        R_homo[:3,:3] = R_origCamToVirtualCam
 
        if 0 and self.shear_augmentation: # debug stuff
            pose_2d_cam_orig = label_dict['2D'].reshape(-1,2)
            pose_2d_cam_def  = projection.applyHomoTransform(S_crop, pose_2d_cam_orig.T).T
            width_new = np.max(np.max(pose_2d_cam_def, axis=0)-np.min(pose_2d_cam_def, axis=0))
            #print('width_orig',width_orig,'width_new',width_new)
            s,V = scipy.linalg.eig(R_homo[:3,:3])
            print('det(R_homo)', la.det(R_homo[:3,:3]))
            print('s', s, 'V',V)

        # Trace(R) = 1 + 2 cos(alpha) => alpha = acos((Tr(R)-1)/2)
        #a1 = angleFromMat(R_origCamToVirtualCam)*180/np.pi
        #
        #        if a1>10:          
        #            IPython.embed()

        # set intrinsic to the new virtual camera parameters (i.e. invert previous K -> identity mapping, and then apply new K)
        # it does not depent on the rotation, since that part is compensated by rotating the global pose
        # WARNING:: this assumes that the intrinsics have not yet been modified! I.e. perspective correction should always be applied first
        transIntr = np.dot(K_crop, np.linalg.inv(K_crop_orig))
        return {'trans2D': S_crop, 'trans3D': R_homo, 'transIntr': transIntr}

class LinearRectangularCorrection(object):
    ''' Correct perspective effect during cropping, by rotating the optical axis to the crop center and zooming to maintain the original focal length.
         This compensates artifacts occurring with large FOV cameras '''

    def __init__(self, shear_augmentation=0):
        self.num_smaller_01 = 0
        self.num_smaller_005 = 0
        self.num_total = 0
        self.shear_augmentation = shear_augmentation

    def parametrize_and_randomize(self, label_dict, batch_index):
        assert 'bounding_box_cam' in label_dict
        assert 'intrinsic_crop' in label_dict
        bbox_cam    = label_dict['bounding_box_cam']
        K_crop_orig = label_dict['intrinsic_crop']

        # get rotation to new optical center
        pose_2d_center = np.array([bbox_cam[0]+bbox_cam[2]/2, bbox_cam[1]+bbox_cam[3]/2, 1])
        if len(bbox_cam.shape) > 1:
            print("len(bbox_cam.shape) > 1:")
            IPython.embed()
 
        # we define the optical center to be in the center of the crop, 
        # and to have equal focal length, the one of the x-axis
        f_crop = K_crop_orig[0,0]
        K_crop = np.diagflat([f_crop, f_crop, 1])
        K_crop[:2,2]  = [0.5, 0.5]
        
        D = la.inv(K_crop) @ K_crop_orig
        
        if 0:
            theta, phi = projection.getAizmuthAndEvaluation(pose_2d_center)
            R_virtualCamToOrigCam = projection.rotationMatrix(theta, phi)
            axis,angle = axisAngleFromRMat(R_virtualCamToOrigCam)        
            if 1: #angle*180/np.pi > 10:          
                I = D @ D.T
                II = np.array(I,dtype=int)  
                DI = np.array(10*D, dtype=int)/10
    
                u,p = scipy.linalg.polar(D)
                s,V = scipy.linalg.eig(D)
                
                x = np.random.rand(3)
                Dx = D @ x
                norm_change = la.norm(x) / la.norm(Dx)
                norm_diff = np.abs(1-norm_change)
                if norm_diff>0.1:
                    self.num_smaller_01 += 1
                if norm_diff>0.05:
                    self.num_smaller_005 += 1
                self.num_total +=1

                print('<0.1:',self.num_smaller_01/self.num_total,'<0.05:',self.num_smaller_005/self.num_total,
                      'det=',la.det(D),'norm_change',norm_change,'norm_diff',norm_diff)#,'\ntransformation=\n',I)
                
                if 0:
                    IPython.embed()
                    
                    pose_3d = label_dict['3D']
                    import matplotlib.pyplot as plt
                    plt.switch_backend('Qt5Agg')
                    fig = plt.figure(0)
                    ax = fig.add_subplot(1,2,1, projection='3d')  
                    utils_plt.plot_3Dpose_simple(ax, pose_3d)
                    
                    plt.plot()
            

        D_homo = np.mat(np.eye(4))
        D_homo[:3,:3] = D
        transIntr = np.dot(K_crop, np.linalg.inv(K_crop_orig))
        
        #if self.shear_augmentation > 0:
        #    return {'trans2D': V, 'trans3D': D_homo, 'transIntr': transIntr}
        #else:
        return {'trans3D': D_homo, 'transIntr': transIntr}


class LinearScaleCentered(object):
    ''' Zooming into the image around the crop center.
        Note!: perform perspective correction before to maintain coherence between 2D and 3D annotation '''

    def __init__(self, factor_range= (0.7,1.3), factor_into_intrinsics=True):
        self.factor_range = factor_range
        self.factor_into_intrinsics = factor_into_intrinsics

    def parametrize_and_randomize(self, labels, batch_index):
        scale = random.uniform(self.factor_range[0], self.factor_range[1])

        # 2D
        center = np.array([1,1])/2 #[input_shape2D[:2]]
        offset = center -scale*center
        #print('img_shape', img_shape)
        S_scale_centered = np.eye(3,3) * scale
        S_scale_centered[:2,2] = offset
        S_scale_centered[2,2]  = 1
        
        # 3D
        S_scale = np.eye(4,4) * scale
        S_scale[2,2]  = 1
        S_scale[3,3]  = 1
        
        # usually transintr should be the same as trans2D
        if self.factor_into_intrinsics:
            return {'trans2D': S_scale_centered, 'transIntr': S_scale_centered}
        else: # keep the intrinsics, but scale the 3D scene to account for 'larger' projection=scaled image
            return {'trans2D': S_scale_centered, 'trans3D': S_scale}
        
class LinearMultiScale(object):
    ''' Selective zooming factor based on batch index
        Note!: perform perspective correction before to maintain coherence between 2D and 3D annotation (by having the optical center at the crop center)'''
    def __init__(self, fixed_factors=(0.7,1,1.3)):
        self.fixed_factors = fixed_factors

    def parametrize_and_randomize(self, labels, batch_index):
        assert batch_index is not None and batch_index < len(self.fixed_factors)
        scale = self.fixed_factors[batch_index]

        # 2D
        center = np.array([1,1])/2
        offset = center - scale*center
        S_scale_centered = np.eye(3,3) * scale
        S_scale_centered[:2,2] = offset
        S_scale_centered[2,2]  = 1

        # TODO:TODO 3D bring closer or further apart!
        
        return {'trans2D': S_scale_centered, 'transIntr': S_scale_centered}


class LinearRotate(object):
    ''' Rotate the virtual camera (extrinsic matrix) and the 2D and 3D annotation relative to the camera.
        Note!: perform perspective correction before to maintain coherence between 2D and 3D annotation (by having the optical center at the crop center)'''
    def __init__(self, angle_range_degree= (-30,30)):
        self.angle_range_rad = np.array(angle_range_degree)*math.pi/180

    def parametrize_and_randomize(self, label_dict, batch_index):
        self.theta = random.uniform(self.angle_range_rad[0], self.angle_range_rad[1])

        # 2D
        center = np.array([[1,1]]).T/2
        center = np.append(center, [[1]], 0)
        R = np.mat([[cos(self.theta),-sin(self.theta), 0],
                    [sin(self.theta), cos(self.theta), 0],
                    [         0,          0, 1]])
        offset = center - R*center
        lin_trans = R
        lin_trans[:,2]  = offset
        lin_trans[2,2]  = 1

        # 3D
        R = np.mat([[cos(self.theta),-sin(self.theta), 0, 0],
            [sin(self.theta), cos(self.theta), 0, 0],
            [         0,          0, 1,0],
            [         0,          0, 0,1]])

        return {'trans2D': lin_trans, 'trans3D': R}

class LinearRotateWorld(object):
    ''' Rotate whole 3D space. WANRING: such out of plane rotation does not work for images.
        Note!: perform perspective correction before to maintain coherence between 2D and 3D annotation
        Note, this is slightly inaccurate due to the perspective projection which is not accounted for at this point. One would need to apply the transformation on extrinsic_rot_inv immediately'''
    def __init__(self, angle_range_degree= (-90,90)):
        self.angle_range_rad = np.array(angle_range_degree)*math.pi/180

    def parametrize_and_randomize(self, label_dict, batch_index):
        self.theta = random.uniform(self.angle_range_rad[0], self.angle_range_rad[1])

        IPython.embed()
        external_rotation_global = projection_util.rotationMatrixXZY(0, self.theta, 0) # Rotates for H36M, needs to be different for 3DHP!!!
        R_cam_2_world = label_dict['extrinsic_rot_inv'][0] #.numpy()
        R_world_in_cam = la.inv(R_cam_2_world) @ external_rotation_global @ R_cam_2_world

        return {'trans3D': R_world_in_cam}

class LinearFlip(object):
    ''' Flip left and right. WARNING: flipping of labels is not yet implemented in the multi-view loss. Requires to expose the label transformation matrix.
        Note!: perform perspective correction before to maintain coherence between 2D and 3D annotation (by having the optical center at the crop center)'''
    def __init__(self, num_joints_input, num_joints_output, horizontal=False, vertical=False, bone_symmetry=None):
        assert bone_symmetry is not None
        self.horizontal= horizontal
        self.vertical  = vertical
        self.bone_symmetry = bone_symmetry
        self.num_joints_input  = num_joints_input
        self.num_joints_output = num_joints_output

    def parametrize_and_randomize(self, label_dict, batch_index):
        if self.horizontal:
            self.sign_h = random.choice([1, -1])
        else:
            self.sign_h = 1
        if self.vertical:
            self.sign_v = random.choice([1, -1])
        else:
            self.sign_v = 1
        #print('sign h,v', self.sign_h, self.sign_v)

        # 2D
        center = np.array([[1,1]]).T/2 # [input_shape2D[:2]]
        center = np.append(center, [[1]], 0)
        F = np.mat([[self.sign_h,0, 0],
                    [0, self.sign_v, 0],
                    [         0,          0, 1]])
        offset = center - F*center
        lin_trans_2D = F
        lin_trans_2D[:,2]  = offset
        lin_trans_2D[2,2]  = 1

        # 3D
        F = np.mat([[self.sign_h,0, 0, 0],
                    [0, self.sign_v, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        # label
        numFlips = int((self.sign_h-1)/2 + (self.sign_v-1)/2)
        if numFlips % 2 == 1:
            label_trans = np.zeros( (self.num_joints_output, self.num_joints_input) )
            for symmetry in self.bone_symmetry:
                if np.max(symmetry) < min(self.num_joints_output, self.num_joints_input):
                    label_trans[symmetry[0], symmetry[1]] = 1
                    label_trans[symmetry[1], symmetry[0]] = 1
        else:
            label_trans =  None

        return {'trans2D': lin_trans_2D, 'trans3D': F, 'transLabel': label_trans}

# class LinearCroppedImage(object):
#     ''' transforms intrinsics and 2D points according to the initial bounding box'''
# 
#     def __init__(self, bbox):
#         self.bbox = bbox # range relative to image size
# 
#     def parametrize_and_randomize(self, label_dict, batch_index):
#         self.offset_relative = [random.uniform(-self.jitter_range[0], self.jitter_range[0]),
#                                 random.uniform(-self.jitter_range[1], self.jitter_range[1])]
# 
#         #print('sign h,v', self.sign_h, self.sign_v)
#         offset = np.array(self.offset_relative)
#         lin_trans_2D = np.eye(3,3)
#         lin_trans_2D[:2,2] = offset
#         lin_trans_2D[2,2]  = 1
#         return {'trans2D': lin_trans_2D, 'transIntr': lin_trans_2D}

class LinearCropJitter(object):
    '''  TODO '''

    def __init__(self, jitter_range):
        self.jitter_range = jitter_range # range relative to image size

    def parametrize_and_randomize(self, label_dict, batch_index):
        self.offset_relative = [random.uniform(-self.jitter_range[0], self.jitter_range[0]),
                                random.uniform(-self.jitter_range[1], self.jitter_range[1])]

        #print('sign h,v', self.sign_h, self.sign_v)
        offset = np.array(self.offset_relative)
        lin_trans_2D = np.eye(3,3)
        lin_trans_2D[:2,2] = offset
        lin_trans_2D[2,2]  = 1
        
        # TODO:TODO 3D shift 3D left-right

        return {'trans2D': lin_trans_2D, 'transIntr': lin_trans_2D}


# utility functions
def projective_to_crop_relative_np(jointPositions_projective, K_crop):
    jointPositions_cam = jointPositions_projective / jointPositions_projective[0,2]
    f_avg = (K_crop[0,0] + K_crop[1,1]) / 2   # set average focal length to third dimension to equalize magnitude of xy and z components
    
    # map to crop-relative coordinates
    jointPositions_crop = np.dot(jointPositions_cam, K_crop.T)

    # projection to image and normalization to image center (in xy direction) and focal length (z direction)
    xy_norm = jointPositions_crop[:,:2] / jointPositions_crop[:,2,np.newaxis] - 0.5
    z_norm  = jointPositions_crop[:, 2]*f_avg - f_avg
    
    jointPositions_crop = np.concatenate([xy_norm, z_norm.reshape(-1,1)],1)
    return jointPositions_crop, jointPositions_cam

def crop_relative_to_projective_tvar(jointPositions_crop, K_crop):
    f_avg = (K_crop.data[0,0] + K_crop.data[1,1]) / 2
    K_crop_inv = torch.autograd.Variable(torch.inverse(K_crop.data))

    # undo normalization
    z_proj  =  (jointPositions_crop[:, 2] + f_avg)/ f_avg
    xy_proj = (jointPositions_crop[:,:2] + 0.5) * z_proj.view(-1,1).expand_as(jointPositions_crop[:,:2])
    
    # map to camera-relative coordinates
    jointPositions_crop = torch.cat([xy_proj, z_proj.view(-1,1)], 1)
    jointPositions_cam = torch.mm(jointPositions_crop, K_crop_inv.transpose(0,1))
    
    return jointPositions_cam

def projective_to_crop_relative_weak_np(jointPositions_projective, K_crop):
    jointPositions_weak_cam = jointPositions_projective / jointPositions_projective[0,2]
    f_avg = (K_crop[0,0] + K_crop[1,1]) / 2   # set average focal length to third dimension to equalize magnitude of xy and z components
    jointPositions_weak_crop = np.dot(jointPositions_weak_cam, K_crop.T)

    xy_norm = jointPositions_weak_crop[:,:2] - 0.5
    z_norm = (jointPositions_weak_crop[:, 2]-1)*f_avg

    jointPositions_crop = np.concatenate([xy_norm, z_norm.reshape(-1,1)],1)
    return jointPositions_crop, jointPositions_weak_cam

def crop_relative_weak_to_projective_tvar(jointPositions_crop, K_crop):
    f_avg = (K_crop.data[0,0] + K_crop.data[1,1]) / 2
    K_crop_inv = torch.autograd.Variable(torch.inverse(K_crop.data))

    z_proj  =  jointPositions_crop[:, 2] / f_avg + 1
    xy_proj = (jointPositions_crop[:,:2] + 0.5)

    jointPositions_weak_crop = torch.cat([xy_proj, z_proj.view(-1,1)], 1)
    jointPositions_weak_cam = torch.mm(jointPositions_weak_crop, K_crop_inv.transpose(0,1))
    return jointPositions_weak_cam


#
#
#
#
# def projective_to_crop_normalized_weak_np(jointPositions_projective, K_crop):
#     K_crop_3d = K_crop.clone()
#     K_crop_3d[2,2] = (K_crop[0,0]+K_crop[1,1])/2
#     jointPositions_weak_cam = jointPositions_projective / jointPositions_projective[0,2]
#     jointPositions_weak_crop = np.dot(jointPositions_weak_cam, K_crop_3d.T)
#     jointPositions_weak_crop -= torch.tensor([[0,0,K_crop_3d[2,2]]])
#     return jointPositions_weak_crop, jointPositions_weak_cam
#
# def crop_normalized_to_weak(jointPositions_crop, K_crop):
#     K_crop_3d = K_crop.clone()
#     K_crop_3d[2,2] = (K_crop[0,0]+K_crop[1,1])/2
#     if isinstance(K_crop, torch.autograd.Variable):
#         K_crop_inv = torch.autograd.Variable(torch.inverse(K_crop_3d.data).transpose(0,1))
#     else:
#         K_crop_inv = torch.inverse(K_crop_3d).transpose(0,1)
#
#     jointPositions_crop += torch.tensor([[0,0,K_crop_3d[2,2]]]).expand_as(jointPositions_crop)
#     jointPositions_cam = np.dot(jointPositions_crop, K_crop_inv.T)
#     return jointPositions_cam

