import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os, shutil
#from pytorch_human_reconstruction.configs.config_3dpose_multi_H36M import config_dict

sys.path.insert(0,'./')
sys.path.insert(0,'../')
sys.path.insert(0,'../../')

import numpy as np

import math
import torch
import torch.optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from time import gmtime, strftime

from datasets.SkiPTZ import SkiPanTiltDataset_DLT

from datasets import utils as utils_data
from util import util as utils_generic


from datasets import transforms as transforms_aug
from PlottingUtil import util as utils_plt

import training

import IPython
import scipy.ndimage.filters

class DatasetFactory():
    def update_defaults(self, config_dict):
        self.config_dict = config_dict
        
        print("Config_common_datasets.update_defaults(self)")
        self.mean   = config_dict['img_mean']
        self.stdDev = config_dict['img_std']

        class Image256toTensor(object):
            def __call__(self, pic):
                img = torch.from_numpy(pic.transpose((2, 0, 1))).float()
                img = img.div(255)
                return img

            def __repr__(self):
                return self.__class__.__name__ + '()'

        self.transform_in = transforms.Compose([
            Image256toTensor(),
            transforms.Normalize(self.mean, self.stdDev) # HACK
        ])
        self.transform_out = []

        
        self.seam_scaling = config_dict['seam_scaling']
        self.every_nth_frame_test = 1
        self.bones = utils_plt.bones_h36m
        
        if config_dict['training_set'] == 'ski':
            self.seam_scaling = 1.6
        if config_dict['training_set'] == 'folder':
            self.seam_scaling = 2
        if config_dict['training_set'] in ['ski_spoerri']:
            self.every_nth_frame_test = 100

        if config_dict['training_set'] == 'hand':
            self.mean   = (0, 0, 0)
            self.stdDev = (1, 1, 1)
            self.every_nth_frame_test = 100
            self.bones = utils_plt.bones_hand
            
        #### 3D #### initialize augmentation code ####
        scale_factor_into_intrinsics = config_dict.get('factor_into_intrinsics',True)
        self.augmentation_train = None
        if any([config_dict[k] is not False for k in ['perspectiveCorrection', 'rotation_augmentation', 'scale_augmentation', 'mirror_augmentation']]):
            self.augmentation_train = transforms_aug.ApplyLinearTransformation()
        if config_dict['perspectiveCorrection']: # note, must be the first transformation
            if config_dict['perspectiveCorrection'] is 'Rect':
                self.augmentation_train.add(transforms_aug.LinearRectangularCorrection())
            else:
                self.augmentation_train.add(transforms_aug.LinearPerspectiveCorrection(shear_augmentation=config_dict['shear_augmentation']))
        if config_dict['rotation_augmentation']:
            self.augmentation_train.add(transforms_aug.LinearRotate((-10, 10)))
            self.augmentation_train.add(transforms_aug.LinearRotate((-10, 10)))
        if config_dict['scale_augmentation']:
            #self.augmentation_train.add(transforms_aug.LinearScaleCentered((0.95, 1.05), factor_into_intrinsics=scale_factor_into_intrinsics))
            #self.augmentation_train.add(transforms_aug.LinearScaleCentered((0.95, 1.05), factor_into_intrinsics=scale_factor_into_intrinsics))
            self.augmentation_train.add( # HACK NEW scaling values
                 transforms_aug.LinearScaleCentered((0.80, 1.20), factor_into_intrinsics=scale_factor_into_intrinsics))
            self.augmentation_train.add(
                 transforms_aug.LinearScaleCentered((0.80, 1.20), factor_into_intrinsics=scale_factor_into_intrinsics))
        if 'crop_jitter_augmentation' in config_dict:
            jrange = config_dict['crop_jitter_augmentation']
            self.augmentation_train.add(transforms_aug.LinearCropJitter(jitter_range=(jrange, jrange)))


        if config_dict['seam_scaling'] != 1.0:
             self.augmentation_train.add(transforms_aug.LinearScaleCentered((self.seam_scaling, self.seam_scaling)))
        if config_dict['mirror_augmentation']:
            mirror_aug = transforms_aug.LinearFlip(num_joints_input=17, num_joints_output=17, horizontal=True, bone_symmetry=utils_plt.joint_symmetry_h36m)
            if config_dict['training_set'] in ['ski_spoerri']:
                mirror_aug = transforms_aug.LinearFlip(num_joints_input=19, num_joints_output=19, horizontal=True, bone_symmetry=utils_plt.joint_symmetry_spoerri)
            self.augmentation_train.add(mirror_aug)

        #### 3D #### Testing ####
        self.augmentation_test = None
        if config_dict['perspectiveCorrection']: # note, must be the first transformation
            self.augmentation_test = transforms_aug.ApplyLinearTransformation()
            if config_dict['perspectiveCorrection'] is 'Rect':
                pass 
                #self.augmentation_2d_train.add(transforms_aug.LinearRectangularCorrection())
            else:   
                self.augmentation_test.add(transforms_aug.LinearPerspectiveCorrection())

        #if config_dict['training_set'] == 'folder':
        #    self.multi_scales = [1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
        #    self.augmentation_test.add(transforms_aug.LinearMultiScale(self.multi_scales))
            
        #### 2D #### initialize augmentation code ####
        self.augmentation_2d_train = transforms_aug.ApplyLinearTransformation()
        #if self.augmentation:
        if config_dict['perspectiveCorrection']: # note, must be the first transformation
                self.augmentation_2d_train.add(transforms_aug.LinearPerspectiveCorrection())
        if config_dict['scale_augmentation']:
            #self.augmentation_2d_train.add(transforms_aug.LinearScaleCentered((0.5, 1.1))) # For person detection
            sval = config_dict['scale_augmentation']
            self.augmentation_2d_train.add(transforms_aug.LinearScaleCentered((1-sval,1+sval)))
        if config_dict['rotation_augmentation']:
            self.augmentation_2d_train.add(transforms_aug.LinearRotate((-10, 10)))
            self.augmentation_2d_train.add(transforms_aug.LinearRotate((-10, 10)))
        if config_dict['mirror_augmentation']:
            mirror_aug = transforms_aug.LinearFlip(num_joints_input=17, num_joints_output=17, horizontal=True, bone_symmetry=utils_plt.joint_symmetry_h36m)
            if config_dict['training_set'] in ['ski_spoerri']:
                mirror_aug = transforms_aug.LinearFlip(num_joints_input=19, num_joints_output=19, horizontal=True, bone_symmetry=utils_plt.joint_symmetry_spoerri)
            self.augmentation_2d_train.add(mirror_aug)
        if 'crop_jitter_augmentation' in config_dict:
            jrange =  config_dict['crop_jitter_augmentation']
            self.augmentation_2d_train.add(transforms_aug.LinearCropJitter(jitter_range=(jrange, jrange)))
        self.augmentation_2d_train.add(transforms_aug.LinearScaleCentered((self.seam_scaling, self.seam_scaling)))

        self.augmentation_2d_test = transforms_aug.ApplyLinearTransformation()
        self.augmentation_2d_test.add(transforms_aug.LinearScaleCentered((self.seam_scaling, self.seam_scaling)))
           

    def load_data_train_ski_spoerri(self):
        #assert not any(v in self.config_dict['actor_subset'] for v in self.config_dict.get('actor_subset_test',[5])+self.config_dict.get('actor_subset_validation',[4]) ) # strict separation of training and testing
        trainset = SkiPanTiltDataset_DLT(
                base_folder='/cvlabsrc1/cvlab/datasets_ski/Kuetai_2011/',
                   #subjects = [0,1,2,3,4],
                   subjects = self.config_dict['actor_subset'], #,1,2,3,4],
                   active_cameras=self.config_dict['active_cameras'],
                   study_id = self.config_dict.get('study_id', 1),
                   augmentation=self.augmentation_train,
                   input_types=self.config_dict['input_types'], label_types=self.config_dict['label_types_train'], input_transform=self.transform_in,
                   input_img_width=self.config_dict['inputDimension'], map_width=self.config_dict['outputDimension_2d'],
                   useCamBatches=self.config_dict['useCamBatches'],
                   joint_transformation=self.config_dict.get('joint_transformation',utils_plt.ski_spoerri_to_h36m),
                   root_index = self.config_dict.get('root_index',utils_plt.root_index_h36m),
                   bbox_margin = self.config_dict['bbox_margin'],
                   every_nth_frame=1,
                   randomize=True,
                   )

        if self.config_dict['check_val_score']:
            cam_batches = 0
            b_size = self.config_dict['batch_val_J_score']
        else:
            cam_batches = self.config_dict['useCamBatches']
            b_size = self.config_dict['batch_size_test']

        valset = SkiPanTiltDataset_DLT(
                base_folder='/cvlabsrc1/cvlab/datasets_ski/Kuetai_2011/',
                   subjects = self.config_dict.get('actor_subset_validation', [4]),
            active_cameras = self.config_dict['active_cameras'],
                   study_id = self.config_dict.get('study_id', 1),
                   augmentation=self.augmentation_test,
                   input_types=self.config_dict['input_types'], label_types=self.config_dict['label_types_train'], input_transform=self.transform_in,
                   input_img_width=self.config_dict['inputDimension'], map_width=self.config_dict['outputDimension_2d'],
                   useCamBatches=cam_batches,
                   joint_transformation=self.config_dict.get('joint_transformation',utils_plt.ski_spoerri_to_h36m),
                   root_index = self.config_dict.get('root_index',utils_plt.root_index_h36m),
                   bbox_margin = self.config_dict['bbox_margin'],
                   every_nth_frame=1,
                   randomize=False, check_score=self.config_dict['check_val_score']
                   )
        if 'test_enabled' in self.config_dict.keys():
            if self.config_dict['test_enabled'] == True:
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.config_dict['batch_size_train'],
                                                          shuffle=False, num_workers=self.config_dict['num_workers'],
                                                          pin_memory=False, drop_last=True,
                                                          collate_fn=utils_data.default_collate_with_string)
            else:
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.config_dict['batch_size_train'],
                                                          shuffle=True, num_workers=self.config_dict['num_workers'],
                                                          pin_memory=False, drop_last=True,
                                                          collate_fn=utils_data.default_collate_with_string)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.config_dict['batch_size_train'], shuffle=True, num_workers=self.config_dict['num_workers'], pin_memory=False, drop_last=True, collate_fn=utils_data.default_collate_with_string)

        valloader  = torch.utils.data.DataLoader(valset,  batch_size=b_size, shuffle=False, num_workers=self.config_dict['num_workers'], pin_memory=False, drop_last=True, collate_fn=utils_data.default_collate_with_string)
        return trainloader, valloader

    def load_data_test_ski_spoerri(self):        
        testset = SkiPanTiltDataset_DLT(
                base_folder='/cvlabsrc1/cvlab/datasets_ski/Kuetai_2011/',
                   subjects = [5], #self.config_dict.get('actor_subset_test', [5]),
            active_cameras=self.config_dict['active_cameras'],
            study_id = self.config_dict.get('study_id', 1),
                   augmentation=self.augmentation_test,
                   input_types=self.config_dict['input_types'], label_types=self.config_dict['label_types_train'], input_transform=self.transform_in,
                   input_img_width=self.config_dict['inputDimension'], map_width=self.config_dict['outputDimension_2d'],
                   useCamBatches=self.config_dict['useCamBatches'],
                   joint_transformation=self.config_dict.get('joint_transformation',utils_plt.ski_spoerri_to_h36m),
                   root_index = self.config_dict.get('root_index',utils_plt.root_index_h36m),
                   bbox_margin = self.config_dict['bbox_margin'],
                   useSequentialFrames=0,
                   every_nth_frame=10,
                   randomize=False# HACK TODO should be one
                   )
        return testset


    
    def load_data_train_ski_mix(self):
        return self.load_data_train_ski_spoerri() 
    def load_data_test_ski_mix(self):
        return self.load_data_test_ski_mayer()


    def load_data_train(self, config_dict):
        self.update_defaults(config_dict)
        training_set = config_dict['training_set']
        if training_set == 'ski_spoerri':
            trainloader, valloader = self.load_data_train_ski_spoerri()

        return trainloader, valloader
    
    def load_data_test(self, config_dict):
        self.update_defaults(config_dict)

        training_set = config_dict['training_set']

        if training_set == 'ski_spoerri':
            testset = self.load_data_test_ski_spoerri()
        self.testset_instance = testset
        testloader  = torch.utils.data.DataLoader(testset,  batch_size=self.config_dict['batch_size_test'], shuffle=False, num_workers=self.config_dict['num_workers'], pin_memory=False, collate_fn=utils_data.default_collate_with_string)
        return testloader

    def load_data_predict(self, config_dict):
        self.update_defaults(config_dict)

        training_set = config_dict['training_set']

        if training_set == 'ski_spoerri':
            testset = self.load_data_test_ski_spoerri()
       
        self.testset_instance = testset
        testloader  = torch.utils.data.DataLoader(testset,  batch_size=self.config_dict['batch_size_test'], shuffle=False, num_workers=self.config_dict['num_workers'], pin_memory=False, collate_fn=utils_data.default_collate_with_string)
        return {0: testloader} # note, validtation and train is set to the same one here...
