import sys
from PlottingUtil import util as utils_plt

numJoints = 17
inputDimension = 128

sys.path.insert(0,'./')
sys.path.insert(0, '../')


config_dict = {
    # general params
    'dpi': 1000,
    'root_folder': '../output_ski/',
    'config_class_file': 'configs/config_class_mvc.py',
    'input_types': ['img', 'bg', 'extrinsic_rot', 'extrinsic_rot_inv', 'extrinsic_pos', 'camera_extrinsics', 'inverse_camera_extrinsics', 'intrinsic', 'inverse_intrinsic', 'file_name_info'],  # ,'iteration'
    'output_types': ['img', 'img_crop', 'spatial_transformer', 'img_downscaled', 'blend_mask', 'blend_mask_crop',
                     'similarity_matrix', 'spatial_transformer_img_crop', 'latent_fg', 'latent_3d',
                     'shuffled_pose', 'shuffled_pose_inv', 'shuffled_appearance',
                     'radiance_normalized', 'ST_depth', 'depth_map'],  # ,'smooth_mask',
    'label_types_train': ['img', 'extrinsic_rot', 'extrinsic_rot_inv', 'intrinsic', 'camera_extrinsics'],
    'label_types_test': ['img', 'extrinsic_rot', 'extrinsic_rot_inv', 'intrinsic','camera_extrinsics'],
    'num_workers': 6,

    # problem class parameters
    'bones': utils_plt.bones_h36m,

    #volumetric parameters
    'volume_size': 16,
    'cuboid_side': 10, #in meters

    # opt parameters
    'num_training_iterations': 600000,
    'save_every': 5000,
    'learning_rate': 1e-3,
    'test_every': [5000, 5000],
    'plot_every': 5000,

    # network parameters
    'batch_size_train': 48,
    'batch_size_test': 48,
    'batch_val_J_score': 48,
    'outputDimension_3d': numJoints * 3,
    'outputDimension_2d': 8,

    # loss
    'train_scale_normalized': True,
    'train_crop_relative': False,

    # dataset
    'training_set': 'ski_spoerri',
    'bbox_margin': 0.6,
    'actor_subset': [0, 1, 2, 3],  # all training subjects; testing: 2,3,4; val: 9,11
    'actor_subset_validation': [5], #[4],  # all default training subjects
    'actor_subset_test': [5],  # all default training subjects
    'actor_subset_3Dpose': [1, 9, 11],
    'active_cameras': False,

    'img_mean': (0.485, 0.456, 0.406),
    'img_std': (0.229, 0.224, 0.225),
    'inputDimension': 128,
    #'fullFrameResolution': [1280, 720],
    'fullFrameResolution': [640, 360],
    'mirror_augmentation': False,
    'perspectiveCorrection': False,
    'rotation_augmentation': False,
    'shear_augmentation': False,
    'scale_augmentation': False,
    'seam_scaling': 1.0,
    'useCamBatches': 4, #2
    'every_nth_frame': 1,
    'params_per_box':5,
    'note': 'resL3',

    #multicam offset
    'offset_consistency_type': 'projection', #'projection', 'loss'
    'pairwise_line_distance': 0.1,

    # encode decode
    'useSubjectBatches': 0,
    'latent_bg': 0,
    'latent_fg': 256,
    'latent_3d': 200 * 3,
    'latent_dropout': 0.3,
    'from_latent_hidden_layers': 0,
    'upsampling_bilinear': 'upper',
    'shuffle_fg': False,
    'shuffle_3d': False,
    'shuffle_prob':0.5,
    'feature_scale': 1,
    'num_encoding_layers': 5,
    'loss_prior':1,
    'loss_prior_prob': 0.1,#used to be 0.1
    'loss_prior_prob_before_softmax': 0,
    'loss_prior_radiance_normalized': 0,#0.1,
    'loss_prior_radiance_normalized_binary': 0,
    'loss_prior_fg':0.1,
    'loss_seg_mask_coldstart': 0.25,
    'loss_centralize_seg_mask':0,
    'loss_weight_rgb': 0,
    'loss_weight_rgb_conf': 0,
    'loss_weight_rgb_sampling': 1,
    'loss_weight_rgb_synt': 0,
    'loss_weight_bg': 0,
    'loss_weight_bg_conf': 0,
    'loss_weight_bg_sampling': 0.1,
    'loss_weight_fg_vs_bg':0,
    'loss_weight_bg_voting': 0,
    'loss_weight_gradient': 0,
    'loss_weight_imageNet': 0,
    'loss_weight_imageNet_conf': 0,
    'loss_weight_imageNet_sampling': 2,
    'loss_weight_imageNet_synt': 0,
    'loss_weight_imageNet_bg': 0,
    'loss_weight_imageNet_bg_conf': 0,
    'loss_weight_imageNet_bg_sampling': 0,
    'loss_weight_imageNet_bg_voting': 0,
    'loss_weight_contour': 0,#0.1,#0.1,
    'loss_weight_3d': 0,
    'do_maxpooling': False,
    'encoderType': 'ResNet',
    'implicit_rotation': False,
    'predict_rotation': False,
    'skip_background': False,  # instead use mask multiplication
    'estimate_background': True,
    'reconstruct_type': 'full', # 'bg' if only background is reconstructed, 'fg' if only foreground is estimated, 'full' if everything is estimated
    'receptive_field' : 'medium', #'medium',
    'bg_estimation_opt': 'static', #'static', #normal
    'bbox_random' : False, #Set it to true for None cases in yolo detections
    'bg_recursion' : 1,
    # detection (V2)
    'flatten_batch': True,
    'params_per_box': 6,
    'choose_cell': 'Importance',  # Max
    'only_center_grid': True,
    'offset_range':2.0, #1.0 for 1/8, 8.0 for 1, 16.0 for 2,
    'spatial_transformer': 'GaussBSqSqr',  # True,
    'spatial_transformer_num': 1,  # True,
    'spatial_transformer_bounds': {'border_factor': 0.8, 'min_size': 0.2, 'max_size': 0.80},  # border_scale,min_size,
    'masked_blending': True,
    'scale_mask_max_to_1': True,
    'predict_transformer_depth': True,
    'pass_transformer_depth': False,
    'normalize_mask_density': False,
    'match_crops': True,
    'shuffle_crops': False,
    'offset_crop': True,
    'test_enabled': False,
    'freeze_all':False,
    'check_val_score': True,
}

#config_dict['pretrained_network_path'] = '../pretrained/inpainting_pretrained_model.pth'
config_dict['pretrained_network_path'] = '../pretrained/mvc_pretrained_model.pth'


if config_dict['test_enabled']:
    config_dict['plot_every'] = 1
    config_dict['actor_subset']= [5]
    config_dict['actor_subset_validation'] = [5]
    config_dict['actor_subset_test'] = [5]
    config_dict['useCamBatches'] = 6

config_dict['predict_transformer_depth'] = False
config_dict['output_types'].remove('ST_depth')
config_dict['output_types'].remove('depth_map')
