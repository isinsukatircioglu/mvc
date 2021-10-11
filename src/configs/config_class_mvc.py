# import matplotlib.pyplot as plt

import sys, os, shutil

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')

import numpy as np

import math
import torch
import torch.optim

from datasets import utils as utils_data

from models import custom_losses
from models import image_losses

from training import LearningRateScheduler
from models import voting_3dgrid_3dcenter

import training

import datasets.dataset_factory as dataset_factory
import PlottingUtil.plot_dict_input_output_labels as plot_iol
import PlottingUtil.plot_convergence_graphs as plot_error


class Config_class():
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.num_training_iterations = config_dict['num_training_iterations']
        self.check_val_score = config_dict['check_val_score']
        self.test_every = config_dict['test_every']
        self.save_every = config_dict['save_every']
        self.print_every = 500
        self.plot_every = config_dict['plot_every']
        self.cuda = True
        self.verbose = False
        self.num_workers = config_dict['num_workers']  # good choice: 8
        self.reconstruct_type = config_dict['reconstruct_type']
        self.config_dict_test = {k: v for k, v in self.config_dict.items()}
        self.config_dict_subjects = {k: v for k, v in self.config_dict.items()}
        self.config_dict_cams = {k: v for k, v in self.config_dict.items()}
        self.config_dict_test['useSubjectBatches'] = 0
        # self.config_dict_test['useCamBatches'] = self.config_dict['useCamBatches']
        self.config_dict_test['batch_size_test'] = self.config_dict['batch_size_test'] // max(1, self.config_dict[
            'useCamBatches'])

        # new mix
        self.config_dict_cams['batch_size_train'] = self.config_dict['batch_size_train'] // max(1, self.config_dict[
            'useSubjectBatches']) // max(1, self.config_dict['useCamBatches'])
        self.config_dict_cams['batch_size_test'] = self.config_dict['batch_size_test'] // max(1, self.config_dict[
            'useSubjectBatches']) // max(1, self.config_dict['useCamBatches'])

        if 'implicit_rotation' not in self.config_dict.keys():
            self.config_dict['implicit_rotation'] = False
        if 'skip_background' not in self.config_dict.keys():
            self.config_dict['skip_background'] = True
        if 'loss_weight_pose3D' not in self.config_dict.keys():
            self.config_dict['loss_weight_pose3D'] = 0
        if 'n_hidden_to3Dpose' not in self.config_dict.keys():
            self.config_dict['n_hidden_to3Dpose'] = 2

    def plot_io_info_iteration(self, inputs_raw, labels_raw, outputs_raw, iteration, save_path, mode):
        return plot_iol.plot_iol_wrapper(inputs_raw, labels_raw, outputs_raw, self.config_dict, mode, iteration,
                                         save_path, reconstruct_type=self.reconstruct_type)

    def plot_train_info_iteration(self, summary, current_iter, save_path):
        return plot_error.plot_train_or_test_error_graphs(summary, current_iter, save_path, self.config_dict)

    def load_network(self):
        self.output_types = self.config_dict['output_types']

        use_billinear_upsampling = 'upsampling_bilinear' in self.config_dict.keys() and self.config_dict[
            'upsampling_bilinear']
        lower_billinear = 'upsampling_bilinear' in self.config_dict.keys() and self.config_dict[
                                                                                   'upsampling_bilinear'] == 'half'
        upper_billinear = 'upsampling_bilinear' in self.config_dict.keys() and self.config_dict[
                                                                                   'upsampling_bilinear'] == 'upper'

        from_latent_hidden_layers = self.config_dict.get('from_latent_hidden_layers', 0)
        num_encoding_layers = self.config_dict.get('num_encoding_layers', 4)

        num_cameras = 4
        if self.config_dict['active_cameras']:  # for H36M it is set to False
            num_cameras = len(self.config_dict['active_cameras'])

        if lower_billinear:
            use_billinear_upsampling = False
        network_single = voting_3dgrid_3dcenter.unet(dimension_bg=self.config_dict['latent_bg'],
                                                                dimension_fg=self.config_dict['latent_fg'],
                                                                dimension_3d=self.config_dict['latent_3d'],
                                                                feature_scale=self.config_dict['feature_scale'],
                                                                shuffle_fg=self.config_dict['shuffle_fg'],
                                                                shuffle_3d=self.config_dict['shuffle_3d'],
                                                                shuffle_prob=self.config_dict['shuffle_prob'],
                                                                latent_dropout=self.config_dict['latent_dropout'],
                                                                in_resolution=self.config_dict['inputDimension'],
                                                                encoderType=self.config_dict['encoderType'],
                                                                is_deconv=not use_billinear_upsampling,
                                                                upper_billinear=upper_billinear,
                                                                lower_billinear=lower_billinear,
                                                                from_latent_hidden_layers=from_latent_hidden_layers,
                                                                n_hidden_to3Dpose=self.config_dict['n_hidden_to3Dpose'],
                                                                num_encoding_layers=num_encoding_layers,
                                                                output_types=self.output_types,
                                                                subbatch_size=self.config_dict['subCamBathces'] if 'subCamBathces' in self.config_dict.keys() else self.config_dict['useCamBatches'],
                                                                params_per_box=self.config_dict.get('params_per_box', 4),
                                                                implicit_rotation=self.config_dict['implicit_rotation'],
                                                                predict_rotation=self.config_dict.get(
                                                                    'predict_rotation', False),
                                                                spatial_transformer=self.config_dict.get(
                                                                    'spatial_transformer', False),
                                                                spatial_transformer_num=self.config_dict.get(
                                                                    'spatial_transformer_num', 1),
                                                                spatial_transformer_bounds=self.config_dict.get(
                                                                    'spatial_transformer_bounds',
                                                                    {'border_factor': 1, 'min_size': 0.1,
                                                                     'max_size': 1}),
                                                                offset_consistency_type=self.config_dict['offset_consistency_type'],
                                                                skip_background=self.config_dict['skip_background'],
                                                                estimate_background=self.config_dict[
                                                                    'estimate_background'],
                                                                reconstruct_type=self.config_dict['reconstruct_type'],
                                                                receptive_field=self.config_dict['receptive_field'],
                                                                bbox_random=self.config_dict['bbox_random'],
                                                                bg_recursion=self.config_dict['bg_recursion'],
                                                                choose_cell=self.config_dict['choose_cell'],
                                                                only_center_grid=self.config_dict['only_center_grid'],
                                                                offset_range=self.config_dict['offset_range'],
                                                                masked_blending=self.config_dict.get('masked_blending',
                                                                                                     True),
                                                                scale_mask_max_to_1=self.config_dict.get(
                                                                    'scale_mask_max_to_1', True),
                                                                num_cameras=num_cameras,
                                                                predict_transformer_depth=self.config_dict.get(
                                                                    'predict_transformer_depth', False),
                                                                pass_transformer_depth=self.config_dict.get(
                                                                    'pass_transformer_depth', False),
                                                                normalize_mask_density=self.config_dict.get(
                                                                    'normalize_mask_density', False),
                                                                match_crops=self.config_dict.get('match_crops', False),
                                                                shuffle_crops=self.config_dict.get('shuffle_crops',
                                                                                                   False),
                                                                offset_crop=self.config_dict.get('offset_crop', False),
                                                                transductive_training=self.config_dict.get(
                                                                    'transductive_training', []),
                                                                similarity_bandwidth=self.config_dict.get(
                                                                    'similarity_bandwidth', 10),
                                                                disable_detector=self.config_dict.get(
                                                                    'disable_detector', False),
                                                                volume_size=self.config_dict.get('volume_size', 64),
                                                                cuboid_side=self.config_dict['cuboid_side'],
                                                                img_mean=self.config_dict['img_mean'],
                                                                img_std=self.config_dict['img_std'],
                                                                )

        if 'pretrained_network_path' in self.config_dict.keys():  # automatic
            print("Loading weights from self.config_dict['pretrained_network_path']")
            pretrained_network_path = self.config_dict['pretrained_network_path']
            pretrained_states = torch.load(pretrained_network_path)
            training.utils.transfer_partial_weights(pretrained_states, network_single, submodule=0)
            print("Done loading weights from self.config_dict['pretrained_network_path']")

        return network_single

    def loadOptimizer(self, network):
        if network.encoderType == "ResNet":
            params_all_id = list(map(id, network.parameters()))
            params_encoder_id = list(map(id, network.encoder.parameters()))
            params_encoder_finetune_id = [] \
                                         + list(map(id, network.encoder.layer4_reg.parameters())) \
                                         + list(map(id, network.encoder.layer3.parameters())) \
                                         + list(map(id, network.encoder.l4_reg_toVec.parameters())) \
                                         + list(map(id, network.encoder.fc.parameters()))

            params_decoder_id = list(map(id, network.decoder.parameters()))
            params_detector_id = list(map(id, network.detector.parameters()))
            params_bg_unet_id = list(map(id, network.bg_unet.parameters()))

            if self.config_dict['bg_estimation_opt'] == 'slow' or self.config_dict['bg_estimation_opt'] == 'static':
                params_except_encode_decode = [id for id in params_all_id if
                                               id not in params_decoder_id + params_encoder_id + params_bg_unet_id]
                params_except_detect_encode_decode = [id for id in params_all_id if
                                                      id not in params_decoder_id + params_encoder_id + params_detector_id + params_bg_unet_id]

            else:
                params_except_encode_decode = [id for id in params_all_id if
                                               id not in params_decoder_id + params_encoder_id]
                params_except_detect_encode_decode = [id for id in params_all_id if
                                                      id not in params_decoder_id + params_encoder_id + params_detector_id]


            if self.config_dict.get('fix_detector_weight', False):
                params_normal_id = params_except_detect_encode_decode + params_encoder_finetune_id
                if self.config_dict['bg_estimation_opt'] == 'slow':
                    params_slow_id = params_decoder_id + params_bg_unet_id
                else:
                    params_slow_id = params_decoder_id  # used to slow down decoder, less ceivir but still necessary after removal of batch norm
            else:
                params_normal_id = params_except_encode_decode + params_encoder_finetune_id
                start_params_normal_id = params_except_encode_decode
                if self.config_dict.get('bg_estimation_opt') == 'slow':
                    params_slow_id = params_decoder_id + params_bg_unet_id
                else:
                    params_slow_id = params_decoder_id  # used to slow down decoder, less ceivir but still necessary after removal of batch norm
                start_params_slow_id = []  # used to slow down decoder, less ceivir but still necessary after removal of batch norm

            params_normal = [p for p in network.parameters() if id(p) in params_normal_id]
            params_slow = [p for p in network.parameters() if id(p) in params_slow_id]
            params_static_id = [id_p for id_p in params_all_id if not id_p in params_normal_id + params_slow_id]

            # disable gradient computation for static params, saves memory and computation
            for p in network.parameters():
                if id(p) in params_static_id:
                    p.requires_grad = False

            print("Normal learning rate: {} params".format(len(params_normal_id)))
            print("Slow learning rate: {} params".format(len(params_slow)))
            print("Static learning rate: {} params".format(len(params_static_id)))
            print("Total: {} params".format(len(params_all_id)), 'sum of all ',
                  len(params_normal_id) + len(params_slow) + len(params_static_id))

            self.opt_params = [
                {'params': filter(lambda p: p.requires_grad, params_normal),
                 'lr': self.config_dict['learning_rate']},
                {'params': filter(lambda p: p.requires_grad, params_slow),
                 'lr': self.config_dict['learning_rate'] / 5}
            ]
            optimizer = torch.optim.Adam(self.opt_params, lr=self.config_dict['learning_rate'])  # weight_decay=0.0005
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()),
                                         lr=self.config_dict['learning_rate'])
        lr_scheduler = LearningRateScheduler(self.config_dict['learning_rate'], number_batches=9999999,
                                             scheduling_type='fixed')
        lr_scheduler.set()
        return optimizer, self.config_dict['learning_rate'], lr_scheduler

    def load_data_train(self):
        if self.config_dict['loss_weight_pose3D'] > 0:  # True: # HACK
            # mix those for subject 1 and all others
            loaders_train = []
            # loaders_test  = []

            actor_subject_without_pose = [s for s in self.config_dict_cams['actor_subset'] if
                                          s not in self.config_dict_cams['actor_subset_3Dpose']]
            actor_subject_with_pose = [s for s in self.config_dict_cams['actor_subset'] if s in self.config_dict_cams[
                'actor_subset_3Dpose']]  # slightly hacky, to remove test frames
            orig_batch_size = self.config_dict_cams['batch_size_train']

            # determine the right ratio
            ratio = len(actor_subject_with_pose) / len(self.config_dict_cams['actor_subset'])
            batch_size_supervised_float = np.maximum(1, orig_batch_size * ratio)
            batch_size_supervised = int(round(batch_size_supervised_float))
            batch_size_unupervised = orig_batch_size - batch_size_supervised

            print('Semi-supervised ratio=', ratio, 'batch_size_supervised_float', batch_size_supervised_float,
                  'batch_size_supervised=', batch_size_supervised, 'batch_size_unupervised', batch_size_unupervised)
            print('actor_subject_without_pose', actor_subject_without_pose, 'actor_subject_with_pose',
                  actor_subject_with_pose)

            # unsupervised subset
            if len(actor_subject_without_pose) > 0:
                config_dict_selected = self.config_dict_cams.copy()  # supervised
                config_dict_selected['actor_subset'] = actor_subject_without_pose
                config_dict_selected['batch_size_train'] = batch_size_unupervised
                factory = dataset_factory.DatasetFactory()
                trainloader_single, valloader_UNUSED = factory.load_data_train(config_dict_selected)
                loaders_train.append(utils_data.PostFlattenInputSubbatchTensor(trainloader_single))

            # supervised subset
            config_dict_selected = self.config_dict_cams.copy()  # supervised
            config_dict_selected['actor_subset'] = actor_subject_with_pose
            config_dict_selected['batch_size_train'] = batch_size_supervised
            factory = dataset_factory.DatasetFactory()
            trainloader_single, valloader_single = factory.load_data_train(config_dict_selected)
            loaders_train.append(utils_data.PostFlattenInputSubbatchTensor(trainloader_single))
            valloader = utils_data.PostFlattenInputSubbatchTensor(valloader_single)

            # merge them all (collocating items, i.e. also the ones without 3D pose should contain 3D pose)
            trainloader = utils_data.DataLoaderMix(loaders_train, callocate_inputs=True, callocate_labels=True)
            # valloader   = utils_data.DataLoaderMix(loaders_test, callocate_inputs=True, callocate_labels=True)
        else:
            factory = dataset_factory.DatasetFactory()
            trainloader, valloader = factory.load_data_train(self.config_dict_cams)
            if self.config_dict.get('flatten_batch', True):
                trainloader = utils_data.PostFlattenInputSubbatchTensor(trainloader)
                # valloader = utils_data.PostFlattenInputSubbatchTensor(valloader)
            if not self.config_dict['check_val_score']:
                if self.config_dict.get('flatten_batch', True):
                    valloader = utils_data.PostFlattenInputSubbatchTensor(valloader)

        return trainloader, valloader

    def load_data_test(self):
        factory = dataset_factory.DatasetFactory()
        testloader_single = factory.load_data_test(self.config_dict_test)
        if self.config_dict.get('flatten_batch', True):
            testloader_single = utils_data.PostFlattenInputSubbatchTensor(testloader_single)
        # testloader_single = factory.load_data_test(self.config_dict_test)
        return testloader_single

    def load_data_prediction(self):
        factory = dataset_factory.DatasetFactory()
        return factory.load_data_predict(self.config_dict)

    def load_loss(self):
        weight = 1
        print("MPJPE test weight = {}, to normalize different number of joints".format(weight))

        img_key = 'img'
        aux_key = 'bg'
        confidence_key = 'confidence'
        proposal_key = 'proposal'
        input_crop_key = 'input_img_crop'
        crop_key = 'inpainting_crop'
        crop_size_key = 'inpainting_size'

        bg_loss = torch.nn.modules.loss.MSELoss()
        pixel_loss = torch.nn.modules.loss.MSELoss()
        if self.config_dict.get('MAE', False):
            if self.config_dict['MAE'] == 'Huber':
                pixel_loss = custom_losses.HuberLossPair(delta=0.1)  # torch.nn.modules.loss.SmoothL1Loss()
            else:
                pixel_loss = torch.nn.modules.loss.L1Loss()
        # normalized loss to counter illumination changes
        if "box" in self.config_dict['training_set'] or "walk_full" in self.config_dict['training_set']:
            pixel_loss = custom_losses.LossInstanceMeanStdFromLabel(pixel_loss)

        image_grad_loss_train = image_losses.SobelCriterium(criterion=pixel_loss,
                                                            weight=self.config_dict['loss_weight_gradient'],
                                                            key=img_key)
        image_grad_loss_test = image_grad_loss_train

        losses_train = []
        losses_test = []

        if 'img' in self.config_dict['output_types']:
            if self.config_dict['loss_weight_rgb'] > 0:
                image_loss_train = custom_losses.SelectSingleLabel(pixel_loss, key=img_key)
                image_loss_test = image_loss_train
                losses_train.append(image_loss_train)
                losses_test.append(image_loss_test)
            if self.config_dict['loss_weight_rgb_sampling'] > 0:
                image_loss_train_sampling = custom_losses.SelectSingleLabel_Sampling(pixel_loss, key=img_key,
                                                                                     confidence_key=confidence_key,
                                                                                     proposal_key=proposal_key,
                                                                                     train=True)
                image_loss_test_sampling = custom_losses.SelectSingleLabel_Sampling(pixel_loss, key=img_key,
                                                                                    confidence_key=confidence_key,
                                                                                    proposal_key=proposal_key,
                                                                                    train=False)
                losses_train.append(image_loss_train_sampling)
                losses_test.append(image_loss_test_sampling)
            if self.config_dict['loss_weight_bg_sampling'] != 0:
                image_loss_train_bg_sampling = custom_losses.BackgroundPixelLoss_Sampling(bg_loss, key=img_key,
                                                                                          aux_key=aux_key,
                                                                                          confidence_key=confidence_key,
                                                                                          proposal_key=proposal_key,
                                                                                          input_crop_key=input_crop_key,
                                                                                          crop_key=crop_key,
                                                                                          crop_size_key=crop_size_key,
                                                                                          weight=self.config_dict[
                                                                                              'loss_weight_bg_sampling'],
                                                                                          train=True)
                image_loss_test_bg_sampling = custom_losses.BackgroundPixelLoss_Sampling(bg_loss, key=img_key,
                                                                                         aux_key=aux_key,
                                                                                         confidence_key=confidence_key,
                                                                                         proposal_key=proposal_key,
                                                                                         input_crop_key=input_crop_key,
                                                                                         crop_key=crop_key,
                                                                                         crop_size_key=crop_size_key,
                                                                                         weight=self.config_dict[
                                                                                             'loss_weight_bg_sampling'],
                                                                                         train=False)
                losses_train.append(image_loss_train_bg_sampling)
                losses_test.append(image_loss_test_bg_sampling)
            if self.config_dict['loss_weight_fg_vs_bg'] != 0:
                image_loss_train_fg_vs_bg = custom_losses.Bg_vs_Fg_PixelLoss_Sampling(bg_loss, key=img_key,
                                                                                      aux_key=aux_key,
                                                                                      confidence_key=confidence_key,
                                                                                      proposal_key=proposal_key,
                                                                                      input_crop_key=input_crop_key,
                                                                                      crop_key=crop_key,
                                                                                      crop_size_key=crop_size_key,
                                                                                      weight=self.config_dict[
                                                                                          'loss_weight_fg_vs_bg'],
                                                                                      train=True)
                image_loss_test_fg_vs_bg = custom_losses.Bg_vs_Fg_PixelLoss_Sampling(bg_loss, key=img_key,
                                                                                     aux_key=aux_key,
                                                                                     confidence_key=confidence_key,
                                                                                     proposal_key=proposal_key,
                                                                                     input_crop_key=input_crop_key,
                                                                                     crop_key=crop_key,
                                                                                     crop_size_key=crop_size_key,
                                                                                     weight=self.config_dict[
                                                                                         'loss_weight_fg_vs_bg'],
                                                                                     train=False)
                losses_train.append(image_loss_train_fg_vs_bg)
                losses_test.append(image_loss_test_fg_vs_bg)
            if self.config_dict['loss_weight_contour'] != 0:
                edge_loss_train = custom_losses.ContourAlignmentLoss(weight=self.config_dict['loss_weight_contour'],
                                                                     type='DotExp')  # L1_Segx_Segy' # 'L2_SegMagnitude', #type=L2 , 'L2_Magnitude', Exp_Magnitude, L2_Magnitude_Squared
                edge_loss_test = edge_loss_train
                losses_train.append(edge_loss_train)
                losses_test.append(edge_loss_test)
            if self.config_dict['loss_weight_gradient'] > 0:
                losses_train.append(image_grad_loss_train)
                losses_test.append(image_grad_loss_test)
            if self.config_dict['loss_weight_imageNet_sampling'] > 0:
                image_imgNet_loss_train_sampling = image_losses.ImageNetCriterium_Sampling(criterion=pixel_loss,
                                                                                           weight=self.config_dict[
                                                                                               'loss_weight_imageNet_sampling'],
                                                                                           key=img_key,
                                                                                           confidence_key=confidence_key,
                                                                                           proposal_key=proposal_key,
                                                                                           do_maxpooling=self.config_dict.get(
                                                                                               'do_maxpooling', True),
                                                                                           train=True)
                image_imgNet_loss_test_sampling = image_losses.ImageNetCriterium_Sampling(criterion=pixel_loss,
                                                                                          weight=self.config_dict[
                                                                                              'loss_weight_imageNet_sampling'],
                                                                                          key=img_key,
                                                                                          confidence_key=confidence_key,
                                                                                          proposal_key=proposal_key,
                                                                                          do_maxpooling=self.config_dict.get(
                                                                                              'do_maxpooling', True),
                                                                                          train=False)
                losses_train.append(image_imgNet_loss_train_sampling)
                losses_test.append(image_imgNet_loss_test_sampling)
            if self.config_dict['loss_weight_imageNet_bg_sampling'] != 0:
                bg_imgNet_loss_train_sampling = image_losses.ImageNetCriteriumBG_Sampling(criterion=bg_loss,
                                                                                          weight=self.config_dict[
                                                                                              'loss_weight_imageNet_bg_sampling'],
                                                                                          key=img_key, aux_key=aux_key,
                                                                                          confidence_key=confidence_key,
                                                                                          proposal_key=proposal_key,
                                                                                          input_crop_key=input_crop_key,
                                                                                          crop_key=crop_key,
                                                                                          crop_size_key=crop_size_key,
                                                                                          do_maxpooling=self.config_dict.get(
                                                                                              'do_maxpooling', True),
                                                                                          train=True)
                bg_imgNet_loss_test_sampling = image_losses.ImageNetCriteriumBG_Sampling(criterion=bg_loss,
                                                                                         weight=self.config_dict[
                                                                                             'loss_weight_imageNet_bg_sampling'],
                                                                                         key=img_key, aux_key=aux_key,
                                                                                         confidence_key=confidence_key,
                                                                                         proposal_key=proposal_key,
                                                                                         input_crop_key=input_crop_key,
                                                                                         crop_key=crop_key,
                                                                                         crop_size_key=crop_size_key,
                                                                                         do_maxpooling=self.config_dict.get(
                                                                                             'do_maxpooling', True),
                                                                                         train=False)
                losses_train.append(bg_imgNet_loss_train_sampling)
                losses_test.append(bg_imgNet_loss_test_sampling)


        if self.config_dict['spatial_transformer']:
            losses_train.append(
                custom_losses.AffineCropPositionPrior(self.config_dict['fullFrameResolution'], weight=1.0))

        if self.config_dict['loss_prior'] != 0:
            losses_train.append(custom_losses.AffineCropLocalPrior(self.config_dict['fullFrameResolution'],
                                                                   weight=self.config_dict['loss_prior']))

        if self.config_dict['loss_prior_prob'] != 0:
            losses_train.append(
                custom_losses.ConfidencePrior(key='confidence', weight=self.config_dict['loss_prior_prob']))

        if self.config_dict['loss_prior_prob_before_softmax'] != 0:
            losses_train.append(custom_losses.ConfidenceBeforeSoftmaxPrior(key='confidence_before_softmax',
                                                                           weight=self.config_dict[
                                                                               'loss_prior_prob_before_softmax']))

        if self.config_dict['loss_prior_radiance_normalized'] != 0:
            losses_train.append(custom_losses.RadianceNormalizedPrior(key='radiance_normalized',
                                                                      weight=self.config_dict[
                                                                          'loss_prior_radiance_normalized']))

        if self.config_dict['loss_prior_radiance_normalized_binary'] != 0:
            losses_train.append(custom_losses.RadianceNormalizedPriorBinary(key='radiance_normalized',
                                                                            weight=self.config_dict[
                                                                                'loss_prior_radiance_normalized_binary']))

        if self.config_dict['loss_seg_mask_coldstart'] != 0:
            losses_train.append(custom_losses.VShapeSegMaskFire(weight=self.config_dict['loss_seg_mask_coldstart'],
                                                                percent=0.001))  # USE THIS

        if self.config_dict['loss_prior_fg'] != 0:
            losses_train.append(
                custom_losses.VShapeSegMaskFire(key='fg', weight=self.config_dict['loss_prior_fg'], percent=0.001))

        if self.config_dict['loss_centralize_seg_mask'] != 0:
            losses_train.append(custom_losses.CentralizeSegMask(weight=self.config_dict['loss_centralize_seg_mask']))

        if self.config_dict['offset_consistency_type'] == 'loss' and self.config_dict['pairwise_line_distance'] != 0:
            losses_train.append(custom_losses.MinPairwiseLineDistance(weight= self.config_dict['pairwise_line_distance']))

        print(losses_train)
        print(losses_test)

        # Mask prior loss
        if 0:
            losses_train.append(custom_losses.MaskPrior(weight=0.1))

        loss_train = custom_losses.PreApplyCriterionListDict(losses_train, sum_losses=False)
        loss_test = custom_losses.PreApplyCriterionListDict(losses_test, sum_losses=False)

        # annotation and pred is organized as a list, to facilitate multiple output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test

    def get_parameter_description(self):  # , config_dict):
        base_folder = self.config_dict['root_folder']
        folder = base_folder + 'experiment_'
        #folder = folder.replace(' ', '').replace('./', '[DOT_SHLASH]').replace('.', 'o').replace('[DOT_SHLASH]','./').replace(',', '_')
        return folder