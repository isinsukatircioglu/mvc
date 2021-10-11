import matplotlib as mpl
from _ast import Or

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys, os, shutil

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')

import numpy as np

import math
import torch
import torch.optim
import torchvision
import imageio
import matplotlib.ticker as ticker

import torchvision.transforms as transforms
import torchvision.models as models_tv
from PIL import Image
from time import gmtime, strftime

from datasets.SkiPTZ import SkiPanTiltDataset_DLT


from datasets import utils as utils_data
from util import util as utils_generic

from PlottingUtil import util as pl_util
from models import custom_losses
from models import resnet_transfer
from models import resnet_VNECT_sep2D
from training import LearningRateScheduler

from datasets import transforms as transforms_aug
from PlottingUtil import util as utils_plt

import training

import IPython
import scipy.ndimage.filters


def plot_iol_wrapper(inputs_raw, labels_raw, outputs_dict, config_dict, mode, iteration, save_path,
                     reconstruct_type='full'):
    # only plot once
    if config_dict['test_enabled'] == False:
        plot_now = ((mode.startswith('training') and
                     ((iteration % config_dict['plot_every'] == 0) or (iteration % 25 == 0 and iteration < 100) or (
                     iteration % 250 == 0 and iteration < 1000))
                     ) or  # any(iteration % item == 0 for item in self.test_every) or
                    not mode.startswith('training') and (iteration == 0 or iteration == 50)
                    or config_dict['plot_every'] == 1
                    # or  mode.startswith('validation_t1')
                    )
    else:
        plot_now = ((mode.startswith('training') and (config_dict['plot_every'] == 1)))
    if not plot_now:
        return

    # determine file name
    if 'frame_index' in labels_raw.keys():
        frame_index = labels_raw['frame_index'][0]
        iteration = int(frame_index)

    def constructFilenameAndCreatePath(mode, iteration):
        img_name = os.path.join(save_path, 'debug_images_{}_{:06d}.jpg'.format(mode, iteration))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return img_name

    # split in two?
    is_hierarchy = labels_raw is not None and (
        isinstance(labels_raw, list) or
        isinstance(labels_raw, dict) and isinstance(labels_raw[next(iter(labels_raw))], dict))  # list or nested dict?

    if is_hierarchy:  # TODO
        subBatchOffset = 0
        # separate plots for varying labels
        if isinstance(labels_raw, list):  # list case
            for subi, labels_sub in enumerate(labels_raw):
                num_labels = len(labels_sub[0])
                outputs_sub = [out[subBatchOffset:subBatchOffset + num_labels] for out in outputs_dict]
                inputs_sub = [inp[subBatchOffset:subBatchOffset + num_labels] for inp in inputs_raw]
                subBatchOffset += num_labels
                modei = mode + "_subBatch" + str(subi)
                # self.plot_io_info_iteration_(inputs_sub, labels_sub, outputs_sub, iteration, save_path, mode+"_subBatch"+str(subi))
                plot_iol(inputs_sub, labels_sub, outputs_sub, config_dict, mode,
                         constructFilenameAndCreatePath(modei, iteration))
        elif isinstance(labels_raw, dict):  # dict case
            for subi, labels_sub in labels_raw.items():
                if isinstance(labels_sub, dict):
                    example = labels_sub[list(labels_sub.keys())[0]]
                elif isinstance(labels_sub, list):
                    example = labels_sub[0]
                else:
                    IPython.embed()
                num_labels = len(example)
                outputs_sub = {key: out[subBatchOffset:subBatchOffset + num_labels] for key, out in
                               outputs_dict.items()}
                inputs_sub = {key: inp[subBatchOffset:subBatchOffset + num_labels] for key, inp in inputs_raw.items()}
                subBatchOffset += num_labels

                modei = mode + "_subBatch" + str(subi)
                plot_iol(inputs_sub, labels_sub, outputs_sub, config_dict, mode,
                         constructFilenameAndCreatePath(modei, iteration))
        else:
            IPython.embed()
            raise ValueError("Expected dict or list")
    else:
        plot_iol(inputs_raw, labels_raw, outputs_dict, config_dict, mode,
                 constructFilenameAndCreatePath(mode, iteration), reconstruct_type)


def accumulate_heat_channels(heat_map_batch):
    plot_heat = heat_map_batch[:, -3:, :, :]
    num_tripels = heat_map_batch.size()[1] // 3
    for i in range(0, num_tripels):
        plot_heat = torch.max(plot_heat, heat_map_batch[:, i * 3:(i + 1) * 3, :, :])
    return plot_heat

def plotAllProposals(ax_img, transformation, width, height,margin=None, box_center=False, box_scale=None, grid_matrix=None, im_out=None, img_name=None, config_dict=None, input_name='', frame_info='', bg_img=None) :
    #remove!!!

    grid_size = 8
    batch_size = transformation.size()[1]
    num_transformers = transformation.size()[0]
    colormap = 'Set1' # cmap = 'hsv'
    clist = ['red','green','blue','orange','cyan','magenta','black','white']
    if config_dict is not None:
        if config_dict['test_enabled']:
            width_aux = bg_img.shape[-1]
            height_aux = bg_img.shape[-2]
            bbox_file = input_name.split('debug')[0]
            bbox_out_folder = bbox_file + 'AllProposals/'
            if not os.path.exists(os.path.dirname(bbox_out_folder)):
                try:
                    os.makedirs(os.path.dirname(bbox_out_folder))
                except:
                    print('Folder cannot be created')
    for i in range(batch_size):
        for j in range(num_transformers):
            affine_matrix = transformation[j, i]
            x_scale = affine_matrix[0, 0].item()
            y_scale = affine_matrix[1, 1].item()
            x_relative = affine_matrix[0, 2].item()
            y_relative = affine_matrix[1, 2].item()
            xwindow = i % 8
            ywindow = i // 8
            #cindex = j*cmap.N // num_transformers
            #color = cmap(cindex)
            color = clist[j]
            if config_dict is not None:
                if config_dict['test_enabled']:
                    plt.figure(1, figsize=(bg_img[i].shape[2]/400, bg_img[i].shape[1]/400), dpi=400)
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
                    plt.gca().yaxis.set_major_locator(ticker.NullLocator())
                    npimg = np.swapaxes(bg_img[i], 0, 2)
                    npimg = np.swapaxes(npimg, 0, 1)
                    npimg = npimg*config_dict['img_std'] + config_dict['img_mean']
                    npimg = np.clip(npimg, 0., 1.)
                    #ax2.imshow(npimg)
                    plt.imshow(npimg)
            for m in range(grid_size):
                for n in range(grid_size):
                    conf = grid_matrix[(ywindow*8)+xwindow, :, n, m]
                    cell_center_x = width * (xwindow + (m+0.5)/grid_size )
                    cell_center_y = height * (ywindow + ((n+0.5)/grid_size))
                    plt.figure(0)
                    ax_img.scatter([cell_center_x], [cell_center_y], color='b', s=0.06 + conf, linewidths=0.)

                    bbox_center_x = width  * (xwindow + (box_center[i, m, n, 0] + 1 - box_scale[i, m, n, 0]) / 2) + (box_scale[i, m, n, 0] * width)/2
                    bbox_center_y = height * (ywindow + (box_center[i, m, n, 1] + 1 - box_scale[i, m, n, 1]) / 2) + (box_scale[i, m, n, 1] * height)/2
                    ax_img.scatter([bbox_center_x.detach().cpu().numpy()], [bbox_center_y.detach().cpu().numpy()], color='g', s=0.006 + conf, linewidths=0.)
                    ax_img.plot([cell_center_x, bbox_center_x.detach().cpu().numpy()], [cell_center_y, bbox_center_y.detach().cpu().numpy()], color='g', linewidth=0.01 + conf)

                    if config_dict is not None:
                        if config_dict['test_enabled']:
                            conf_aux = grid_matrix[i, :, n, m]
                            cell_center_x_aux = width_aux * ((m + 0.5) / grid_size)
                            cell_center_y_aux = height_aux * (((n + 0.5) / grid_size))
                            bbox_center_x_aux = width_aux * ((box_center[i, m, n, 0] + 1 - box_scale[i, m, n, 0]) / 2) + (box_scale[i, m, n, 0] * width_aux) / 2
                            bbox_center_y_aux = height_aux * ((box_center[i, m, n, 1] + 1 - box_scale[i, m, n, 1]) / 2) + (box_scale[i, m, n, 1] * height_aux) / 2
                            plt.figure(1)
                            plt.scatter([cell_center_x_aux], [cell_center_y_aux], color='b', s=1.4 + 20*conf_aux, linewidths=0.)
                            plt.scatter([bbox_center_x_aux.detach().cpu().numpy()], [bbox_center_y_aux.detach().cpu().numpy()],
                                        color='g', s=0.6 + conf_aux, linewidths=0.)
                            plt.plot([cell_center_x_aux, bbox_center_x_aux.detach().cpu().numpy()],
                                     [cell_center_y_aux, bbox_center_y_aux.detach().cpu().numpy()], color='g',
                                     linewidth=0.5 + conf)

            if config_dict is not None:
                if config_dict['test_enabled']:
                    plt.figure(1)
                    plt.tight_layout(pad=0)
                    f_name = bbox_out_folder + 'all_proposals_trial_' + str(int(frame_info[i][2])) + '_cam_' + str(int(frame_info[i][0])) + '_frame_' + str(int(frame_info[i][1])) + '.jpg'
                    if not os.path.exists(f_name):
                        plt.savefig(f_name, bbox_inches='tight', pad_inches = 0)
                    plt.close()
            plt.figure(0)


def plotTransformerBatch(ax_img, transformation, width, height,margin=None, cell_center=None, box_center=False, grid_matrix=None, im_out=None, img_name=None, config_dict=None, input_name='',frame_info='', bg_img=None, synt_img=None, transformation_rpn=None, bbox_viz = True) :
    #remove!!!
    grid_size = 8
    batch_size = transformation.size()[1]
    num_transformers = transformation.size()[0]
    colormap = 'Set1' # cmap = 'hsv'
    clist = ['red','green','blue','orange','cyan','magenta','black','white']

    if config_dict is not None:
        if config_dict['test_enabled']:
            width_aux = bg_img.shape[-1]
            height_aux = bg_img.shape[-2]
            bbox_file = input_name.split('debug')[0]
            bbox_out_folder = bbox_file + 'BoundingBox/'
            synt_out_folder = bbox_file + 'SyntImgBox/'

            if not os.path.exists(os.path.dirname(bbox_out_folder)):
                try:
                    os.makedirs(os.path.dirname(bbox_out_folder))
                except:
                    print('Folder cannot be created')


            if not os.path.exists(os.path.dirname(synt_out_folder)):
                try:
                    os.makedirs(os.path.dirname(synt_out_folder))
                except:
                    print('Folder cannot be created')
    for i in range(batch_size):
        for j in range(num_transformers):
            affine_matrix = transformation[j, i]
            x_scale = affine_matrix[0, 0].item()
            y_scale = affine_matrix[1, 1].item()
            x_relative = affine_matrix[0, 2].item()
            y_relative = affine_matrix[1, 2].item()
            if transformation_rpn is not None:
                affine_matrix_rpn = transformation_rpn[j, i]
                x_scale_rpn = affine_matrix_rpn[0, 0].item()
                y_scale_rpn = affine_matrix_rpn[1, 1].item()
                x_relative_rpn = affine_matrix_rpn[0, 2].item()
                y_relative_rpn = affine_matrix_rpn[1, 2].item()
            xwindow = i % 8
            ywindow = i // 8


            #cindex = j*cmap.N // num_transformers
            #color = cmap(cindex)
            color = clist[j]
            #Margin argument
            if transformation_rpn is not None:
                rect_rpn = patches.Rectangle(
                    (width * (xwindow + (x_relative_rpn + 1 - x_scale_rpn) / 2),
                     height * (ywindow + (y_relative_rpn + 1 - y_scale_rpn) / 2)),
                    x_scale_rpn * width, y_scale_rpn * height,
                    linewidth=0.2, linestyle='dashed', edgecolor='g', facecolor='none')
                ax_img.add_patch(rect_rpn)
            testing = False
            if config_dict is not None:
                if config_dict['test_enabled']:
                    testing = True
                    rect2 = patches.Rectangle(
                        (width_aux * ((x_relative + 1 - x_scale) / 2),
                         height_aux * ((y_relative + 1 - y_scale) / 2)),
                        x_scale * width_aux, y_scale * height_aux,
                        linewidth=1.2, linestyle='dashed', edgecolor=color, facecolor='none')

                    fig3 = plt.figure(2, figsize=(bg_img[i].shape[2] / 400, bg_img[i].shape[1] / 400), dpi=400)
                    ax3 = fig3.add_subplot(111)
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
                    plt.gca().yaxis.set_major_locator(ticker.NullLocator())

                    # plt.cla()
                    npimg = np.swapaxes(bg_img[i], 0, 2)
                    npimg = np.swapaxes(npimg, 0, 1)
                    npimg = npimg * config_dict['img_std'] + config_dict['img_mean']
                    npimg = np.clip(npimg, 0., 1.)
                    ax3.imshow(npimg)
                    ax3.add_patch(rect2)
                    f_name = bbox_out_folder + 'bbox_trial_' + str(int(frame_info[i][2])) + '_cam_' + str(
                        int(frame_info[i][0])) + '_frame_' + str(int(frame_info[i][1])) + '.jpg'
                    if not os.path.isfile(f_name):
                        plt.savefig(f_name)  # when padding is removed the error is fixed
                        plt.close()

                    rect3 = patches.Rectangle(
                        (width_aux * ((x_relative + 1 - x_scale) / 2),
                         height_aux * ((y_relative + 1 - y_scale) / 2)),
                        x_scale * width_aux, y_scale * height_aux,
                        linewidth=1.2, linestyle='dashed', edgecolor=color, facecolor='none')
                    fig4 = plt.figure(3, figsize=(bg_img[i].shape[2] / 400, bg_img[i].shape[1] / 400), dpi=400)
                    ax4 = fig4.add_subplot(111)
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
                    plt.gca().yaxis.set_major_locator(ticker.NullLocator())
                    plt.figure(3)
                    npimg2 = np.swapaxes(synt_img[i], 0, 2)
                    npimg2 = np.swapaxes(npimg2, 0, 1)
                    npimg2 = npimg2 * config_dict['img_std'] + config_dict['img_mean']
                    npimg2 = np.clip(npimg2, 0., 1.)
                    ax4.imshow(npimg2)
                    ax4.add_patch(rect3)
                    f_name2 = synt_out_folder + 'synt_' + str(int(frame_info[i][2])) + '_cam_' + str(
                        int(frame_info[i][0])) + '_frame_' + str(int(frame_info[i][1])) + '.jpg'
                    if not os.path.isfile(f_name2):
                        plt.savefig(f_name2)
                        plt.close()
                    plt.figure(0)
            if not testing:
                if margin is not None:
                    rect = patches.Rectangle(
                        (width * (xwindow + (x_relative + 1 - x_scale) / 2) - max(margin[1][j, i], margin[0][j, i]),
                         height * (ywindow + (y_relative + 1 - y_scale) / 2) - max(margin[1][j, i], margin[0][j, i])) ,
                         x_scale  * width + 2*max(margin[1][j, i], margin[0][j, i]),
                         y_scale * height + 2*max(margin[1][j, i], margin[0][j, i]),
                        linewidth=0.2, linestyle='dashed', edgecolor=color, facecolor='none')

                else:
                    rect = patches.Rectangle(
                        (width  * (xwindow + (x_relative + 1 - x_scale) / 2),
                         height * (ywindow + (y_relative + 1 - y_scale) / 2)),
                        x_scale * width, y_scale * height,
                        linewidth=0.2, linestyle='dashed', edgecolor=color, facecolor='none')

                plt.figure(0)
                if bbox_viz:
                    ax_img.add_patch(rect)


                if cell_center is not None and box_center == True and grid_matrix is not None:
                    for m in range(grid_size):
                        for n in range(grid_size):
                            plt.figure(0)
                            conf = grid_matrix[(ywindow*8)+xwindow, :, n, m]
                            cell_center_x = width * (xwindow + (m+0.5)/grid_size )
                            cell_center_y = height * (ywindow + ((n+0.5)/grid_size))
                            ax_img.scatter([cell_center_x], [cell_center_y], color='b', s=0.03 + conf, linewidths=0.)


                    bbox_center_x = width  * (xwindow + (x_relative + 1 - x_scale) / 2) + (x_scale * width)/2
                    bbox_center_y = height * (ywindow + (y_relative + 1 - y_scale) / 2) + (y_scale * height)/2

                    cell_center_x = width  * (xwindow + (cell_center[0][i] + 1) / 2)
                    cell_center_y = height * (ywindow + (cell_center[1][i] + 1) / 2)

                    plt.figure(0)
                    ax_img.plot([cell_center_x, bbox_center_x], [cell_center_y, bbox_center_y], color='g', linewidth=0.2)

                    plt.figure(0)

def plot_iol(inputs_raw, labels_raw, outputs_dict, config_dict, keyword, image_name, reconstruct_type='full'):
    # handle multi-view
    # if labels_raw is not None and isinstance(labels_raw[0], list):
    #    labels_raw  = labels_raw[0] #in case of multi-view, or other database splits
    # print("labels_raw.keys() = {}".format([type(l) for l in labels_raw]))
    print("labels_raw.keys() = {}, inputs_raw.keys() = {}, outputs_dict.keys() = {}".format(labels_raw.keys(),
                                                                                            inputs_raw.keys(),
                                                                                            outputs_dict.keys()))

    # init figure grid dimensions in an recursive call
    created_sub_plots = 0
    if not hasattr(plot_iol, 'created_sub_plots_last'):
        plot_iol.created_sub_plots_last = {}
    if keyword not in plot_iol.created_sub_plots_last:
        plot_iol.created_sub_plots_last[keyword] = 100  # some defaul value to fit all..
        # call recursively once, to determine number of subplots
        plot_iol(inputs_raw, labels_raw, outputs_dict, config_dict, keyword, image_name)

    num_subplots_columns = 2
    title_font_size = 1
    num_subplots_rows = math.ceil(plot_iol.created_sub_plots_last[keyword] / 2)

    # create figure
    plt.close("all")
    verbose = False
    if verbose:
        plt.switch_backend('Qt5Agg')
    fig = plt.figure(0)
    plt.clf()

    ############### inputs ################
    # display input images
    image_keys = ['img', 'img_crop', 'bg_crop']  # ['img','img_crop','bg_crop','bg']
    for img_key in image_keys:
        if img_key in inputs_raw.keys():
            images_fg = inputs_raw[img_key].cpu().data
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
            ax_img.set_title("Input {}".format(img_key), size=title_font_size, y=0.79)
            grid_t = torchvision.utils.make_grid(images_fg, padding=0)
            if 'frame_info' in labels_raw.keys() and len(images_fg) < 8:
                frame_info = labels_raw['frame_info'].data
                cam_idx_str = ', '.join([str(int(tensor)) for tensor in frame_info[:, 0]])
                global_idx_str = ', '.join([str(int(tensor)) for tensor in frame_info[:, 1]])
                x_label = "cams: {}".format(cam_idx_str)
            else:
                x_label = ""
            utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                                   stdDev=config_dict['img_std'], x_label=x_label, clip=True)
            # do it gain with crop highlighted
            if img_key == 'img':  # ''_crop' not in img_key:
                created_sub_plots += 1
                ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
                ax_img.set_title("Input {}".format(img_key), size=title_font_size, y=0.79)
                grid_gray = grid_t  # torch.mean(grid_t, dim=0, keepdim=True)*0.333+grid_t*0.333+0.334
                utils_generic.tensor_imshow_normalized(ax_img, grid_gray, mean=config_dict['img_mean'],
                                                       stdDev=config_dict['img_std'], x_label=x_label, clip=True)
                height, width = images_fg.shape[2:4]
                if reconstruct_type != 'bg':
                    if 'spatial_transformer_rpn' in outputs_dict:
                        plotTransformerBatch(ax_img, outputs_dict['spatial_transformer'].cpu().data, width, height,
                                             input_name=image_name, bg_img=images_fg.numpy(),
                                             synt_img=outputs_dict['img'].detach().cpu().data.numpy(),
                                             config_dict=config_dict, frame_info=inputs_raw['file_name_info'],
                                             transformation_rpn=outputs_dict['spatial_transformer_rpn'].cpu().data)
                    else:
                        plotTransformerBatch(ax_img, outputs_dict['spatial_transformer'].cpu().data, width, height,
                                             input_name=image_name, bg_img=images_fg.numpy(),
                                             synt_img=outputs_dict['img'].detach().cpu().data.numpy(),
                                             config_dict=config_dict, frame_info=inputs_raw['file_name_info'])

    if 0:
        if 'img_crop' in inputs_raw.keys():
            images_fg = inputs_raw['img_crop'].cpu().data
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
            ax_img.set_title("Input images", size=title_font_size, y=0.79)
            grid_t = torchvision.utils.make_grid(images_fg, padding=0)
            if 'frame_info' in labels_raw.keys() and len(images_fg) < 8:
                frame_info = labels_raw['frame_info'].data
                cam_idx_str = ', '.join([str(int(tensor)) for tensor in frame_info[:, 0]])
                global_idx_str = ', '.join([str(int(tensor)) for tensor in frame_info[:, 1]])
                x_label = "cams: {}".format(cam_idx_str)
            else:
                x_label = ""
            utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                                   stdDev=config_dict['img_std'], x_label=x_label, clip=True)
        # display input images
        if 'bg_crop' in inputs_raw.keys():
            images_bg = inputs_raw['bg_crop'].cpu().data
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
            ax_img.set_title("Background images", size=title_font_size, y=0.79)
            grid_t = torchvision.utils.make_grid(images_bg, padding=0)
            x_label = ""
            utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                                   stdDev=config_dict['img_std'], x_label=x_label, clip=True)

            # difference
            images_diff = torch.abs(images_fg - images_bg)
            images_diff_max, i = torch.max(images_diff, dim=1, keepdim=True)

            images_diff = images_diff_max.expand_as(images_diff)
            images_diff = images_diff / torch.max(images_diff)
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
            ax_img.set_title("Background - foreground", size=title_font_size, y=0.79)
            grid_t = torchvision.utils.make_grid(images_diff, padding=0)
            utils_generic.tensor_imshow_normalized(ax_img, grid_t, x_label=x_label, clip=True)

    # display input heat map
    if '2D_heat' in inputs_raw.keys():
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_img.set_title("2D heat input", size=title_font_size, y=0.79)
        input_heat = accumulate_heat_channels(inputs_raw['2D_heat']).data.cpu()
        utils_generic.tensor_imshow(ax_img, torchvision.utils.make_grid(input_heat, padding=0))

    # display input images
    depth_maps_norm = None
    if 'depth_map' in inputs_raw.keys():
        depth_maps = inputs_raw['depth_map'].cpu().data
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)

        msk_valid = depth_maps != 0
        msk_zero = depth_maps == 0
        if msk_valid.sum() > 0:
            min_v = depth_maps[msk_valid].min()
            max_v = depth_maps[msk_valid].max()
        else:
            min_v = 0
            max_v = 1
        depth_maps_norm = (depth_maps - min_v) / (max_v - min_v) * 1.
        # display background as black for better contrast
        if msk_zero.sum() > 0:
            depth_maps_norm[msk_zero] = 1.

        # print("plot corner value depth_maps[0,0,0,0]={}, depth_maps_norm[0,0,0,0]={}".format(depth_maps[0,0,0,0],depth_maps_norm[0,0,0,0]))


        grid_t = torchvision.utils.make_grid(depth_maps_norm, padding=0)
        utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                               stdDev=config_dict['img_std'], clip=True)
        ax_img.set_title("Input depth map\n(min={:0.4f},\n max={:0.4f})".format(min_v, max_v), size=title_font_size,
                         y=0.79)

    ############### labels_raw ################
    # heatmap label
    if '2D_heat' in labels_raw.keys():
        created_sub_plots += 1
        ax_label = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_label.set_title("2D heat label", size=title_font_size, y=0.79)
        label_heat = labels_raw['2D_heat'].data.cpu()
        numJoints = label_heat.size()[1]
        plot_heat = accumulate_heat_channels(label_heat)
        utils_generic.tensor_imshow(ax_label, torchvision.utils.make_grid(plot_heat[:, :, :, :], padding=0))

    # 2D labelss
    if '2D' in labels_raw.keys():
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_img.set_title("2D labels_raw (crop relative)", size=title_font_size, y=0.79)
        if 'img_crop' in inputs_raw.keys():
            grid_t = torchvision.utils.make_grid(images_fg, padding=0)
            utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                                   stdDev=config_dict['img_std'], clip=True)
        elif depth_maps_norm is not None:
            grid_t = torchvision.utils.make_grid(depth_maps_norm, padding=0)
            utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                                   stdDev=config_dict['img_std'], clip=True)

        label_pose = labels_raw['2D'].data.cpu()
        # outputs_pose_3d = outputs_pose.numpy().reshape(-1,3)
        pl_util.plot_2Dpose_batch(ax_img, label_pose.numpy() * 256, offset_factor=256, bones=config_dict['bones'],
                                  colormap='hsv')

    if '2D_noAug' in labels_raw.keys() and 'img_crop_noAug' in inputs_raw.keys():
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_img.set_title("2D labels_raw (noAug)", size=title_font_size, y=0.79)
        images_noAug = inputs_raw['img_crop_noAug'].cpu().data
        grid_t = torchvision.utils.make_grid(images_noAug, padding=0)
        utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                               stdDev=config_dict['img_std'], clip=True)
        label_pose = labels_raw['2D_noAug'].data.cpu()
        img_shape = images_noAug[0].size()[1]
        pl_util.plot_2Dpose_batch(ax_img, label_pose.numpy() * img_shape, offset_factor=img_shape,
                                  bones=config_dict['bones'], colormap='hsv')

    # plot 3D pose labels_raw
    if any(x in labels_raw.keys() for x in ['3D', '3D_crop_coord']):
        try:
            lable_pose = labels_raw['3D']
        except:
            try:
                lable_pose = labels_raw['3D_crop_coord']
            except:
                lable_pose = None

        if lable_pose is not None:
            created_sub_plots += 1
            ax_3d_l = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots, projection='3d')
            ax_3d_l.set_title("3D pose labels_raw", size=title_font_size, y=0.79)
            if len(lable_pose.shape) > 3:  # flatten sub batches
                s = lable_pose.shape
                lable_pose = lable_pose.view(s[0] * s[1], -1)
            pl_util.plot_3Dpose_batch(ax_3d_l, lable_pose.data.cpu().numpy(), bones=config_dict['bones'], radius=0.01,
                                      colormap='hsv')
            ax_3d_l.invert_zaxis()
            ax_3d_l.grid(False)
            if 1:  # display a rotated version
                created_sub_plots += 1
                ax_3d_l = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots, projection='3d')
                ax_3d_l.set_title("3D pose labels_raw (rotated)", size=title_font_size, y=0.79)
                a = -np.pi / 2
                R = np.array([[np.cos(a), 0, -np.sin(a)],
                              [0, 1, 0],
                              [np.sin(a), 0, np.cos(a)]])
                pose_orig = lable_pose.data.cpu().numpy()
                pose_rotated = pose_orig.reshape(-1, 3) @ R.T
                pl_util.plot_3Dpose_batch(ax_3d_l, pose_rotated.reshape(pose_orig.shape), bones=config_dict['bones'],
                                          radius=0.01, colormap='hsv')
                ax_3d_l.invert_zaxis()
                ax_3d_l.grid(False)

    # draw projection of 3D pose
    if '3D_global' in labels_raw.keys():
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_img.set_title("Projected 3D labels_raw", size=title_font_size, y=0.79)
        if 'img_crop' in inputs_raw.keys():
            grid_t = torchvision.utils.make_grid(images_fg, padding=0)
            utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                                   stdDev=config_dict['img_std'], clip=True)

        # lable_crop_relative = labels_raw[labels_raw.keys().index('3D_crop_coord')].data.cpu()
        lable_3D_glob = labels_raw['3D_global'].data.cpu()
        lable_2D = labels_raw['2D'].data.cpu()
        # bbox = labels_raw[labels_raw.keys().index('bounding_box')].data.cpu().numpy()
        K_crop = labels_raw['intrinsic_crop'].data.cpu().view(-1, 3, 3).numpy()
        for bi in range(0, lable_3D_glob.size()[0]):
            jointPositions_2D = lable_2D[bi].view(-1, 2).numpy() * 256
            jointPositions_3D = lable_3D_glob[bi].view(-1, 3).numpy()
            jointPositions_3D_2D = jointPositions_3D
            jointPositions_3D_2D = np.dot(jointPositions_3D_2D, K_crop[bi].T)
            jointPositions_3D_2D = jointPositions_3D_2D / jointPositions_3D_2D[:, 2, np.newaxis]

            # 2D part matches image coordinates
            jointPositions_3D_crop, jointPositions_3D_weak = transforms_aug.projective_to_crop_relative_np(
                jointPositions_3D, K_crop[bi])
            jointPositions_weak_reconstructed = transforms_aug.crop_relative_to_projective_tvar(
                torch.autograd.Variable(torch.from_numpy(jointPositions_3D_crop)),
                torch.autograd.Variable(torch.from_numpy(K_crop[bi]))).data.numpy()

            # 2D part is weak projected, no perfect match with 2D annotation
            #                jointPositions_3D_crop, jointPositions_3D_weak = transforms_aug.projective_to_crop_relative_weak_np(jointPositions_3D, K_crop[bi])
            #                jointPositions_weak_reconstructed = transforms_aug.crop_relative_weak_to_projective_tvar(torch.autograd.Variable(torch.from_numpy(jointPositions_3D_crop)),
            # jointPositions_3D_crop_2D = jointPositions_3D_crop / jointPositions_3D_crop[:,2,np.newaxis]

            jointPositions_3D_crop += 0.5  # invert normalization/centering for plotting
            # jointPositions_3D_crop_2D += 0.5  # invert normalization/centering for plotting
            by = bi // 8
            bx = bi % 8

            ax_img.plot(256 * bx + jointPositions_2D[:, 0], 256 * by + jointPositions_2D[:, 1], '.', color='green',
                        ms=1)
            # ax_img.plot(256*(bx+jointPositions_3D_crop[:,0]), 256*(by+jointPositions_3D_crop[:,1]), '.', colormap='hsv', ms=3)
            pl_util.plot_2Dpose(ax_img, np.concatenate((256 * (bx + jointPositions_3D_crop[:, 0, np.newaxis]),
                                                        256 * (by + jointPositions_3D_crop[:, 1, np.newaxis])), 1).T,
                                bones=config_dict['bones'], colormap='hsv')
            # ax_img.plot((256*(bx+jointPositions_3D_crop_2D[:,0]), (256*(by+jointPositions_3D_crop_2D[:,1])), '.', color='cyan', ms=3)
            ax_img.plot(256 * (bx + jointPositions_3D_2D[:, 0]), 256 * (by + jointPositions_3D_2D[:, 1]), '.',
                        color='red', ms=1)

    ############### network output ################
    # 3D pose label
    # train_crop_relative = hasattr(self, 'train_crop_relative') and self.train_crop_relative
    if '3D' in outputs_dict.keys():
        if 1:  # not hasattr(self, 'train_crop_relative') or not self.train_crop_relative or not 'intrinsic_crop' in labels_raw.keys():
            outputs_pose = outputs_dict['3D']
            if config_dict['train_scale_normalized'] == 'mean_std':
                m_pose = labels_raw['pose_mean']
                s_pose = labels_raw['pose_std']
                if len(m_pose.shape) > 3:  # flatten sub batches
                    s = m_pose.shape
                    m_pose = m_pose.view(s[0] * s[1], -1)
                    s_pose = s_pose.view(s[0] * s[1], -1)
                    outputs_pose = transforms_aug.denormalize_mean_std_tensor(outputs_pose,
                                                                              {'pose_mean': m_pose, 'pose_std': s_pose})
                else:
                    outputs_pose = transforms_aug.denormalize_mean_std_tensor(outputs_pose, labels_raw)
            outputs_pose = outputs_pose.cpu().data

            created_sub_plots += 1
            ax_3dp_p = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots, projection='3d')
            ax_3dp_p.set_title("3D prediction", size=title_font_size, y=0.79)
            pl_util.plot_3Dpose_batch(ax_3dp_p, outputs_pose.numpy(), bones=config_dict['bones'], radius=0.01,
                                      colormap='hsv')
            ax_3dp_p.invert_zaxis()
            ax_3dp_p.grid(False)
            if 1:  # display a rotated version
                created_sub_plots += 1
                ax_3d_l = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots, projection='3d')
                ax_3d_l.set_title("3D pose prediction (rotated)", size=title_font_size, y=0.79)
                a = -np.pi / 2
                R = np.array([[np.cos(a), 0, -np.sin(a)],
                              [0, 1, 0],
                              [np.sin(a), 0, np.cos(a)]])
                pose_rotated = outputs_pose.numpy().reshape(-1, 3) @ R.T
                pl_util.plot_3Dpose_batch(ax_3d_l, pose_rotated.reshape(outputs_pose.numpy().shape),
                                          bones=config_dict['bones'], radius=0.01, colormap='hsv')
                ax_3d_l.invert_zaxis()
                ax_3d_l.grid(False)

        else:  # display crop relative perspective
            K_crop_tvar = labels_raw['intrinsic_crop'].view(-1, 3, 3)
            K_crop = K_crop_tvar.data.cpu().numpy()

            # print out 3D pose
            outputs_pose_crop = outputs_dict["3D"]
            created_sub_plots += 1
            ax_3dp_p = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots, projection='3d')
            ax_3dp_p.set_title("3D prediction (from crop relative)", size=title_font_size, y=0.79)
            poses_3d = []
            for bi in range(0, outputs_pose_crop.size()[0]):
                pose_3d = transforms_aug.crop_relative_to_projective_tvar(outputs_pose_crop[bi].view(-1, 3),
                                                                          K_crop_tvar[bi]).data.cpu()
                pose_3d_center = pose_3d[utils_plt.root_index_h36m, :]
                pose_3d_centered = pose_3d - pose_3d_center.expand_as(pose_3d)  # root centered coordinates
                poses_3d.append(pose_3d_centered)
            poses_3d = torch.stack(poses_3d)
            pl_util.plot_3Dpose_batch(ax_3dp_p, poses_3d.numpy(), bones=config_dict['bones'], radius=0.01,
                                      colormap='hsv')
            ax_3dp_p.invert_zaxis()
            ax_3dp_p.grid(False)
            # ax_3dp_p.set_aspect(2)

            # overlay 2D part on image
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
            ax_img.set_title("Projected 3D prediction (crop relative)", size=title_font_size, y=0.79)
            if 'img_crop' in inputs_raw.keys():
                grid_t = torchvision.utils.make_grid(images_fg, padding=0)
                utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                                       stdDev=config_dict['img_std'], clip=True)

            # lable_crop_relative = labels_raw[labels_raw.keys().index('3D_crop_coord')].data.cpu()
            # lable_3D_glob = labels_raw[labels_raw.keys().index('3D_global')].data.cpu()
            # bbox = labels_raw[labels_raw.keys().index('bounding_box')].data.cpu().numpy()
            for bi in range(0, outputs_pose_crop.size()[0]):
                jointPositions_3D_crop = outputs_pose_crop[bi].data.cpu().view(-1, 3).numpy().copy()

                # 2D part matches image coordinates
                #                jointPositions_3D_crop, jointPositions_3D_weak = transforms_aug.projective_to_crop_relative_np(jointPositions_3D, K_crop[bi])
                # 2D part is weak projected, no perfect match with 2D annotation
                #                jointPositions_3D_crop, jointPositions_3D_weak = transforms_aug.projective_to_crop_relative_weak_np(jointPositions_3D, K_crop[bi])

                jointPositions_3D_crop += 0.5  # invert normalization/centering for plotting
                by = bi // 8
                bx = bi % 8
                pl_util.plot_2Dpose(ax_img, np.concatenate((256 * (bx + jointPositions_3D_crop[:, 0, np.newaxis]),
                                                            256 * (by + jointPositions_3D_crop[:, 1, np.newaxis])),
                                                           1).T, bones=config_dict['bones'])
                # ax_img.plot(256*(bx+jointPositions_3D_crop[:,0]), 256*(by+jointPositions_3D_crop[:,1]), '.', ms=3)

    if 0 and '2D_heat' in outputs_dict.keys():
        #           output_index = config_dict['output_types'].index("2D_heat")
        output_heat = outputs_dict['2D_heat'].cpu().data
        # utils_plt.plot_2Dpose(ax_img, pose_2d_cat.T, bones=bones_cat, colormap=colormap, color_order=color_order_cat)
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_img.set_title("Predicted 2D labels_raw", size=title_font_size, y=0.79)
        if 'img_crop' in inputs_raw.keys():
            grid_t = torchvision.utils.make_grid(images_fg, padding=0)
            utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                                   stdDev=config_dict['img_std'], clip=True)
        for bi in range(0, output_heat.size()[0]):
            jointPositions_2D, confidences, joints_confident = utils_generic.jointPositionsFromHeatmap(output_heat[bi])
            map_width = output_heat[bi].size()[2]
            jointPositions_2D_crop = jointPositions_2D / map_width  # normalize to 0..1
            #            X = [xy[0] / 32*256 for xy in joints_confident.values()]
            #            Y = [xy[1] / 32*256 for xy in joints_confident.values()]
            by = bi // 8
            bx = bi % 8
            jointPositions_2D_pix = np.concatenate((256 * (bx + jointPositions_2D_crop[:, 0, np.newaxis]),
                                                    256 * (by + jointPositions_2D_crop[:, 1, np.newaxis])), 1)
            utils_plt.plot_2Dpose(ax_img, jointPositions_2D_pix.T, bones=utils_plt.bones_h36m, colormap='hsv')

        created_sub_plots += 1
        ax_label = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_label.set_title("Predicted 2D heatmaps", size=title_font_size, y=0.79)
        plot_heat = accumulate_heat_channels(output_heat)
        utils_generic.tensor_imshow(ax_label, torchvision.utils.make_grid(plot_heat, padding=0))

        # also display backtransformed heatmaps (undoing augmentation and perspective correction)
        if 0:  # 'trans_2d_inv' in labels_raw.keys():
            numJoints = output_heat.size()[1] // 3
            heat_batch = outputs_dict['2D_heat'].cpu().data
            batch_size = heat_batch.size()[0]
            heatmap_width = heat_batch.size()[2]
            output_heats_global = []
            for bi in range(0, batch_size):
                trans_2D_inv = labels_raw['trans_2d_inv'][bi].numpy()
                heatmap_bi = heat_batch[bi].numpy().transpose((1, 2, 0)) + 0.2
                heatmap_bi_trans = self.augmentation_test.apply2DImage(trans_2D_inv, heatmap_bi, [256, 256]).transpose(
                    (2, 0, 1))
                output_heats_global.append(torch.from_numpy(heatmap_bi_trans))
            output_heats_global = torch.stack(output_heats_global)
            output_heats_global_mean = torch.stack([sum(output_heats_global) / batch_size])
            # output_heat = output_heats_global
            ax_label2 = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
            ax_label2.set_title("2D prediction transformed", size=title_font_size, y=0.79)
            utils_generic.tensor_imshow(ax_label2, torchvision.utils.make_grid(
                output_heats_global[:, numJoints - 3:numJoints, :, :], padding=0))
            ax_label3 = fig.add_subplot(4, 3, 9)
            ax_label3.set_title("2D prediction averaged", size=title_font_size, y=0.79)
            utils_generic.tensor_imshow(ax_label3, torchvision.utils.make_grid(
                output_heats_global_mean[:, numJoints - 3:numJoints, :, :], padding=0))

    # generated image
    image_keys = ['img', 'optical_flow_inp', 'img_crop', 'bg_crop', 'optical_flow_bg', 'bg_inp', 'bg_pred_frame', 'bg_gt_frame', 'bg', 'all_offsets', 'grid_matrix', 'cell_map', 'blend_mask',
                  'blend_mask_crop', 'depth_map', 'spatial_transformer_img_crop']
    for img_key in image_keys:

        if img_key in outputs_dict.keys():
            # import pdb
            # pdb.set_trace()
            if img_key == 'all_offsets':
                images_out = outputs_dict['img_downscaled'].cpu().data
            else:
                images_out = outputs_dict[img_key].cpu().data
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
            if img_key == 'bg_inp':
                ax_img.set_title("Input inpainting {}".format(img_key), size=title_font_size, y=0.79)
            else:
                ax_img.set_title("Output {}".format(img_key), size=title_font_size, y=0.79)
            grid_t = torchvision.utils.make_grid(images_out, padding=0)
            if img_key in ['smooth_mask']:  # only a single image in this case, constant
                ax_img.imshow(images_out)
                continue
            elif img_key in ['blend_mask_crop', 'blend_mask', 'smooth_mask']:  # don't denormalize in this case
                utils_generic.tensor_imshow(ax_img, grid_t)
            elif img_key in ['grid_matrix']:
                utils_generic.tensor_heatshow(ax_img, grid_t)
            elif img_key in ['cell_map']:
                grid_matrix_inp = outputs_dict['grid_matrix'].cpu().data
                grid_inp = torchvision.utils.make_grid(grid_matrix_inp, padding=0)
                utils_generic.tensor_mapshow(ax_img, grid_t, grid_inp)
            else:
                utils_generic.tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'],
                                                       stdDev=config_dict['img_std'], clip=True)
            if '_crop' not in img_key and img_key not in ['depth_map']:
                height, width = images_out.shape[2:4]
                if img_key == 'grid_matrix' or img_key == 'cell_map':
                    plotTransformerBatch(ax_img, outputs_dict['spatial_transformer'].cpu().data, width, height, bbox_viz=False)
                else:
                    if img_key != 'all_offsets':
                        plotTransformerBatch(ax_img, outputs_dict['spatial_transformer'].cpu().data, width, height)

                    else:
                        plotAllProposals(ax_img, outputs_dict['spatial_transformer'].cpu().data, width, height,
                                         box_center=outputs_dict['all_offsets'], box_scale=outputs_dict['all_scales'],
                                         grid_matrix=outputs_dict['grid_matrix'].detach().cpu().numpy(), im_out=images_out,
                                         img_name=image_name, config_dict=config_dict, input_name=image_name,
                                         frame_info=inputs_raw['file_name_info'], bg_img=images_fg.numpy())

            if img_key in ['bg']:
                if config_dict['test_enabled'] == True:
                    bg_file = image_name.split('debug')[0]
                    bg_folder = bg_file + 'InpaintingOutput/'
                    if not os.path.exists(os.path.dirname(bg_folder)):
                        try:
                            os.makedirs(os.path.dirname(bg_folder))
                        except:
                            print('Folder cannot be created')
                    for bg in range(outputs_dict[img_key].shape[0]):
                        bg_out = outputs_dict[img_key][bg].cpu().data.numpy().transpose(1, 2, 0)
                        bg_out = bg_out * config_dict['img_std'] + config_dict['img_mean']
                        bg_out = np.clip(bg_out, 0., 1.)
                        f_name = bg_folder + 'inpainting_trial_' + str(
                            int(inputs_raw['file_name_info'][bg][2])) + '_cam_' + str(
                            int(inputs_raw['file_name_info'][bg][0])) + '_frame_' + str(
                            int(inputs_raw['file_name_info'][bg][1])) + '.jpg'
                        if not os.path.isfile(f_name):
                            imageio.imsave(f_name, bg_out)

    key = 'radiance_normalized'
    if key in outputs_dict.keys():
        img_shape = outputs_dict[key].shape[-2:]
        transmittance_normalized = outputs_dict[key].cpu().data.transpose(1, 0).contiguous().view(-1, 1, img_shape[0],
                                                                                                  img_shape[1])
        grid_t = torchvision.utils.make_grid(transmittance_normalized, padding=1, pad_value=0)
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_img.set_title("Output {}".format(key), size=title_font_size, y=0.79)
        utils_generic.tensor_imshow(ax_img, grid_t)

        if config_dict['test_enabled'] == True:

            seg_mask_file = image_name.split('debug')[0]
            seg_out_folder = seg_mask_file + 'SegmentationMasks/'
            if not os.path.exists(os.path.dirname(seg_out_folder)):
                try:
                    os.makedirs(os.path.dirname(seg_out_folder))
                except:
                    print('Folder cannot be created')

            if len(outputs_dict[key].shape) == 4:
                for sm in range(outputs_dict[key].shape[0]):
                    soft_seg_mask = outputs_dict[key].squeeze(1)[sm].cpu().data.numpy()
                    f_name = seg_out_folder + 'soft_seg_mask_trial_' + str(
                        int(inputs_raw['file_name_info'][sm][2])) + '_cam_' + str(
                        int(inputs_raw['file_name_info'][sm][0])) + '_frame_' + str(
                        int(inputs_raw['file_name_info'][sm][1])) + '.jpg'

                    if not os.path.isfile(f_name):
                        imageio.imsave(f_name, soft_seg_mask)

            else:
                for sm in range(outputs_dict[key].shape[1]):
                    soft_seg_mask = outputs_dict[key].squeeze(0).squeeze(1)[sm].cpu().data.numpy()
                    binary_seg_mask_01 = soft_seg_mask.copy()
                    binary_seg_mask_02 = soft_seg_mask.copy()
                    binary_seg_mask_03 = soft_seg_mask.copy()

                    binary_seg_mask_01[binary_seg_mask_01 > 0.1] = 255
                    binary_seg_mask_01[binary_seg_mask_01 <= 0.1] = 0

                    binary_seg_mask_02[binary_seg_mask_02 > 0.2] = 255
                    binary_seg_mask_02[binary_seg_mask_02 <= 0.2] = 0

                    binary_seg_mask_03[binary_seg_mask_03 > 0.3] = 255
                    binary_seg_mask_03[binary_seg_mask_03 <= 0.3] = 0

                    f_name = seg_out_folder + 'soft_seg_mask_trial_' + str(
                        int(inputs_raw['file_name_info'][sm][2])) + '_cam_' + str(
                        int(inputs_raw['file_name_info'][sm][0])) + '_frame_' + str(
                        int(inputs_raw['file_name_info'][sm][1])) + '.jpg'

                    if not os.path.isfile(f_name):
                        imageio.imsave(f_name, soft_seg_mask)


    print("Writing image to {} at dpi={}".format(image_name, config_dict['dpi']))
    if not config_dict['test_enabled']:
        plt.savefig(image_name, dpi=config_dict['dpi'], transparent=True)

    if verbose:
        plt.show()
    plt.close("all")
    plot_iol.created_sub_plots_last[keyword] = created_sub_plots
