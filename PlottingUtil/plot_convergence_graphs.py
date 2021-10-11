import matplotlib as mpl
from _ast import Or
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys, os, shutil
sys.path.insert(0,'./')
sys.path.insert(0,'../')
sys.path.insert(0,'../../')

import numpy as np

import math
import torch
import torch.optim
import torchvision
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

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib.ticker import FormatStrFormatter

import training

import IPython
import scipy.ndimage.filters

history_length = 50000

def plot_train_or_test_error_graphs(summary, current_iter, save_path, config_dict):
    """
    This function is called at every iteration and is given the current summary.
    Any plots that use the training information like the loss should go in this function.
    :param summary: The current summary
    :param current_iter: The current iteration
    :param save_path: The root forlder of the current training
    """
    plot_testing_now = any(current_iter % item == 0 for item in config_dict['test_every'])

    plot_training_now = plot_testing_now or (current_iter % 500 == 0 and current_iter<5000)

    if plot_training_now or plot_testing_now:
        plot_path = os.path.join(save_path, 'plots')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
            
    if plot_training_now:
        plot_file_name = os.path.join(plot_path, 'train_val_loss_{:06d}.png'.format(current_iter))
        plot_train_info_iteration(summary, current_iter, plot_file_name, config_dict)
    if plot_testing_now:
        plot_file_name = os.path.join(plot_path, 'train_val_loss_list_{:06d}.png'.format(current_iter))
        plot_test_info_iteration(summary, current_iter, plot_file_name, config_dict)
    
def plot_train_info_iteration(summary, current_iter, plot_file_name, config_dict):
    training_loss = summary.get("training.loss")
    training_loss_x = training_loss[0]
    selected_values = training_loss_x > training_loss_x[-1]-history_length
    #training_loss_y = list(map(lambda x: math.log(abs(x)), training_loss[1].tolist()))
    training_loss_y = list(map(lambda x: (abs(x)), training_loss[1].tolist()))


    # overall training loss
    fig = plt.figure(0)
    plt.clf()
    num_plots = len(config_dict["test_every"])+1
    ax_train   = fig.add_subplot(num_plots,1,1)
    ax_train.set_ylabel('training set')
    
    loss_min = np.min(training_loss[1])
    ax_train.plot(training_loss_x[selected_values], np.array(training_loss_y)[selected_values], label="training loss, min={:0.5f}".format(loss_min))
    
    # individual training losses (if multiview loss present)
    key = "training.loss_list"
    if summary.has_tag(key):
        val_losses = summary.get(key)
        val_loss_x = val_losses[0]
        selected_values = val_loss_x > training_loss_x[-1]-history_length
        for li, test_l in enumerate(val_losses[1].T.tolist()):
            #val_loss_y =  list(map(lambda x: math.log(abs(x)), test_l))
            val_loss_y =  list(map(lambda x: (abs(x)), test_l))
            loss_min = np.min(test_l)
            loss_mean_local= np.mean(np.array(test_l)[selected_values])
            # filter the training error, it is computed over a short time, is too noisy without smoothiing
            val_loss_y = scipy.ndimage.filters.gaussian_filter1d(np.array(val_loss_y), sigma=8, axis=0, mode='nearest')

            ax_train.plot(val_loss_x[selected_values], np.array(val_loss_y)[selected_values], label="training loss, li={}, min={:0.5g}, mean={:0.2g}".format(li, loss_min, loss_mean_local))
    ax_train.legend(fontsize=5)

    # validation loss
    for ti,v in enumerate(config_dict["test_every"]):
        val_loss = summary.get("validation.t{}.loss".format(ti))
        val_loss_x = val_loss[0]
        selected_values = val_loss_x > training_loss_x[-1]-history_length
        ax_test = fig.add_subplot(num_plots,1,2+ti)
        ax_test.set_ylabel('validation set')
        val_loss_y = val_loss[1]
        loss_min = np.min(val_loss[1])
        loss_mean_local = np.mean(val_loss[1][selected_values])
        ax_test.plot(val_loss_x[selected_values], val_loss_y[selected_values], label="validation loss {}, min={:0.5f}, mean={:0.2g}".format(ti, loss_min, loss_mean_local))
        ax_test.legend()
        ax_test.legend(fontsize=5)
       
#                 key = "validation.t{}.loss_list".format(ti)
#                 if key in summary:
#                     val_losses = summary.get(key)
#                     val_loss_x = val_losses[0]
#                     for li, test_l in enumerate(val_losses[1].T.tolist()):
#                         val_loss_y = test_l #list(map(lambda x: math.log(x), val_l))
#                         loss_min = np.min(test_l)
#                         ax_train.plot(val_loss_x.tolist(), val_loss_y, label="validation loss={}, li={}, min={:0.5f}".format(ti, li, loss_min))

    
    # save training and validation loss
    print("Saving loss image as {}".format(plot_file_name))
    plt.savefig(plot_file_name, dpi=900)
    plt.clf()    

def plot_test_info_iteration(summary, current_iter, plot_file_name, config_dict):
    # Separately split individual validation losses, if in multi loss case
    fig = plt.figure(0)
    plt.clf()
    num_plots = len(config_dict["test_every"]) 
    for ti,v in enumerate(config_dict["test_every"]):
        key = "validation.t{}.loss_list".format(ti)
        if summary.has_tag(key):
            val_losses = summary.get(key)
            val_loss_x = val_losses[0]
            selected_values = val_loss_x > val_loss_x[-1]-history_length
            ax_test = host_subplot(num_plots,1,1+ti, axes_class=AA.Axes)
            plt.subplots_adjust(right=0.75)
            pars = []
            for li, val_l in enumerate(val_losses[1].T.tolist()):
                val_loss_y = np.array(val_l) #list(map(lambda x: math.log(x), val_l))
                
                # draw in the same plot, but at different scales
                if li>0:
                    par = ax_test.twinx()
                    offset = 60*(li-1)
                    new_fixed_axis = par.get_grid_helper().new_fixed_axis
                    par.axis["right"] = new_fixed_axis(loc="right", axes=par, offset=(offset, 0))
                    par.axis["right"].toggle(all=True) # show all ticks etc.
                    par.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                else:
                    par = ax_test

                loss_min = np.min(val_l)
                val_loss_y_selected = val_loss_y[selected_values]
                loss_seg_min = np.min(val_loss_y_selected)
                loss_seg_max = np.max(val_loss_y_selected)
                description_short = "validation loss={}, li={}".format(ti, li)
                description_long = "validation loss={}, li={}, min={:0.5f}".format(ti, li, loss_min)
                handle = par.plot(val_loss_x[selected_values], val_loss_y_selected, label=description_long)
                par.set_ylabel(description_short)
                par.set_ylim(loss_seg_min, loss_seg_max+0.000001)
                pars.append([par,handle])
            ax_test.legend()
            for par,handle in pars:
                if par != ax_test:
                    par.axis["right"].label.set_color(handle[0].get_color())
    
    print("Saving loss image as {}".format(plot_file_name))
    plt.savefig(plot_file_name, dpi=900)
    plt.clf()    
