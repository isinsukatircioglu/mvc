import os
import logging
import h5py
#import cv2
import numpy as np
import torch
import csv

from training import trainer
import training.utils as utils
from util import nestedListToCuda
from util import nestedListToCPU
from util import nestedListToVariable
from tqdm import tqdm
from time import time
import IPython

import matplotlib.pyplot as plt
from PlottingUtil import util as utils_plt
import sys
sys.path.insert(0, '../')
logger = logging.getLogger(__name__)

def predict(network, dataset_loader, use_cuda, save_folder=None):
    network.train(False)  # Freeze the network, no more grandiant computation
    logger.info("Predicting {} batches".format(len(dataset_loader)))

    for data_idx, data in enumerate(iter(dataset_loader)):
        #print(data_idx)
        inputs, labels = data

        # Convert the inputs and label to Tensors to be able to use them in torch
        if use_cuda:
            inputs = nestedListToCuda(inputs)
        else:
            inputs = nestedListToCPU(inputs)
        inputs = nestedListToVariable(inputs)

        start = time()
        output = network(inputs)
        duration = time() - start

        def setAxisLabels(ax):
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off',
                labelsize=8) # labels along the bottom edge are off
            ax.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off',
                labelsize=8) # labels along the bottom edge are off
            ax.tick_params(
                axis='z',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off',
                labelsize=8) # labels along the bottom edge are off
            ax.grid(False)
            # make the bg white
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            plt.axis('off')
            
        # Save the input image, output pose, and label if given the save folder
        if save_folder:
            batch_size = len(inputs[0].cpu())
            for i,img in enumerate(inputs[0].cpu()):
                full_image_path = os.path.join(save_folder, 'input_{:06d}.png'.format(data_idx*batch_size+i))
                logger.info('Saving input image as {}'.format(full_image_path))
                image_data = np.transpose(torch.squeeze(img).data.cpu().numpy(), (1, 2, 0))
                mean   = (0.485, 0.456, 0.406)
                stdDev = (0.229, 0.224, 0.225)
                image_data = (image_data*stdDev+mean)*255
                image_data = np.array(np.clip(image_data,0,255))/255
                #print(cv2.imwrite(full_image_path, image_data))

                # create plot
                #plt.switch_backend('Qt5Agg')
                plt.close('all')
                fig = plt.figure(0)
                
                # input image
                ax   = fig.add_subplot(221)
                ax.imshow(image_data)
                
                # replace label if not available
                pred_vec = output[0][i].data.cpu().view(1,-1)
                if np.abs(torch.sum(labels[0][i])) > 0.001:
                    gt_vec = labels[0][i].view(1,-1)
                else: # scale by own norm
                    gt_vec = 3 * pred_vec / np.sqrt(torch.sum(pred_vec**2))
                    
                # 3D pose GT
                pose_3d = gt_vec.numpy()
                pose_3d = pose_3d.reshape(-1,3)
                norm_label = np.sqrt(np.sum(pose_3d**2))
                ax   = fig.add_subplot(222, projection='3d')
                utils_plt.plot_3Dpose(ax, pose_3d.T, bones=utils_plt.bones_h36m, radius=0.01, set_limits=False)
                utils_plt.drawDummyPoints(ax)
                ax.invert_zaxis()
                #ax.set_xlabel('pose_w_{:04d}_cam_{:04d}_trial{}.png'.format(frame,camera,trial))
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                setAxisLabels(ax)
                height_label = np.max(pose_3d[:,2]) - np.min(pose_3d[:,2])
        
                # scale normalization
                dot_pose_pose = torch.sum(torch.mul(pred_vec,pred_vec),1)
                dot_pose_gt   = torch.sum(torch.mul(pred_vec,gt_vec),1)
                s_opt = dot_pose_gt / dot_pose_pose
                pose_3d = s_opt[0,0]*pred_vec.numpy()
                
                #norm = np.sqrt(np.sum(pose_3d**2))
                #pose_3d = pose_3d / norm * norm_label
                pose_3d = pose_3d.reshape(-1,3)
                #height_pred =  np.max(pose_3d[:,2]) - np.min(pose_3d[:,2])
                #pose_3d = pose_3d * height_label / height_pred
                ax   = fig.add_subplot(224, projection='3d')
                utils_plt.plot_3Dpose(ax, pose_3d.reshape(-1,3) .T, bones=utils_plt.bones_h36m, radius=0.01, set_limits=False)
                utils_plt.drawDummyPoints(ax)
                ax.invert_zaxis()
                #ax.set_xlabel('pose_w_{:04d}_cam_{:04d}_trial{}.png'.format(frame,camera,trial))
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                setAxisLabels(ax)
                fig.tight_layout()
                plt.subplots_adjust(wspace = 0, hspace=0, left=0,right=1,bottom=0,top=1)
                plt.savefig(full_image_path,  dpi=400, transparent=True)
                #plt.show()
                #break # only show first image of a batch

                


        del inputs # free before next iteration
        # # Save the output if given the save folder.
        # if save_folder:
        #     cv2.imwrite(os.path.join(save_folder, 'input_{}.jpg'.format(data_idx)), np.sum(output[1].data.cpu().numpy(), axis=1))
        yield output, labels, duration

def main(save_path, config_instance, iteration, save_folder, debug_flag=False, save_h5=True, save_csv=False):
    """
    Step 1: Load the data
    Step 2: Load the network with the correct weights
    Step 3: For each d in data get predictions
    Step 4: Save predictions
    """
    save_folder_path = save_folder or os.path.join(save_path, 'predictions')
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # Init
    if debug_flag:
        utils.config_logger(os.path.join(save_folder, 'debug_log.log'))
        config_instance.verbose = True
    else:
        utils.config_logger("/dev/null")
        config_instance.verbose = False

    # Step 1: Load the data
    predloader_dict = config_instance.load_data_prediction()

    # Step 2: Load the network
    logger.info("Creating network...")
    network = config_instance.load_network()
    if debug_flag:
        logger.debug('CUDA is {}'.format(config_instance.cuda))
    if config_instance.cuda:
        network = network.cuda()
    else:
        network = network.cpu()

    trainer_instance = trainer.Trainer(
                                 save_path=save_path,
                                 managed_objects=trainer.managed_objects({"network": network}),
                                 training_step=None,)
    if iteration:
        trainer_instance.load(iteration)
    
    loaded_network_iteration = trainer_instance.iteration

    prediction_path = []
    prediction_dict = {}
    gt_dict = {}
    for pred_loader_key in predloader_dict.keys():
        print(pred_loader_key)
        pred_loader = predloader_dict[pred_loader_key]

        if debug_flag:
            save_image_path = os.path.join(save_folder, 'images_sequence_{}'.format(pred_loader_key))
            if not os.path.exists(save_image_path):
                os.makedirs(save_image_path)
        else:
            save_image_path = None

        pose_list = []
        label_list = []
        with tqdm(total=len(pred_loader)) as pbar:
            full_time = 0
            for prediction_idx, (prediction, label, prediction_time) in enumerate(predict(network, pred_loader, config_instance.cuda, save_folder=save_image_path)):
                full_time += prediction_time
                prediction_numpy = prediction[0].data.cpu().numpy()
                label_numpy = label[0].cpu().numpy()
                for row_pred, row_out in zip(prediction_numpy, label_numpy):
                    pose_list.append(row_pred)
                    label_list.append(row_out)
                pbar.update(1)

        pose_tensor = np.stack(pose_list, axis=0)
        label_tensor = np.stack(label_list, axis=0)

        prediction_dict[pred_loader_key] = pose_tensor
        gt_dict[pred_loader_key] = label_tensor
        save_file_path = None
        if save_h5:
            save_file_path = os.path.join(save_folder_path, 'prediction_{}_model_{}_image_order_test.h5'.format(pred_loader_key, iteration))

            h5_pred_file = h5py.File(save_file_path, 'w')
            h5_pred_file['predictions'] = pose_tensor
            h5_pred_file['gt'] = label_tensor
            h5_pred_file['loaded_network_iteration'] = np.array([loaded_network_iteration])
            h5_pred_file.close()

        if save_csv:
            save_file_path = os.path.join(save_folder_path, 'prediction_{}_model_{}'.format(pred_loader_key, iteration))
            csv_file = open(save_file_path+'.csv', 'w')
            csv_writer = csv.writer(csv_file)

            for count, row_pred in enumerate(pose_list):
                if count % 100 == 0:
                    print("Done {} of {} frames".format(count, len(pose_list)))
                list_prediction = row_pred.tolist()
                csv_writer.writerow(list_prediction)
            csv_file.close()

        print('Prediction took {} sec per image'.format(full_time / len(pred_loader)))
        prediction_path.append(save_file_path)

        if save_file_path:
            print('Results were saved in {}'.format(save_file_path))
            
    return prediction_dict, gt_dict, prediction_path


def argument_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',
                        required=True,
                        help='Path to the folder containing the trained model')
    parser.add_argument('-c', '--config',
                        required=True,
                        help='Path to the file containing the configuration file for the network')
    parser.add_argument('-i', '--iteration',
                        default=None,
                        help='Name of the model that should be loaded. If none it means that the model is loaded by the config already')
    parser.add_argument('-s', '--save_folder',
                        default=None,
                        help='path to the save folder. If none the folder where the model is found will be used')

    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Debug flag')

    return parser.parse_args()


if __name__ == '__main__':
    import importlib

    args = argument_parser()
    folder_path = args.path
    model_name = args.iteration
    config_path = args.config
    save_folder = args.save_folder
    debug_flag = args.debug
    print('Train folder path is : {}'.format(folder_path))
    print('Modle name is : {}'.format(model_name))
    print('Config file is : {}'.format(config_path))

    i = importlib.import_module("configs")
    config_instance_attr = getattr(i, config_path)
    config_instance = config_instance_attr()

    main(folder_path, config_instance, model_name, save_folder, debug_flag=debug_flag)
