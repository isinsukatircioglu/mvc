import os
import sys
import logging
import numpy as np
import time
import torch

from training import trainer
import training.utils as train_utils

from util import nestedListToCuda
from util import nestedListToCPU
from util import nestedListToVariable
from util import nestedDictToCuda
from util import nestedDictToCPU
from util import nestedDictToVariable

import gc

sys.path.insert(0, '../')
from datasets import utils as utils_data

logger = logging.getLogger(__name__)

time_elapsed_loading_sum = 0
time_elapsed_optimizing_sum = 0
time_elapsed_testing_sum = 0

import IPython
import cv2


def training_step(trainloader_endless, network, optimizer, lr_scheduler, loss_function, niter,
                  use_cuda=True, plot_instance=None, save_path=None, stat_print_every=100, optimize_flag=True):
    """
    Called at each training iteration. You can change this function
    in any way you want. This function has to return a dictionary
    with at least the key `loss` and any other key. The content of that
    dictionary will be saved in the Trainer.Summary.

    """
    global time_elapsed_loading_sum, time_elapsed_optimizing_sum, time_elapsed_testing_sum
    if niter % stat_print_every == 0:
        time_total = time_elapsed_loading_sum + time_elapsed_optimizing_sum + time_elapsed_testing_sum + 0.000001
        logger.info('Run time total={} ({} per iteration)'.format(time_total, time_total / (niter + 1)))
        logger.info('Load={:.3f}min, Train={:.3f}min Test={:.3f}min'.format(time_elapsed_loading_sum / 60,
                                                                            time_elapsed_optimizing_sum / 60,
                                                                            time_elapsed_testing_sum / 60))
        logger.info('Load={:.1f}%, Train={:.1f}% Test={:.1f}%'.format(100 * time_elapsed_loading_sum / time_total,
                                                                      100 * time_elapsed_optimizing_sum / time_total,
                                                                      100 * time_elapsed_testing_sum / time_total))

    tic_loading = time.time()

    inputs, labels = next(trainloader_endless)  # niter

    # Convert the inputs and labels to Tensors to be able to use them in torch
    if use_cuda:
        labels = nestedDictToCuda(labels)
        inputs = nestedDictToCuda(inputs)
    else:
        labels = nestedDictToCPU(labels)
        inputs = nestedDictToCPU(inputs)
    labels = nestedDictToVariable(labels)
    inputs = nestedDictToVariable(inputs)

    time_elapsed_loading = time.time() - tic_loading
    time_elapsed_loading_sum += time_elapsed_loading

    tic_training = time.time()

    # Reset the parameter gradients
    optimizer.zero_grad()
    # Change the learning rate according to the scheduler
    optimizer = lr_scheduler.update_lr(optimizer, niter)

    # don't change the network in the first iteration, to make the first validation iteration reflect the pre-trained model
    # optimize_flag = True
    if niter == 0:
        network.train(False)
    else:
        network.train(optimize_flag)
    if optimize_flag == False:
        network.eval()

    # forward + backward + optimize
    outputs = network.forward(inputs, niter)

    losses = loss_function.forward(outputs, labels)

    # handle multiple losses
    if isinstance(losses, list):
        loss = sum(losses)
    else:
        loss = losses

    if optimize_flag:
        loss.backward()
        optimizer.step()

    time_elapsed_optimizing = time.time() - tic_training
    time_elapsed_optimizing_sum += time_elapsed_optimizing

    # debug output
    if plot_instance is not None:
        plot_instance.plot_io_info_iteration(inputs, labels, outputs, niter, save_path, mode='training')

    # Return the loss for the trainer log
    loss_cpu = loss.cpu().data.numpy()[()]
    dict_ret = {"loss": loss_cpu}

    if isinstance(losses, list):
        loss_list_cpu = [l.cpu().data.numpy()[()] for l in losses]
        dict_ret['loss_list'] = loss_list_cpu

    return dict_ret


def validate(dataset_loader, network, loss_function, mode, use_cuda=True, plot_instance=None, save_path=None,
             check=False):
    """
    Validate a the network. You can change this function
    in any way you want. This function should return a dictionary with the
    information you want to be saved into the Trainer.Summary.
    """
    global time_elapsed_testing_sum

    network.train(False)  # Freeze the network, no more grandiant computation

    losses = []
    tic = time.time()
    J_scores = []

    max_th = -1
    max_J = -1
    max_F = -1

    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.55, 0.60]

    def jaccard_score(prediction, label, thresholds):

        """
        pred: NxHxW - soft mask outputs between [0,1]
        label: NxHxW - binary segmentation masks
        """
        scores = []
        # import pdb
        # pdb.set_trace()
        for th in thresholds:
            pred_ = (prediction > th).int()

            inter = (pred_ & label).sum(dim=2).sum(dim=1).float()
            union = (pred_ | label).sum(dim=2).sum(dim=1).float()

            iou = (inter / (union + 1e-6)).mean()

            scores.append(iou.item())

        return np.array(scores)

    def db_eval_iou(annotation, segmentation_, thresholds, void_pixels=None):
        """ Compute region similarity as the Jaccard Index.
        Arguments:
            annotation   (ndarray): binary annotation   map.
            segmentation (ndarray): binary segmentation map.
            void_pixels  (ndarray): optional mask with void pixels
        Return:
            jaccard (float): region similarity
        """
        scores = []
        for th in thresholds:
            segmentation = (segmentation_ > th)

            assert annotation.shape == segmentation.shape, \
                f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
            annotation = annotation.astype(np.bool)
            segmentation = segmentation.astype(np.bool)

            if void_pixels is not None:
                assert annotation.shape == void_pixels.shape, \
                    f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
                void_pixels = void_pixels.astype(np.bool)
            else:
                void_pixels = np.zeros_like(segmentation)

            # Intersection between all sets
            inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
            union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

            j = inters / union
            if j.ndim == 0:
                j = 1 if np.isclose(union, 0) else j
            else:
                j[np.isclose(union, 0)] = 1
            #return j
            scores.append(j)
        return np.array(scores).mean(1)

    def db_eval_boundary(annotation, segmentation_, thresholds, void_pixels=None, bound_th=0.008):
        scores = []
        for th in thresholds:
            segmentation = (segmentation_ > th)
            assert annotation.shape == segmentation.shape
            if void_pixels is not None:
                assert annotation.shape == void_pixels.shape
            if annotation.ndim == 3:
                n_frames = annotation.shape[0]
                f_res = np.zeros(n_frames)
                p_res = np.zeros(n_frames)
                r_res = np.zeros(n_frames)
                for frame_id in range(n_frames):
                    void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
                    f_res[frame_id], p_res[frame_id], r_res[frame_id] = f_measure(segmentation[frame_id, :, :, ],
                                                                                  annotation[frame_id, :, :],
                                                                                  void_pixels_frame, bound_th=bound_th)
            elif annotation.ndim == 2:
                f_res, p_res, r_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
            else:
                raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
            scores.append(f_res)
        return np.array(scores).mean(1)
            #return f_res, p_res, r_res

    def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
        """
        Compute mean,recall and decay from per-frame evaluation.
        Calculates precision/recall for boundaries between foreground_mask and
        gt_mask using morphological operators to speed it up.
        Arguments:
            foreground_mask (ndarray): binary segmentation image.
            gt_mask         (ndarray): binary annotated image.
            void_pixels     (ndarray): optional mask with void pixels
        Returns:
            F (float): boundaries F-measure
        """
        assert np.atleast_3d(foreground_mask).shape[2] == 1
        if void_pixels is not None:
            void_pixels = void_pixels.astype(np.bool)
        else:
            void_pixels = np.zeros_like(foreground_mask).astype(np.bool)

        bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

        # Get the pixel boundaries of both masks
        fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
        gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

        from skimage.morphology import disk

        # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
        fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
        # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
        gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

        # Get the intersection
        gt_match = gt_boundary * fg_dil
        fg_match = fg_boundary * gt_dil

        # Area of the intersection
        n_fg = np.sum(fg_boundary)
        n_gt = np.sum(gt_boundary)

        # % Compute precision and recall
        if n_fg == 0 and n_gt > 0:
            precision = 1
            recall = 0
        elif n_fg > 0 and n_gt == 0:
            precision = 0
            recall = 1
        elif n_fg == 0 and n_gt == 0:
            precision = 1
            recall = 1
        else:
            precision = np.sum(fg_match) / float(n_fg)
            recall = np.sum(gt_match) / float(n_gt)

        # Compute F measure
        if precision + recall == 0:
            F = 0
        else:
            F = 2 * precision * recall / (precision + recall)

        return F, precision, recall

    def _seg2bmap(seg, width=None, height=None):
        """
        From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries.  The boundary pixels are offset by 1/2 pixel towards the
        origin from the actual segment boundary.
        Arguments:
            seg     : Segments labeled from 1..k.
            width	  :	Width of desired bmap  <= seg.shape[1]
            height  :	Height of desired bmap <= seg.shape[0]
        Returns:
            bmap (ndarray):	Binary boundary map.
         David Martin <dmartin@eecs.berkeley.edu>
         January 2003
        """

        seg = seg.astype(np.bool)
        seg[seg > 0] = 1

        assert np.atleast_3d(seg).shape[2] == 1

        width = seg.shape[1] if width is None else width
        height = seg.shape[0] if height is None else height

        h, w = seg.shape[:2]

        ar1 = float(width) / float(height)
        ar2 = float(w) / float(h)

        assert not (
            width > w | height > h | abs(ar1 - ar2) > 0.01
        ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

        e = np.zeros_like(seg)
        s = np.zeros_like(seg)
        se = np.zeros_like(seg)

        e[:, :-1] = seg[:, 1:]
        s[:-1, :] = seg[1:, :]
        se[:-1, :-1] = seg[1:, 1:]

        b = seg ^ e | seg ^ s | seg ^ se
        b[-1, :] = seg[-1, :] ^ e[-1, :]
        b[:, -1] = seg[:, -1] ^ s[:, -1]
        b[-1, -1] = 0

        if w == width and h == height:
            bmap = b
        else:
            bmap = np.zeros((height, width))
            for x in range(w):
                for y in range(h):
                    if b[y, x]:
                        j = 1 + math.floor((y - 1) + height / h)
                        i = 1 + math.floor((x - 1) + width / h)
                        bmap[j, i] = 1

        return bmap

    for i, data in enumerate(dataset_loader):

        if len(data) == 1:
            inputs = {}
            labels = {}
            img_mean = torch.FloatTensor([0.485, 0.456, 0.406]).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(2)
            img_std = torch.FloatTensor([0.229, 0.224, 0.225]).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(2)
            inputs['img'] = data[0]['img']
            inputs['img'] = inputs['img'].permute(0, 3, 1, 2)
            inputs['img'] = inputs['img'].float()
            inputs['img'] /= 255.
            inputs['img'] = (inputs['img'] - img_mean) / img_std
            inputs['file_name_info'] = data[0]['labels']

            labels['img'] = data[0]['img'].clone()
            labels['img'] = labels['img'].permute(0, 3, 1, 2)
            labels['img'] = labels['img'].float()
            labels['img'] /= 255.
            labels['img'] = (labels['img'] - img_mean) / img_std
            if 'optical_flow' in data[0].keys():
                inputs['optical_flow'] = data[0]['optical_flow']
                inputs['optical_flow'] = inputs['optical_flow'].permute(0, 3, 1, 2)
                inputs['optical_flow'] = inputs['optical_flow'].float()
                inputs['optical_flow'] /= 255.
                inputs['optical_flow'] = (inputs['optical_flow'] - img_mean) / img_std
            if 'mask' in data[0].keys():
                labels['annotated'] = data[0]['mask'].permute(0, 3, 1, 2)[:,0,:,:]
                labels['annotated'] = (labels['annotated'] > 200).float()
            #calibration info
            calib_keys = ["intrinsic", "inverse_intrinsic", "extrinsic_rot", "extrinsic_rot_inv", "camera_extrinsics", "inverse_camera_extrinsics", "extrinsic_pos"]
            for c_key in calib_keys:
                if c_key in  data[0].keys():
                    inputs[c_key] = data[0][c_key].clone()
                    inputs[c_key] = inputs[c_key].float()
            
        else:
            inputs, labels = data
        if i % 250 == 0:
            logger.info('Testing validation data at frame {} of {}'.format(i, len(dataset_loader)))

        # Convert the inputs and label to Tensors to be able to use them in torch
        if use_cuda:
            labels = nestedDictToCuda(labels)
            inputs = nestedDictToCuda(inputs)
        else:
            labels = nestedDictToCPU(labels)
            inputs = nestedDictToCPU(inputs)
        labels = nestedDictToVariable(labels)
        inputs = nestedDictToVariable(inputs)

        outputs = network.forward(inputs, -1)

        # Check J Score using 36 annotated frames
        if check and 'validation_t0' in mode:
            out_key = 'blend_mask' if 'img_pred' not in outputs.keys() else 'error'

            real = labels['annotated'].cpu().int() if 'img_pred' not in outputs.keys() else labels['annotated'].int()
            pred = outputs[out_key].squeeze().detach()

            if len(J_scores) == 0:

                #J_scores = jaccard_score(pred, real, thresholds)
                J_scores = db_eval_iou(real.numpy(), pred.numpy(), thresholds)
                F_scores = db_eval_boundary(real.numpy(), pred.numpy(), thresholds, void_pixels=None, bound_th=0.008)

            else:

                #scores_new = jaccard_score(pred, real, thresholds)
                scores_new = db_eval_iou(real.numpy(), pred.numpy(), thresholds)
                scores_F_new = db_eval_boundary(real.numpy(), pred.numpy(), thresholds, void_pixels=None, bound_th=0.008)

                J_scores += scores_new
                F_scores += scores_F_new

        if plot_instance is not None:
            if 'validation_t0' in mode:
                iters = int(mode.split('i')[-1])
                test_every = 5000
                if iters % test_every == 0:
                    plot_instance.plot_io_info_iteration(inputs, labels, outputs, i, save_path, mode=mode)
            else:
                plot_instance.plot_io_info_iteration(inputs, labels, outputs, i, save_path, mode=mode)

        losses_cuda_var = loss_function.forward(outputs, labels)

        if isinstance(losses_cuda_var, list):
            loss_list_cpu = [l.cpu().data.numpy()[()] for l in losses_cuda_var]
            loss_cpu = sum(loss_list_cpu)
        else:
            loss_list_cpu = [losses_cuda_var.cpu().data.numpy()]

        losses.append(loss_list_cpu)
        # free some (GPU) memory before next iteration, otherwise they are only freed once overwritten, increasing the peak memory consumption

        del labels, inputs, outputs, losses_cuda_var
    # del dataiter
    loss_list_avg = [np.mean(np.hstack(l)) for l in zip(*losses)]

    time_elapsed_testing = time.time() - tic
    time_elapsed_testing_sum += time_elapsed_testing
    logger.info('Validation took {} sec.'.format(time_elapsed_testing))

    if check and 'validation_t0' in mode:
        J_scores = J_scores / len(dataset_loader)
        max_th = thresholds[np.argmax(J_scores)]
        max_J = np.max(J_scores)

        F_scores = F_scores / len(dataset_loader)
        #max_F = np.max(F_scores)
        max_F = F_scores[np.argmax(J_scores)]

    dict_ret = {"loss": sum(loss_list_avg),
                'loss_list': loss_list_avg,
                'best_J_th': max_th,
                'best_J': max_J,
                'best_F': max_F,
                }

    return dict_ret


def main(save_path, config_instance, log=False, optimization_flag=True):
    if log:
        train_utils.config_logger(os.path.join(save_path, 'train_log.log'))
    else:
        train_utils.config_logger("/dev/null")

    # Save some hyperparameters
    try:
        logger.info('Batch Size : {}'.format(config_instance.batch_size))
        logger.info('Learning Rate : {}'.format(config_instance.learning_rate))
        logger.info('Weight Decay : {}'.format(config_instance.weight_decay))
    except:
        pass

    trainloader, validloader = config_instance.load_data_train()
    testloader = config_instance.load_data_test()
    trainloader_iterator = utils_data.DataIteratorEndless(trainloader)

    # Print some training information
    logger.info("{} training data points".format(len(trainloader)))
    logger.info("{} validation data points".format(len(validloader)))
    logger.info("{} testing data points".format(len(testloader)))

    logger.info("Creating network...")

    network = config_instance.load_network()
    use_cuda = config_instance.cuda
    if use_cuda:
        network = network.cuda()
    else:
        network = network.cpu()
    loss, loss_test = config_instance.load_loss()

    logger.info("Creating optimizer...")
    optimizer, learning_rate, lr_scheduler = config_instance.loadOptimizer(network)

    logger.info("Creating plotter...")
    plot_instance = config_instance  # Switch to None if no ploting wanted

    if hasattr(config_instance, 'continueTrainingFrom'):
        save_path = config_instance.continueTrainingFrom['path']

    # create output dir
    logger.info("Setting output dir to {}".format(save_path))

    trainer_instance = trainer.Trainer(training_step=lambda niter: training_step(trainloader_iterator, network,
                                                                                 optimizer, lr_scheduler, loss, niter,
                                                                                 use_cuda=use_cuda,
                                                                                 plot_instance=plot_instance,
                                                                                 save_path=save_path,
                                                                                 stat_print_every=
                                                                                 config_instance.test_every[0],
                                                                                 optimize_flag=optimization_flag),
                                       save_every=config_instance.save_every,
                                       save_path=save_path,
                                       managed_objects=trainer.managed_objects({"network": network,
                                                                                "optimizer": optimizer}),
                                       test_functions=[
                                           lambda mode: validate(validloader, network, loss, mode, use_cuda=use_cuda,
                                                                 plot_instance=plot_instance, save_path=save_path,
                                                                 check=config_instance.check_val_score),
                                           lambda mode: validate(testloader, network, loss_test, mode,
                                                                 use_cuda=use_cuda, plot_instance=plot_instance,
                                                                 save_path=save_path)],
                                       test_every=config_instance.test_every,
                                       plot_instance=plot_instance)

    # start off from previous iteration?
    if hasattr(config_instance, 'continueTrainingFrom'):
        logger.info("Continue training from {}".format(save_path))
        trainer_instance.load(config_instance.continueTrainingFrom['iteration'])

    # run training
    trainer_instance.train(config_instance.num_training_iterations, print_every=config_instance.print_every)

    logger.info("{} testing data points".format(len(testloader)))
    val_res = validate(testloader, network, loss_test, 'test', use_cuda=use_cuda, plot_instance=plot_instance,
                       save_path=save_path)
    print(val_res)
