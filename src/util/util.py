import numpy as np
from torch.autograd import Variable
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm

import IPython

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """

    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap(tp, conf, labels):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, labels = np.array(tp), np.array(conf), np.array(labels)

    # Sort by objectness
    # i = np.argsort(-conf)
    # tp, conf = tp[i], conf[i]

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    n_gt = len((labels != (0, 0, 0, 0)).all(axis=1).nonzero()[0])
    n_p = sum(tp)  # Number of predicted objects

    if (n_p == 0) and (n_gt == 0):
        ap.append(1)
        r.append(1)
        p.append(1)
    elif (n_p == 0) or (n_gt == 0):
        ap.append(0)
        r.append(0)
        p.append(0)
    else:
        # Accumulate FPs and TPs
        fpc = np.cumsum(1 - tp)
        tpc = np.cumsum(tp)

        # Recall
        recall_curve = tpc / (n_gt + 1e-16)
        r.append(tpc[-1] / (n_gt + 1e-16))

        # Precision
        precision_curve = tpc / (tpc + fpc)
        p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

        # AP from recall-precision curve
        ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), np.array(r), np.array(p)

def tensor_imshow(ax, img):
    npimg = img.numpy()
    npimg = np.swapaxes(npimg, 0, 2)
    npimg = np.swapaxes(npimg, 0, 1)

    npimg = np.clip(npimg, 0., 1.)
    ax.imshow(npimg)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off', # labels along the bottom edge are off
        labelsize=1,
        width=1,
        length=5)

def tensor_edgeshow(ax, img):
    npimg = img.numpy()
    npimg = npimg.transpose(1,2,0)
    pos1 = ax.get_position()
    jet = plt.get_cmap('jet')
    im = ax.imshow(npimg)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = plt.colorbar(im, cax=cax, cmap= jet, boundaries=np.arange(0,1.2,.2))
    #cbar.ax.tick_params(labelsize=10)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off', # labels along the bottom edge are off
        labelsize=1,
        width=1,
        length=5)

def tensor_segfireshow(ax, img):
    npimg = img.numpy()
    npimg = npimg[0,:,:]

    pos1 = ax.get_position()
    jet = plt.get_cmap('jet')
    im = ax.imshow(npimg, cmap=jet, vmin=-0.1, vmax=0.15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, cmap= jet, boundaries=np.arange(-0.1,0.15,0.01))
    cbar.ax.tick_params(labelsize=10)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off', # labels along the bottom edge are off
        labelsize=1,
        width=1,
        length=5)

def tensor_heatshow(ax, img):
    npimg = img.numpy()
    npimg = npimg[0,:,:]
    pos1 = ax.get_position()
    jet = plt.get_cmap('jet')
    im = ax.imshow(npimg, cmap=jet, vmin=0., vmax=1.)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, cmap= jet, boundaries=np.arange(0,1.2,.2))
    cbar.ax.tick_params(labelsize=10)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off', # labels along the bottom edge are off
        labelsize=1,
        width=1,
        length=5)

def tensor_mapshow(ax, map, img):
    npmap = map.numpy()
    npmap = npmap[0,:,:]
    npimg = img.numpy()
    npimg = npimg[0, :, :]
    pos1 = ax.get_position()
    # jet = plt.get_cmap('jet')
    # im = ax.imshow(npimg, cmap=jet, vmin=0., vmax=1.)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(im, cax=cax, cmap=jet, boundaries=np.arange(0, 1.2, .2))
    # cbar.ax.tick_params(labelsize=10)

    im2 = ax.imshow(npmap, vmin=0., vmax=1.,)

    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off',  # labels along the bottom edge are off
        labelsize=1,
        width=1,
        length=5)

def bboxFromHeatmap(heatmap_single_tensor, treshold=None):
    numJoints  = heatmap_single_tensor.size()[0]
    heatmapWidth = heatmap_single_tensor.size()[1]
    
    print('bboxFromHeatmap numJoints={}, heatmapWidth={}'.format(numJoints, heatmapWidth))
    #jointPositions_2D = np.zeros( (numJoints,2) )
    
    heatmap_single_tensor_flattened = heatmap_single_tensor.max(0)[0]
    overall_max = heatmap_single_tensor_flattened.max()
    treshold_adaptive = overall_max*treshold
    
    if 0: # per joint approach
        #confidences = np.zeros( (numJoints) )
        #joints_confident = {}
        heatmap_row_max = heatmap_single_tensor.max(1)[0]
        mc, c = torch.max(heatmap_row_max, 2)
        heatmap_col_max = heatmap_single_tensor.max(2)[0]
        mr, r = torch.max(heatmap_col_max, 1)
        
        # flip y axis
        r[:,0,0] = heatmapWidth-r[:,0,0]
        
        r_min = 9999
        r_max = 0
        c_min = 9999
        c_max = 0
        for ji in range(0, numJoints):
            
            if mc[ji,0,0] > treshold_adaptive and mr[ji,0,0] > treshold_adaptive:
                c_min = min(c_min,c[ji,0,0])
                c_max = max(c_max,c[ji,0,0])
                r_min = min(r_min,r[ji,0,0])
                r_max = max(r_max,r[ji,0,0])
                
                
        if 0:
            plt.switch_backend('Qt5Agg')
            fig,ax = plt.subplots(1)
            heatmap = heatmap_single_tensor.numpy()[-3:,:,:].transpose((1, 2, 0))
            ax.imshow(heatmap)
            rect_new = patches.Rectangle([c_min,r_min],
                                      (c_max-c_min),
                                      (r_max-r_min),
                                      linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect_new)
    
            X = [c[ji,0,0] for ji in range(0, numJoints)]
            Y = [r[ji,0,0] for ji in range(0, numJoints)]
            ax.plot(X, Y, '*', color='green', linewidth=1)
    
            plt.axis('equal')
            plt.show()        
    
    if 1: # tresholding complete heatmap approach
        heatmap_row_max = heatmap_single_tensor_flattened.max(1)[0]
        #mr, r = torch.max(heatmap_row_max,2)
        heatmap_col_max = heatmap_single_tensor_flattened.max(2)[0]
        # mc, c = torch.max(heatmap_col_max,1)
         
        overall_max = heatmap_single_tensor_flattened.max()
        treshold_adaptive = overall_max*treshold
         
        r_min = 9999
        r_max = 0
        c_min = 9999
        c_max = 0
        for c in range(0, heatmap_row_max.size()[2]):
            if heatmap_row_max[0,0,c] > treshold_adaptive:
                c_min = min(c_min,c)
                c_max = max(c_max,c)
        for r in range(0, heatmap_col_max.size()[1]):
            if heatmap_col_max[0,r,0] > treshold_adaptive:
                r_min = min(r_min,r)
                r_max = max(r_max,r)

    # note flipping of y axis
    bbox = [c_min, r_min, c_max-c_min, r_max-r_min]
    
    print('bboxFromHeatmap bbox=', bbox)
    
    return bbox

def jointPositionsFromHeatmap(heatmap_single_tensor, treshold=None):
    numJoints  = heatmap_single_tensor.size()[0]
    heatmapWidth = heatmap_single_tensor.size()[1]
    jointPositions_2D = np.zeros( (numJoints,2) )
    confidences = np.zeros( (numJoints) )
    joints_confident = {}
    for j in range(0, numJoints):
        mr, mr_ii = heatmap_single_tensor[j].max(0)

        #print('mr:',mr.size(),'mr_ii',mr_ii.size())
        confidence, mc_i  = mr.max(1)
        mc_i = mc_i[0][0] # from 1x1 variable tensor to scalar, variable is not allowed as index, not differentiable..
        confidence = confidence[0][0]
        #print('mc_i:', mc_i, 'confidence',confidence)
        mr_i = mr_ii[0][mc_i]
        #print('mr_i:', mr_i, 'mc_i',mc_i)
        joint_position = np.array([mc_i,mr_i])
        #print('jp:', joint_position)
        jointPositions_2D[j,:] = joint_position
        confidences[j] = confidence
        #print('confidence',confidence)
        
        if treshold is not None and confidence > treshold:
            joints_confident[j] = joint_position
        #print('l',label[bi,:,j].cpu())
        #print('jp',jointPositions[:,j])
    
    # different joint order
    return jointPositions_2D, confidences, joints_confident

def jointPositionsFromHeatmap_cropRelative_batch(heatmap_tensor, treshold=None):
    heatmap_width = heatmap_tensor.size()[2]
    num_batch_elements = heatmap_tensor.size()[0]
    jointPositions_2D_crop_batch = []
    jointPositions_2D_confidences = []
    for bi in range(0, num_batch_elements):
        jointPositions_2D, confidences, joints_confident = jointPositionsFromHeatmap(heatmap_tensor[bi])
        jointPositions_2D_crop = jointPositions_2D / heatmap_width # normalize to 0..1
        jointPositions_2D_crop_batch.append(jointPositions_2D_crop)
        jointPositions_2D_confidences.append(confidences)
    jointPositions_2D_crop_batch = np.stack(jointPositions_2D_crop_batch, 0)
    jointPositions_2D_confidences = np.stack(jointPositions_2D_confidences, 0)
    return jointPositions_2D_crop_batch, jointPositions_2D_confidences

def tensor_imshow_normalized(ax, img, mean=None, stdDev=None, im_plot_handle=None, x_label=None, clip=False):
    npimg = img.numpy()
    npimg = np.swapaxes(npimg, 0, 2)
    npimg = np.swapaxes(npimg, 0, 1)

    if mean is None:
        mean = (0.0, 0.0, 0.0)
    mean = np.array(mean)
    if stdDev is None:
        stdDev = np.array([1.0, 1.0, 1.0])
    stdDev = np.array(stdDev)

    npimg = npimg * stdDev + mean  # unnormalize
    
    if clip:
        npimg = np.clip(npimg, 0, 1)

    if im_plot_handle is not None:
        im_plot_handle.set_array(npimg)
    else:
        im_plot_handle = ax.imshow(npimg)
        
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off', # labels along the bottom edge are off
        labelsize=1,
        width=1,
        length=5)
    # when plotting 2D keypoints on top, this ensures that it only plots on the image region
    ax.set_ylim([img.size()[1],0])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if x_label is not None:
        plt.xlabel(x_label)   

    return im_plot_handle


def nestedListToCuda(nested_list):
    """
    Converts a nested list of Tensors to CudaTensors
    :param nested_list: A list of Tensors
    :return: A list of CudaTensors
    """
    nested_list_cuda = []
    for el in nested_list:
        if isinstance(el, list):
            nested_list_cuda.append(nestedListToCuda(el))
        elif hasattr(el, 'cuda'):
            nested_list_cuda.append(el.cuda())
        else:
            nested_list_cuda.append(el)
            
    return nested_list_cuda


def nestedListToCPU(nested_list):
    """
    Converts a nested list of Tensors to CPU Tensors
    :param nested_list: A list of Tensors
    :return: A list of CPU Tensors
    """
    nested_list_cuda = []
    for el in nested_list:
        if isinstance(el, list):
            nested_list_cuda.append(nestedListToCPU(el))
        elif hasattr(el, 'cpu'):
            nested_list_cuda.append(el.cpu())
        else:
            nested_list_cuda.append(el)
    return nested_list_cuda


def nestedListToVariable(nested_list):
    """
    Converts a list of Tensors to pytorch Variables.
    :param nested_list: A list of Tensors
    :return: A list of Variables
    """
    nested_list_cuda = []
    for el in nested_list:
        if isinstance(el, list):
            nested_list_cuda.append(nestedListToVariable(el))
        elif isinstance(el, torch.cuda.DoubleTensor) or isinstance(el, torch.DoubleTensor):
            raise Exception('ERROR: Double tensor not supported!!!!!!')
        elif isinstance(el, torch.cuda.FloatTensor) or isinstance(el, torch.FloatTensor):
            nested_list_cuda.append(Variable(el))
        else:
            nested_list_cuda.append(el)
    return nested_list_cuda


def nestedListToDataCPU(nested_list):
    """
    Converts a list of pytorch Variables to plain Tensors.
    :param nested_list: A list of Variables
    :return: A list of Tensors
    """
    nested_list_cuda = []
    for el in nested_list:
        if isinstance(el, list):
            nested_list_cuda.append(nestedListToDataCPU(el))
        elif isinstance(el, Variable):
            nested_list_cuda.append(el.cpu().data)
        else:
            nested_list_cuda.append(el)
    return nested_list_cuda


def nestedDictToCuda(nested_dict):
    """
    Converts a nested list of Tensors to CudaTensors
    :param nested_list: A list of Tensors
    :return: A list of CudaTensors
    """
    nested_list_cuda = {}
    for key, val in nested_dict.items():
        if isinstance(val, (dict)):
            nested_list_cuda[key] = nestedDictToCuda(val)
        else:
            nested_list_cuda[key] = val.cuda()
    return nested_list_cuda


def nestedDictToCPU(nested_list):
    """
    Converts a nested list of Tensors to CPU Tensors
    :param nested_list: A list of Tensors
    :return: A list of CPU Tensors
    """
    nested_list_cuda = {}
    if isinstance(nested_list, (list,)):
        import pdb
        pdb.set_trace()
    for key, val in nested_list.items():
        if isinstance(val, (dict)):
            nested_list_cuda[key] = nestedDictToCPU(val)
        elif isinstance(val, (list,)):
            nested_list_cuda[key] = nestedListToDataCPU(val)
        elif isinstance(val, (str,int,float)):
            nested_list_cuda[key] = val
        else:
            nested_list_cuda[key] = val.cpu()
    return nested_list_cuda


def nestedDictToVariable(nested_list):
    """
    Converts a list of Tensors to pytorch Variables.
    :param nested_list: A list of Tensors
    :return: A list of Variables
    """
    nested_list_cuda = {}
    for key, val in nested_list.items():
        if isinstance(val, (dict,list)):
            nested_list_cuda[key] = nestedDictToVariable(val)
        else:
            nested_list_cuda[key] = Variable(val)
    return nested_list_cuda

def nestedDictToData(nested_list):
    """
    Converts a list of Variables to pytorch Tensors.
    :param nested_list: A list of Variables
    :return: A list of Variables
    """
    nested_list_trans = {}
    for key, val in nested_list.items():
        if isinstance(val, (dict,list)):
            nested_list_trans[key] = nestedDictToData(val)
        else:
            nested_list_trans[key] = val.data
    return nested_list_trans

import importlib.util
def loadModule(module_path_and_name):
    # if contained in module it would be a oneliner:
    # config_dict_module = importlib.import_module(dict_module_name)
    module_child_name = module_path_and_name.split('/')[-1].replace('.py','')
    spec = importlib.util.spec_from_file_location(module_child_name, module_path_and_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module