import numpy as np
import torch
import torch.utils.data
import torch.utils.data.dataloader
import IPython
import pdb
import collections

#import cv2
import itertools
import queue
#from matplotlib.sphinxext.plot_directive import out_of_date
string_classes = (str, bytes)

def gaussian(mu_x, mu_y, sig, size):
    xy = np.indices(size)
    x = xy[0,:,:]
    y = xy[1,:,:]

    psf  = np.exp(-((x-mu_x)**2/(2*sig**2) + (y-mu_y)**2/(2*sig**2)))
    return psf

def construct_heatmap(points_crop, points_visibility, map_width, sigma, heatmaps_init=None, sum_channel=True):
    """Construct a heatmap, possibly starting from an existing stack (for multi-person) and adding an optional sum channel"""
    numJoints = points_crop.shape[0]
    numChannels = numJoints
    if sum_channel:
        numChannels += 1
    if heatmaps_init is None:
        heatmaps = np.zeros((map_width, map_width, numChannels),dtype='float32')
    else:
        heatmaps = heatmaps_init
    for pi in range(0,numJoints):
        if points_visibility==True or points_visibility[pi]:
            heatmap = gaussian(points_crop[pi,1]*map_width, points_crop[pi,0]*map_width, sigma, heatmaps.shape[0:2])
            heatmaps[:,:,pi] = np.maximum(heatmap, heatmaps[:,:,pi])
    if sum_channel:
        # combined heatmap for debugging purposes, also add as target, might help to supervise
        averageMap = np.sum(heatmaps[:,:,:numJoints], 2) # sum everything except the previous average
        averageMap = np.clip(averageMap, 0., 1.)
        heatmaps[:,:,numJoints] = averageMap
    return heatmaps

# def global2crop(pose_3d_center, bbox_width, K_crop):
#     # intrinsics
#     focal_avg = (K_crop[0,0] + K_crop[1,1])/2
#     # cam->pix, weak_project, crop_zoom
#     T = np.eye(3,3) * focal_avg / pose_3d_center[2] / bbox_width

class MeanAndStd(object):
    def __init__(self):
        self.number_elems = 0
        self.var = None
        self.mean = None

        self.init_shape_done = False

    def add_data(self, data):
        if not self.init_shape_done:
            self._init_shape(data)
        self.number_elems += 1
        previous_mean = self.mean
        self.mean += (data - previous_mean) / self.number_elems
        self.var += (data - self.mean) * (data - previous_mean)

    def get_values(self):
        var = self.var / self.number_elems
        return_dict = {'mean': self.mean,
                       'std': np.sqrt(var)}
        return return_dict

    def _init_shape(self, data):
        self.mean = np.zeros(data.shape, dtype='float32')
        self.var = np.zeros(data.shape, dtype='float32')


def heatmap_as_channels(heatmap_array):
    """
    Takes the concatenated heatmaps and returns an array of 15 channels. The initial data is in one channel.
    :param heatmap_array: Concatenated heatmaps along the x axis
    :return: Array where the heatmaps are as channnels
    """
    # Assume that the initial data is (y, x)
    # Get the size of a heatmap for the original data
    heatmap_width = heatmap_array.shape[1] // 15
    # Check that the size is correct by making sure that no pixels are missing
    assert heatmap_array.shape[1] % heatmap_width == 0, 'Initial array size is not a multiple of the image size.'

    list_of_heatmaps = np.hsplit(heatmap_array, 15)
    final_array = np.zeros((heatmap_array.shape[0], heatmap_width, 15))
    for i, single_heatmap in enumerate(list_of_heatmaps):
        final_array[:, :, i] = single_heatmap
    return final_array


def crop_heatmap(heatmap, bounding_box_norm, square=True, pad=False):
    """
    Crops the given heatmap given the coordinates of the bounding box. The bounding box is in the order
    [x, y, width, height]. The values represent pixels.

    :param heatmap: The heatmap to crop
    :param bounding_box_norm: The bounding box determining where to crop. Format [x, y, width, height]. Values are between 0 and 1
    :return: The cropped region of the heatmap
    """
    if pad:
        square = True
    width = heatmap.shape[1]
    height = heatmap.shape[0]

    bounding_box = [bounding_box_norm[0]*width, bounding_box_norm[1]*height, bounding_box_norm[2]*width, bounding_box_norm[3]*height]
    bounding_box = [int(elem) for elem in bounding_box]

    if square and not pad:
        if bounding_box[3] > bounding_box[2]:
            diff = bounding_box[3] - bounding_box[2]
            bounding_box[2] = bounding_box[3]
            bounding_box[0] -= diff // 2
        elif bounding_box[2] > bounding_box[3]:
            bounding_box[3] = bounding_box[2]
            diff = bounding_box[2] - bounding_box[3]
            bounding_box[1] -= diff // 2

    crop_shape = (bounding_box[3], bounding_box[2], heatmap.shape[2])
    crop_data = np.zeros(tuple(crop_shape), dtype=heatmap.dtype)

    if not (bounding_box[0] > width or bounding_box[0] + bounding_box[2] < 0 or
            bounding_box[1] > height or bounding_box[1] + bounding_box[3] < 0):
        img_min_x = max(0, bounding_box[0])
        img_max_x = min(bounding_box[0] + bounding_box[2], width)
        img_min_y = max(0, bounding_box[1])
        img_max_y = min(bounding_box[1] + bounding_box[3], height)

        crop_min_x = img_min_x-bounding_box[0]
        crop_max_x = img_max_x-bounding_box[0]
        crop_min_y = img_min_y-bounding_box[1]
        crop_max_y = img_max_y-bounding_box[1]
    else:
        raise Exception

    crop_data[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :] = heatmap[img_min_y:img_max_y, img_min_x:img_max_x, :]

    if pad:
        # Check that the crop shape is not already square
        if crop_shape[0] == crop_shape[1]:
            return crop_data

        # Create the new bounding box
        pad_dir = None
        diff = 0
        if bounding_box[3] > bounding_box[2]:
            pad_dir = 0
            diff = bounding_box[3] - bounding_box[2]
            bounding_box[2] = bounding_box[3]
            bounding_box[0] -= diff // 2
        elif bounding_box[2] > bounding_box[3]:
            pad_dir = 1
            diff = bounding_box[2] - bounding_box[3]
            bounding_box[3] = bounding_box[2]
            bounding_box[1] -= diff // 2


        new_crop_data = np.zeros((bounding_box[3], bounding_box[2], heatmap.shape[2]), dtype=heatmap.dtype)
        if pad_dir == 0:
            new_crop_data[:, diff//2:diff//2+crop_shape[1], :] = crop_data
        elif pad_dir == 1:
            new_crop_data[diff//2:diff//2+crop_shape[0], :, :] = crop_data

        return new_crop_data

    return crop_data


def read_heatmap(heatmap_path):
    """
    Reads the heatmap file given the path
    :return: The heatmap data in the range 255
    """
    # Heatmaps are stored in an image, we read it as grayscale
    # Values are between 0 and 255
    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    return heatmap

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

_use_shared_memory = False

def default_collate_with_string(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        #print("IN","torch.is_tensor(batch[0])")
        #IPython.embed()
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        #print("batch:",[e.numpy().shape for e in batch])
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        #print("IN", "type(batch[0]).__module__ == 'numpy'")
        #IPython.embed()
        if type(elem).__name__ == 'ndarray':
            if elem.dtype.kind in {'U', 'S'}:
                return np.stack(batch, 0)
            else:
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate_with_string([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate_with_string(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def cat_collate(list_of_dicts):
    #
    "Puts each data field into a tensor with outer dimension batch size"
    if len(list_of_dicts) == 0:
        raise ValueError("List of dicts is empty")

    assert isinstance(list_of_dicts, collections.Sequence)

    #batch_elements_paired = tuple(zip(*batch_list))

    batch_concatenated = {}
    for key in list_of_dicts[0]:
        example = list_of_dicts[0][key]
        
        value_list = [dict_instance[key] for dict_instance in list_of_dicts]

        # merge two sets together, filling entries with zeroes if data is not available
        if torch.is_tensor(example):
            #labels_filled = [l if type(l) != list or l != [] else torch.zeros(example.size()) for l in label_index]
            batch_concatenated[key] = torch.cat(value_list, 0)
        elif type(example).__module__ == 'numpy' and type(value_list[0]).__name__ == 'ndarray':
            if example.dtype.kind in {'U', 'S'}:
                batch_concatenated[key] = np.concatenate(value_list,0)
            else:
                batch_concatenated[key] = torch.cat([torch.from_numpy(b) for b in value_list], 0)
        elif isinstance(example, int):
            batch_concatenated[key] = torch.LongTensor(value_list)
        elif isinstance(example, float):
            batch_concatenated[key] = torch.FloatTensor(value_list)
        elif isinstance(example, collections.Sequence) and isinstance(example[0][0], string_classes):
            #mats_transposed = [np.array(l).T for l in label_index]
            concatenated = np.concatenate(value_list,0)
            batch_concatenated[key] = concatenated
        else:
            raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}".format(type(value_list[0]))))

    return batch_concatenated

class DataIteratorEndless(object):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(iterable)
        self.new_epoch = False

    class ListEnd:
        pass

    def __next__(self):
        self.new_epoch = False
        end = self.ListEnd()
        value = next(self.iterator, end)
        if value == end:
            self.new_epoch = True
            del self.iterator
            self.iterator = iter(self.iterable)
            value = self.iterator.__next__()
        return value

    def __len__(self):
        self.iterator.__len__()

    def __del__(self):
        del self.iterator

# TODO: use alternative collate function
class PostFlattenInputSubbatchTensorIter(object):
    def __init__(self, data_loader_orig):
        self.data_loader_orig = data_loader_orig
        self.iterator = iter(data_loader_orig)

    def __iter__(self):
        return self

    def __next__(self):
        iterator_result = next(self.iterator)
        inputs_flat = {}
        labels_flat = {}
        if len(iterator_result) == 1:
            img_mean = torch.FloatTensor([0.485, 0.456, 0.406]).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(2)
            img_std = torch.FloatTensor([0.229, 0.224, 0.225]).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(2)
            iterator_result_keys = list(iterator_result[0].keys())
            inputs = {}
            labels = {}
            inputs['img'] = iterator_result[0]['img']
            inputs['img'] = inputs['img'].permute(0, 3, 1, 2)
            inputs['img'] = inputs['img'].float()
            inputs['img'] /= 255.
            inputs['img'] = (inputs['img'] - img_mean) / img_std
            inputs['file_name_info'] = iterator_result[0]['labels']
            labels['img'] = iterator_result[0]['img'].clone()
            labels['img'] = labels['img'].permute(0, 3, 1, 2)
            labels['img'] = labels['img'].float()
            labels['img'] /= 255.
            labels['img'] = (labels['img'] - img_mean) / img_std
            if 'optical_flow' in iterator_result[0].keys():
                inputs['optical_flow'] = iterator_result[0]['optical_flow']
                inputs['optical_flow'] = inputs['optical_flow'].permute(0, 3, 1, 2)
                inputs['optical_flow'] = inputs['optical_flow'].float()
                inputs['optical_flow'] /= 255.
                inputs['optical_flow'] = (inputs['optical_flow'] - img_mean) / img_std
            # calibration info
            calib_keys = ["intrinsic", "inverse_intrinsic", "extrinsic_rot", "extrinsic_rot_inv", "camera_extrinsics", "inverse_camera_extrinsics", "extrinsic_pos"]
            for c_key in calib_keys:
                if c_key in iterator_result[0].keys():
                    inputs[c_key] = iterator_result[0][c_key].clone()
                    inputs[c_key] = inputs[c_key].float()

            for key in inputs:
                x = inputs[key]
                inputs_flat[key] = x

            for key in labels:
                x = labels[key]
                labels_flat[key] = x

        else:
            inputs, labels = iterator_result

            for key in inputs:
                x = inputs[key]
                if isinstance(x, torch._C._TensorBase):
                    inputs_flat[key] = x.view([-1, *list(x.size())[2:]]) # squeeze first dimension
                elif type(x).__module__ == 'numpy' and type(x).__name__ == 'ndarray':
                #if x.dtype.kind in {'U', 'S'}:
                    inputs_flat[key] = np.reshape(x,[-1, *x.shape[2:]]) # merge first two dimensions into one
                else:
                    inputs_flat[key] = x


            for key in labels:
                x = labels[key]
                if isinstance(x, torch._C._TensorBase):
                    labels_flat[key] = x.view([-1, *list(x.size())[2:]]) # squeeze first dimension
                elif type(x).__module__ == 'numpy' and type(x).__name__ == 'ndarray':
                #if x.dtype.kind in {'U', 'S'}:
                    labels_flat[key] = np.reshape(x,[-1, *x.shape[2:]]) # merge first two dimensions into one
                else:
                    labels_flat[key] = x


        #print(inputs_flat['img'].shape)
        #print(inputs_flat['file_name_info'].shape)
        return inputs_flat, labels_flat

    def __len__(self):
        return len(self.iterator)

    def __del__(self):
        del self.iterator

class PostFlattenInputSubbatchTensor(object):
    """Flattens the first two dimensions (batch and sub-batch) to make subbatch data compatible with networks accepting a single image"""
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return PostFlattenInputSubbatchTensorIter(self.data_loader)

class DataLoaderMixIter(torch.utils.data.DataLoader):
    """Mixes..."""
    def __init__(self, data_loader_list, callocate_inputs, callocate_labels):
        self.data_loader_iter_list = []
        self.callocate_inputs = callocate_inputs
        self.callocate_labels = callocate_labels

        # total length is maximum length
        self.len_max = max([len(diter) for diter in data_loader_list])

        # return entries multiple times for those which have shorter length
        for loader in data_loader_list:
            if len(loader) < self.len_max:
                iter_new = DataIteratorEndless(loader)
            else:
                iter_new = iter(loader)
            self.data_loader_iter_list.append(iter_new)

    def __iter__(self):
        return self


    def __next__(self):
        inputs_per_loader = []
        labels_per_loader = []

        #print('Mixing from {} loaders'.format(len(self.data_loader_iter_list)))
        for data_iter in self.data_loader_iter_list:
            data = next(data_iter)
            inputs_per_loader.append(data[0])
            labels_per_loader.append(data[1])

        if self.callocate_inputs:
            inputs_per_loader = cat_collate(inputs_per_loader)

        if self.callocate_labels:
            labels_per_loader = cat_collate(labels_per_loader)
        else: # otherwise turn the list into a dict
            labels_per_loader = {i : v for i,v in enumerate(labels_per_loader)}
        return inputs_per_loader, labels_per_loader

    def __len__(self):
        return self.len_max

    def __del__(self):
        for i in self.data_loader_iter_list:
            del i

class DataLoaderMix(torch.utils.data.DataLoader):
    """Mixes..."""
    def __init__(self, data_loader_list, callocate_inputs=True, callocate_labels=True):
        self.data_loader_list = data_loader_list
        self.callocate_inputs = callocate_inputs
        self.callocate_labels = callocate_labels

    def __len__(self):
        lengths = []
        for loader in self.data_loader_list:
            lengths.append(len(loader))
        return max(lengths)

    def __iter__(self):
        return DataLoaderMixIter(self.data_loader_list, self.callocate_inputs, self.callocate_labels)
