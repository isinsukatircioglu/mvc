import numpy as np
from . import srng


class Sampler(object):
    def __init__(self,
                 datasets,
                 minibatch_size,
                 rng=None):
        """

        :param datasets: List of datasets
        :param minibatch_size:
        :param rng:
        """

        self.datasets = datasets
        self.num_samples = len(datasets[0])
        self.minibatch_size = minibatch_size
        self.rng = rng or srng.RNG()
        self.iters_per_epoch = self.num_samples // self.minibatch_size

        self._indices = None
        self._indices_epoch = None

    def __getitem__(self, index):
        return self.get_minibatch(index)

    def get_minibatch(self, index):
        epoch, i = divmod(index, self.iters_per_epoch)
        self._shuffle(epoch)

        minibatch_indices = self._indices[i * self.minibatch_size:(i+1) * self.minibatch_size]

        # Required for H5 compatibility.
        return tuple(np.array([some_dataset[some_idx] for some_idx in minibatch_indices]) for some_dataset in self.datasets)

    def _shuffle(self, epoch):
        """
        Shuffles the indices of the inputs of the datasets. This is to randomize the order during an epoch.
        This function only shuffles once per epoch.
        :param epoch: The current epoch
        """
        if self._indices_epoch == epoch:
            return

        self._indices = np.arange(self.num_samples)
        self.rng.shuffle(epoch, self._indices)
        self._indices_epoch = epoch


class VaeSampler(Sampler):

    def __init__(self,
                 num_latent_variables,
                 train_x,
                 minibatch_size,
                 minibatch_shape=None,
                 rng=np.random.RandomState()):
        super(VaeSampler, self).__init__([train_x, train_x], minibatch_size, rng=rng)
        self.k = num_latent_variables

    def get_minibatch(self, index):
        current_x, current_y = super(VaeSampler, self).get_minibatch(index)
        current_e = np.random.normal(size=(self.minibatch_size, self.k))

        return current_x, current_y, np.float32(current_e)


# class SamplerWithConstraints(Sampler):
#
#     def __init__(self,
#                  train_x,
#                  train_y,
#                  constraints_x,
#                  minibatch_size,
#                  constraints_size,
#                  minibatch_shape=None,
#                  rng=np.random.RandomState()):
#
#         super(SamplerWithConstraints, self).__init__(
#                 [train_x, train_y],
#                 minibatch_size,
#                 minibatch_shape,
#                 constraints_size,
#                 rng)
#
#         self.constraints_x = constraints_x
#         self.constraints_num_samples = len(self.constraints_x)
#         self.constraints_indices = np.arange(self.constraints_num_samples)
#
#         self.constraints_iters_per_epoch = self.constraints_num_samples // self.constraints_size
#
#     def get_minibatch_constraints(self, index):
#
#         i = index % self.constraints_iters_per_epoch
#         minibatch_elements = self.constraints_indices[i * self.constraints_size : (i + 1) * self.constraints_size]
#
#         # H5 requires sorted indices
#         argsort = np.argsort(minibatch_elements)
#         inv_argsort = np.argsort(argsort)
#         minibatch_elements = list(minibatch_elements[argsort])
#
#         res = self.constraints_x[minibatch_elements][inv_argsort]
#
#         if self.minibatch_shape is not None:
#             res = np.reshape(res, self.minibatch_shape)
#
#         return res
#
#     def shuffle(self):
#         super(SamplerWithConstraints, self).shuffle()
#         self.rng.shuffle(self.constraints_indices)
#
#     def save_state(self, filename):
#         raise NotImplementedError
#
