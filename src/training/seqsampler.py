import numpy as np


class RandomSampler(object):
    """
    Randomizes from which dataset the data comes from. It is given the different datasets from which to sample as
    well as the batch size.
    """
    def __init__(self, sequences, sample_length, rng=np.random.RandomState()):
        if any(len(seq) < sample_length for seq in sequences):
            raise ValueError("All sequences must be larger than sample_length")

        self.sequences = sequences
        self.sample_length = sample_length
        self.rng = rng

        self.iters_per_epoch = len(sequences)

    def get_minibatch(self, it):
        """
        :param it: For compatibility only (not used)
        """
        sequence_index = self.rng.randint(len(self.sequences))
        sequence = self.sequences[sequence_index]

        start = self.rng.randint(len(sequence) - self.sample_length)
        stop = start + self.sample_length

        return sequence[start:stop], -1

    def shuffle(self):
        pass


class SequentialSampler(object):
    # TODO : Understand this
    def __init__(self, sequences_x, sequences_y, sample_length, sample_step=None, rng=np.random.RandomState()):
        if any(len(s_x) != len(s_y) for s_x, s_y in zip(sequences_x, sequences_y)):
            raise ValueError("Sequences x and y should have the same length")

        self.sequences_x = sequences_x
        self.sequences_y = sequences_y
        self.sample_length = sample_length

        self.sample_step = sample_step or sample_length

        self.rng = rng

        self.indices = np.arange(len(self.sequences_x))
        self._update()

    def iter_to_sequence(self, epoch_iter):

        i = np.searchsorted(self.cumiters_per_sequence, epoch_iter, side='right')

        sequence_index = self.indices[i]

        sequence_iter = epoch_iter
        if i > 0:
            sequence_iter = epoch_iter - self.cumiters_per_sequence[i - 1]

        return sequence_index, sequence_iter

    def get_minibatch(self, it):

        epoch_iter = it % self.iters_per_epoch
        sequence_index, sequence_iter = self.iter_to_sequence(epoch_iter)

        sequence_x = self.sequences_x[sequence_index]
        sequence_y = self.sequences_y[sequence_index]
        start = sequence_iter * self.sample_step
        stop = start + self.sample_length

        x = sequence_x[start:stop]
        y = sequence_y[start:stop]

        step = self.sample_step
        if stop >= len(sequence_x):
            step = -1

        return x, y, step

    def shuffle(self):

        self.rng.shuffle(self.indices)
        self._update()

    def _update(self):
        self.iters_per_sequence = np.array([(len(self.sequences_x[i]) - self.sample_length + self.sample_step - 1) // self.sample_step + 1
                                        for i in self.indices])
        self.iters_per_epoch = np.sum(self.iters_per_sequence)
        self.cumiters_per_sequence = np.cumsum(self.iters_per_sequence)

    def save_state(self, filename):
        pass

    def load_state(self, filename):
        pass

