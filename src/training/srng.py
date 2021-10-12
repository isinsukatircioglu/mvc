import numpy as np
import hashlib

maxint64 = (1 << 63)

class RNG(object):
    """Stateless RNG"""
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(maxint64)

        self.seed = bytes(str(seed) + ",", "utf-8")

    def randint(self, index, low=maxint64, high=None):
        # TODO: Add size and make more efficient
        if high is None:
            high = low
            low = 0
        n = high - low

        if n <= 0:
            raise ValueError("low >= high")

        aux = hashlib.sha256(self.seed + bytes(str(index), "utf-8"))
        # Get an integer using the first 64 bits
        v = int(aux.hexdigest()[:16], 16)

        return (v % n) + low

    def uniform(self, index, low=0.0, high=1.0):
        aux = self.randint(str(index) + "f") / maxint64
        return aux * (high - low) + low

    def shuffle(self, index, lst):
        for i in range(len(lst) - 1, 0, -1):
            j = self.randint("{},{}".format(index, i), i + 1)
            lst[i], lst[j] = lst[j], lst[i]
        return lst

