import os
import os.path
from collections import defaultdict
from functools import reduce

import numpy as np
import h5py


class Summary(object):
    def __init__(self):
        self.content = defaultdict(dict)

    def register(self, tag, index, value):
        self.content[tag][index] = value

    def has_tag(self, tag):
        return tag in self.content

    def keys(self):
        return self.content.keys()

    def get(self, tag):
        if tag not in self.content:
            raise KeyError(tag)

        data = self.content[tag]

        indices = []
        values = []
        for index in sorted(data):
            indices.append(index)
            values.append(data[index])

        return np.asarray(indices), np.asarray(values)

    def get_many(self, tags):
        for tag in tags:
            if tag not in self.content:
                raise KeyError

        dicts = [self.content[tag] for tag in tags]
        indices = [list(d.keys()) for d in dicts]
        indices = reduce(np.intersect1d, indices)
        indices = sorted(indices)

        results = tuple([] for _ in tags)
        for index in indices:
            for d, res in zip(dicts, results):
                res.append(d[index])

        return indices, results

    def save(self, filename, backup=False):
        if backup and os.path.isfile(filename):
            os.rename(filename, filename + ".bak")
        np.save(filename, self.content)

    def load(self, filename):
        self.content = np.load(filename).item()


def save_h5(summary, filename):
    f = h5py.File(filename, "w")
    for tag in summary.content.keys():
        grp = f.create_group(tag)

        indices, values = summary.get(tag)

        grp.create_dataset("indices", data=indices)
        grp.create_dataset("values", data=values)
    f.close()


def load_h5(summary, filename=None):
    if filename is None:
        filename = summary
        summary = Summary()

    content = defaultdict(dict)

    if os.path.exists(filename):
        f = h5py.File(filename, "r")
        for tag in f.keys():
            grp = f[tag]
            indices = grp["indices"][:]
            values = grp["values"][:]
            for index, value in zip(indices, values):
                content[tag][index] = value
        f.close()
    else:
        print("No summary present, settign default values")
        content['training.time'] = np.array([[0,0]])

    summary.content = content
    return summary
