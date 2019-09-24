from abc import ABC
from mxnet import nd
from mxnet.gluon import data


class AbstractDataset(data.Dataset, ABC):

    def __init__(self, *args, transforms=[], **kwargs):
        super(AbstractDataset, self).__init__()
        self._arr_x = None
        self._arr_y = None

        self._transforms = transforms

    def __getitem__(self, idx):
        return nd.array(self._apply_transforms(self._arr_x[idx]))

    def __len__(self):
        return self._arr_x.shape[0]

    def num_features(self):
        return self._arr_x.shape[1]

    def width(self):
        return self._arr_x.shape[2]

    def min(self):
        return self._arr_x.min()

    def max(self):
        return self._arr_x.max()

    def mean(self):
        return self._arr_x.mean()

    def std(self):
        return self._arr_x.std()

    def get_block(self, start_idx, end_idx):
        """
        Extract a time block from a serie
        :param start_idx: cursor start index
        :param end_idx: cursor end index (excluded)
        :return: np.array
        """
        return self._apply_transforms(self._arr_x[:, :, start_idx:end_idx])

    def add_transform(self, func, lazy=True):
        if lazy:
            self._transforms.append(func)
        else:
            self._arr_x = func(self, self.get_block(0, self.width()))

    def _apply_transforms(self, item):
        for t in self._transforms:
            item = t(self, item)

        return item
