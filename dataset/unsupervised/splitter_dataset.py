from mxnet import nd
from dataset.abstract_dataset import AbstractDataset


class SplitterDataset(AbstractDataset):

    def __init__(self, dataset, forecast_window, *args, **kwargs):
        super(SplitterDataset, self).__init__(*args, **kwargs)

        x = dataset.get_block(0, dataset.width() - forecast_window)
        y = dataset.get_block(dataset.width() - forecast_window, dataset.width())

        self._arr_x = x
        self._arr_y = y

        assert x.shape[0] == y.shape[0], 'x and y does number of elements is different'
        assert x.shape[1] == y.shape[1], 'x and y does not have the same number of features'

    def __getitem__(self, idx):
        return nd.array(self._arr_x[idx]), nd.array(self._arr_y[idx])

    def width(self):
        return self._arr_x.shape[2], self._arr_y.shape[2]
