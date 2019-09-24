from model.layers.output_projectors.abstract.simple_projector import SimpleProjector
from model.loss.quantile_loss import QuantileLoss


class QuantileProjector(SimpleProjector):

    def __init__(self, quantiles=None, **kwargs):
        self._quantiles = quantiles if quantiles is not None else [.05, .5, .95]
        assert type(self._quantiles) == list
        super(QuantileProjector, self).__init__(len(self._quantiles), **kwargs)

    def get_loss(self):
        return QuantileLoss(self._quantiles)

    def get_sample(self, *args):
        tensor, = args
        return tensor[:, 1, :]
