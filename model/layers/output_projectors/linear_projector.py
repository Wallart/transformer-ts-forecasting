from mxnet.gluon import loss
from model.layers.output_projectors.abstract.simple_projector import SimpleProjector


class LinearProjector(SimpleProjector):

    def __init__(self, **kwargs):
        super(LinearProjector, self).__init__(1, **kwargs)

    def get_loss(self):
        return loss.HuberLoss()

    def get_sample(self, *args):
        tensor, = args
        return tensor
