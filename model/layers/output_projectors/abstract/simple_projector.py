from abc import ABCMeta
from mxnet.gluon import nn
from model.layers.output_projectors.abstract.abstract_projector import AbstractProjector


class SimpleProjector(AbstractProjector, metaclass=ABCMeta):

    def __init__(self, units, **kwargs):
        super(AbstractProjector, self).__init__(**kwargs)
        self._units = units

        with self.name_scope():
            self._linear = nn.Dense(units, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self._linear(x).swapaxes(1, 2)
