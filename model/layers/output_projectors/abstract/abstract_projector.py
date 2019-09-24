from abc import abstractmethod, ABCMeta
from mxnet import gluon


class AbstractProjector(gluon.HybridBlock, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        super(AbstractProjector, self).__init__(**kwargs)

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_sample(self, *args):
        pass
