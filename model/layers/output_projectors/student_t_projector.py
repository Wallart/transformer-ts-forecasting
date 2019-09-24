from mxnet.gluon import nn
from model.layers.output_projectors.abstract.abstract_projector import AbstractProjector


class StudentTProjector(AbstractProjector):

    def __init__(self, *args, **kwargs):
        super(StudentTProjector, self).__init__(**kwargs)

        with self.name_scope():
            self._mu = nn.Dense(1, flatten=False)  # mean
            self._nu = nn.Dense(1, flatten=False)  # degrees of freedom
            self._sigma = nn.Dense(1, flatten=False)  # std

    def hybrid_forward(self, F, x, *args, **kwargs):
        sigma = F.Activation(self._sigma(x), 'softrelu')  # std cannot be negative
        nu = 2.0 + F.Activation(self._nu(x), 'softrelu')  # cannot be negative
        mu = self._mu(x)
        return mu, sigma, nu

    def get_loss(self):
        pass

    def get_sample(self):
        pass