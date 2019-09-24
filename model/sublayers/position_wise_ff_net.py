from mxnet import nd, gluon
from mxnet.gluon import nn


class PositionWiseFFNet(gluon.HybridBlock):

    def __init__(self, dim_ff, dim_model, *args, **kwargs):
        super(PositionWiseFFNet, self).__init__(*args, **kwargs)
        with self.name_scope():
            self._linear1 = nn.Dense(dim_ff, flatten=False)
            self._linear2 = nn.Dense(dim_model, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self._linear2(F.relu(self._linear1(x)))


if __name__ == '__main__':
    ffn = PositionWiseFFNet(2048, 512)
    ffn.initialize()

    x = nd.ones((32, 344, 5))
    out = ffn(x)
    print(out.shape)
