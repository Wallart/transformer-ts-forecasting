from mxnet import nd, gluon


class PositionalEncoding(gluon.HybridBlock):

    def __init__(self, dim_model, max_len=5000, *args, **kwargs):
        super(PositionalEncoding, self).__init__(*args, **kwargs)

        x = nd.arange(0, max_len).expand_dims(-1)
        x = x / nd.power(10000, nd.arange(0, dim_model, 2) / dim_model)

        pos = nd.zeros((1, max_len, dim_model))  # NTC
        pos[:, :, 0::2] = nd.sin(x)
        pos[:, :, 1::2] = nd.cos(x)

        self.pos = self.params.get_constant('pos', pos)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.broadcast_add(x, kwargs['pos'].slice_like(x, axes=1))


if __name__ == '__main__':
    pe = PositionalEncoding(512, 0, 344)
    pe.initialize()

    y = pe(nd.zeros((32, 344, 512)))
    print(y.shape)
