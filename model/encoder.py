from mxnet import gluon
from mxnet.gluon import nn
from model.layers.encoder_layer import EncoderLayer
from model.layers.positional_encoding import PositionalEncoding

import math


class Encoder(gluon.HybridBlock):

    def __init__(self, opts, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

        self._dim_model = opts.dim_model

        with self.name_scope():
            self._projector = nn.Dense(opts.dim_model, flatten=False)

            self._pos_encoding = PositionalEncoding(opts.dim_model)
            self._dropout = nn.Dropout(opts.dropout_rate)

            self._blocks = nn.HybridSequential()
            for i in range(opts.num_layers):
                self._blocks.add(EncoderLayer(opts.dim_model, opts.dim_ff, opts.num_heads, opts.dropout_rate))

    def hybrid_forward(self, F, x, *args, **kwargs):
        mask, = args

        x = self._projector(x.swapaxes(1, 2))  # NCT -> NTC
        x = x * math.sqrt(self._dim_model)

        x = self._pos_encoding(x)
        x = self._dropout(x)

        # using a loop because we cannot pass multiple args to HybridSequential's hybrid_forward
        for block in self._blocks:
            x = block(x, mask)

        return x

    def get_projector_params(self):
        return self._projector.params
