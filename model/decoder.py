from mxnet import gluon
from mxnet.gluon import nn
from model.layers.decoder_layer import DecoderLayer
from model.layers.positional_encoding import PositionalEncoding

import math


class Decoder(gluon.HybridBlock):

    def __init__(self, opts, projector_params, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

        self._dim_model = opts.dim_model
        self._num_layers = opts.num_layers

        with self.name_scope():
            self._projector = nn.Dense(opts.dim_model, flatten=False, params=projector_params)

            self._pos_encoding = PositionalEncoding(opts.dim_model)
            self._dropout = nn.Dropout(opts.dropout_rate)

            self._blocks = nn.HybridSequential()
            for _ in range(opts.num_layers):
                self._blocks.add(DecoderLayer(opts.dim_model, opts.dim_ff, opts.num_heads, opts.dropout_rate))

    def hybrid_forward(self, F, x, *args, **kwargs):
        enc_output, look_ahead_mask, padding_mask = args

        x = self._projector(x.swapaxes(1, 2))  # NCT -> NTC
        x = x * math.sqrt(self._dim_model)

        x = self._pos_encoding(x)
        x = self._dropout(x)

        # using a loop because we cannot pass multiple args to HybridSequential's hybrid_forward
        attn_weights_1, attn_weights_2 = [], []
        for block in self._blocks:
            x, attn_w_1, attn_w_2 = block(x, enc_output, look_ahead_mask, padding_mask)

            attn_weights_1.append(attn_w_1.expand_dims(1))
            attn_weights_2.append(attn_w_2.expand_dims(1))

        attn_weights_1 = F.concat(*attn_weights_1, dim=1)
        attn_weights_2 = F.concat(*attn_weights_2, dim=1)

        return x, attn_weights_1, attn_weights_2
