from mxnet import gluon
from mxnet.gluon import nn
from model.sublayers.multi_head_attn import MultiHeadAttn
from model.sublayers.position_wise_ff_net import PositionWiseFFNet


class EncoderLayer(gluon.HybridBlock):

    def __init__(self, dim_model, dff, heads, dropout_rate=0.1, *args, **kwargs):
        super(EncoderLayer, self).__init__(*args, **kwargs)

        self._attn = MultiHeadAttn(dim_model, heads)
        self._ffn = PositionWiseFFNet(dff, dim_model)

        self._layer_norm_1 = nn.LayerNorm(epsilon=1e-6)
        self._layer_norm_2 = nn.LayerNorm(epsilon=1e-6)

        self._dropout_1 = nn.Dropout(dropout_rate)
        self._dropout_2 = nn.Dropout(dropout_rate)

    def hybrid_forward(self, F, x, *args, **kwargs):
        mask, = args

        attn_output, _ = self._attn(x, x, x, mask)
        attn_output = self._dropout_1(attn_output)
        out1 = self._layer_norm_1(x + attn_output)

        ffn_output = self._ffn(out1)
        ffn_output = self._dropout_2(ffn_output)
        out2 = self._layer_norm_2(out1 + ffn_output)

        return out2


if __name__ == '__main__':
    from mxnet import nd

    enc_blk = EncoderLayer(512, 2048, 8, dropout_rate=.5)
    enc_blk.initialize()

    x = nd.ones((32, 614, 512))
    print(enc_blk(x, None).shape)
