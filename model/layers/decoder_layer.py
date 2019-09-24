from mxnet import gluon
from mxnet.gluon import nn
from model.sublayers.multi_head_attn import MultiHeadAttn
from model.sublayers.position_wise_ff_net import PositionWiseFFNet


class DecoderLayer(gluon.HybridBlock):

    def __init__(self, dim_model, dff, heads, dropout_rate=0.1, *args, **kwargs):
        super(DecoderLayer, self).__init__(*args, **kwargs)

        self._attn_1 = MultiHeadAttn(dim_model, heads)
        self._attn_2 = MultiHeadAttn(dim_model, heads)
        self._ffn = PositionWiseFFNet(dff, dim_model)

        self._layer_norm_1 = nn.LayerNorm(epsilon=1e-6)
        self._layer_norm_2 = nn.LayerNorm(epsilon=1e-6)
        self._layer_norm_3 = nn.LayerNorm(epsilon=1e-6)

        self._dropout_1 = nn.Dropout(dropout_rate)
        self._dropout_2 = nn.Dropout(dropout_rate)
        self._dropout_3 = nn.Dropout(dropout_rate)

    def hybrid_forward(self, F, x, *args, **kwargs):
        enc_output, look_ahead_mask, padding_mask = args

        attn_1, attn_weights_1 = self._attn_1(x, x, x, look_ahead_mask)
        attn_1 = self._dropout_1(attn_1)
        out1 = self._layer_norm_1(x + attn_1)

        attn_2, attn_weights_2 = self._attn_2(enc_output, enc_output, out1, padding_mask)
        attn_2 = self._dropout_2(attn_2)
        out2 = self._layer_norm_2(out1 + attn_2)

        ffn_output = self._ffn(out2)
        ffn_output = self._dropout_3(ffn_output)
        out3 = self._layer_norm_3(out2 + ffn_output)

        return out3, attn_weights_1, attn_weights_2


if __name__ == '__main__':
    from mxnet import nd
    dec_blk = DecoderLayer(0, 512, 2048, 8, .5)
    dec_blk.initialize()

    x = nd.ones((32, 624, 512))

    out, _, _ = dec_blk(x)
    print(out.shape)
