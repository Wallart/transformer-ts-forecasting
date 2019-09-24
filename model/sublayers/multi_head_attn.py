from mxnet import nd, gluon
from mxnet.gluon import nn
from model.sublayers.scaled_dot_product_attn import ScaledDotProductAttn


class MultiHeadAttn(gluon.HybridBlock):

    def __init__(self, dim_model, heads, *args, **kwargs):
        super(MultiHeadAttn, self).__init__(*args, **kwargs)
        self._num_heads = heads
        self._dim_model = dim_model

        assert dim_model % heads == 0
        self._depth = dim_model // self._num_heads

        with self.name_scope():
            self._attn = ScaledDotProductAttn(dim_model)

            self._w_v = nn.Conv1D(dim_model, kernel_size=1, use_bias=True)
            self._w_k = nn.Conv1D(dim_model, kernel_size=3, padding=1, use_bias=True)
            self._w_q = nn.Conv1D(dim_model, kernel_size=3, padding=1, use_bias=True)

            self._linear = nn.Dense(dim_model, use_bias=False, flatten=False)

    def hybrid_forward(self, F, *args, **kwargs):
        value, key, query, mask = args  # -> (batch, seq_len, d_model)

        query = self._w_q(query.swapaxes(1, 2)).swapaxes(1, 2)  # -> (batch_size, seq_len, d_model)
        key = self._w_k(key.swapaxes(1, 2)).swapaxes(1, 2)  # -> (batch_size, seq_len, d_model)
        value = self._w_v(value.swapaxes(1, 2)).swapaxes(1, 2)  # -> (batch_size, seq_len, d_model)

        query = self._split_heads(query)  # -> (batch_size, num_heads, seq_len_q, depth)
        key = self._split_heads(key)  # -> (batch_size, num_heads, seq_len_k, depth)
        value = self._split_heads(value)  # -> (batch_size, num_heads, seq_len_v, depth)

        # scaled_attn -> (batch_size, num_heads, seq_len_q, depth)
        # attn_weights -> (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attn, attn_weights = self._attn(query, key, value, mask)
        scaled_attn = scaled_attn.transpose((0, 2, 1, 3))  # -> batch_size, seq_len_q, num_heads, depth
        concat_attn = scaled_attn.reshape((0, 0, self._dim_model))  # -> batch_size, seq_len_q, dim_model

        output = self._linear(concat_attn)  # -> batch_size, seq_len_q, dim_model
        return output, attn_weights

    def _split_heads(self, x):
        x = x.reshape((0, 0, self._num_heads, self._depth))
        return x.swapaxes(1, 2)  # -> (batch_size, num_heads, seq_len, depth)
