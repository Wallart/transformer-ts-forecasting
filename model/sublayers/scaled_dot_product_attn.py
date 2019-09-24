from mxnet import gluon

import math


class ScaledDotProductAttn(gluon.HybridBlock):

    def __init__(self, dim_k, *args, **kwargs):
        super(ScaledDotProductAttn, self).__init__(*args, **kwargs)
        self._dim_k = dim_k

    def hybrid_forward(self, F, *args, **kwargs):
        query, key, value, mask = args

        matmul_qk = F.linalg.gemm2(query, key, transpose_b=True)  # seq_len_q, seq_len_k
        scaled_attn_logits = matmul_qk / math.sqrt(self._dim_k)

        if mask is not None:
            scaled_attn_logits = F.broadcast_add(scaled_attn_logits, mask * -1e9)

        attn_weights = F.softmax(scaled_attn_logits)  # seq_len_q, seq_len_k
        output = F.linalg.gemm2(attn_weights, value)  # seq_len_q, seq_len_k
        return output, attn_weights
