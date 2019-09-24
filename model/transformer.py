from mxnet import gluon, autograd
from model.decoder import Decoder
from model.encoder import Encoder


class Transformer(gluon.HybridBlock):

    def __init__(self, opts, projector, *args, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)

        self._batch_size = opts.batch_per_gpu
        self._num_layers = opts.num_layers
        self._num_features = opts.num_features

        self._x_max_size = opts.enc_data_size
        self._y_max_size = opts.dec_data_size

        with self.name_scope():
            self._enc = Encoder(opts)
            self._dec = Decoder(opts, self._enc.get_projector_params())
            self.projector = projector

    def hybrid_forward(self, F, x, *args, **kwargs):
        x_valid_len, y, y_valid_len = args

        look_ahead_mask = self.look_ahead_mask(F, self._y_max_size)
        dec_y_pad_mask = self.mask_from_lengths(F, y_valid_len, self._y_max_size)
        #combined_mask = F.broadcast_maximum(look_ahead_mask, dec_y_pad_mask)
        combined_mask = F.broadcast_maximum(look_ahead_mask, dec_y_pad_mask.swapaxes(2, 3))

        enc_pad_mask = self.mask_from_lengths(F, x_valid_len, self._x_max_size)
        dec_pad_mask = enc_pad_mask

        enc_outputs = self._enc(x, enc_pad_mask)
        dec_outputs, attn_w_1, attn_w_2 = self._dec(y, enc_outputs, combined_mask, dec_pad_mask)
        logits = self.projector(dec_outputs)

        if autograd.is_training():
            return logits, attn_w_1, attn_w_2

        return logits

    def look_ahead_mask(self, F, size):
        if size <= 1:
            return None

        num_elem_per_trian = size * (size - 1) // 2  # -1 diagonal excluded. +1 for diag included
        return F.linalg.maketrian(F.ones((num_elem_per_trian,)), offset=1, lower=False)

    def mask_from_lengths(self, F, lengths, max_len):
        ids = F.arange(0, max_len)
        return F.logical_not(F.broadcast_lesser(ids, lengths)).expand_dims(1).expand_dims(1)

    def model_name(self):
        return 'transformer'
