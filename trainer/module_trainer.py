from mxnet import mod, sym, metric, io
from mxnet.gluon import utils

from learning_rate import TransformerSchedule
from trainer.trainer import Trainer

import mxnet as mx


class ModuleTrainer(Trainer):

    def __init__(self, opts, net, ctx):
        super(ModuleTrainer, self).__init__(opts, net)

        self._ctx = ctx

        self._loss = self._net.projector.get_loss()
        self._loss_metric = metric.Loss('loss')

        output = self._net(sym.Variable('x'), sym.Variable('x_len'), sym.Variable('dec_input'), sym.Variable('y_len'))
        #self._graph = sym.MakeLoss(sym.square(output - sym.Variable('y')), normalization='batch')
        self._graph = sym.MakeLoss(self._loss(output, sym.Variable('y')), normalization='batch')

        self._optimizer_params = {
            'beta1': self._opts.beta1,
            'beta2': self._opts.beta2,
            'epsilon': self._opts.eps
        }

        self._scheduler = TransformerSchedule(self._opts.dim_model)
        self._optimizer = mx.optimizer.Adam(**self._optimizer_params, lr_scheduler=self._scheduler)

        self._data_shapes = {
            'x': (opts.batch_size, 1, opts.enc_data_size),
            'x_len': (opts.batch_size, 1),
            'dec_input': (opts.batch_size, 1, opts.forecast_window),
            'y_len': (opts.batch_size, 1)
        }
        self._data_shapes = [(k, v) for k, v in self._data_shapes.items()]

        self._label_shapes = {'y': (opts.batch_size, 1, opts.dec_data_size)}
        self._label_shapes = [(k, v) for k, v in self._label_shapes.items()]

        self._module = mod.Module(self._graph, data_names=['x', 'x_len', 'dec_input', 'y_len'], label_names=['y'], context=self._ctx)
        self._module.bind(data_shapes=self._data_shapes, label_shapes=self._label_shapes)
        self._module.init_params()
        self._module.init_optimizer(optimizer=self._optimizer)

    def train(self, train_data, *args):
        for epoch in range(self._epochs):
            self.e_tick()

            self._loss_metric.reset()

            for i, batch in enumerate(train_data):
                self.b_tick()

                X, X_len, Y, Y_len = self._split_and_load(batch)

                for x, x_len, y, y_len in zip(X, X_len, Y, Y_len):
                    dec_input = x.slice_axis(begin=-self._opts.forecast_window, end=None, axis=-1)
                    data_batch = io.DataBatch([x, x_len, dec_input, y_len], [y])
                    self._module.forward(data_batch, is_train=True)
                    self._module.update_metric(self._loss_metric, data_batch.label)
                    outputs = self._module.get_outputs()[0]
                    print(outputs)

                # backpropagate
                self._module.backward()
                self._module.update()

            # per epoch logging
            _, loss = self._loss_metric.get()

            global_step = epoch + 1
            self._log('[Epoch {}] exec time: {:.2f}'.format(global_step, self.e_duration()))
            self._log('\tloss = {:.6f}'.format(loss))

    def _split_and_load(self, batch):
        conf = {'ctx_list': self._ctx, 'batch_axis': 0}
        (x, x_len), (y, y_len) = batch

        x = utils.split_and_load(x, **conf)
        x_len = utils.split_and_load(x_len, **conf)
        y = utils.split_and_load(y, **conf)
        y_len = utils.split_and_load(y_len, **conf)

        return x, x_len, y, y_len

    def _export_model(self, num_epoch):
        pass

    def _do_checkpoint(self, cur_epoch):
        pass