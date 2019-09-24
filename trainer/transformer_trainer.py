from mxnet import nd, autograd, gluon, profiler, metric
from mxnet.gluon import utils, Parameter
from mxboard import SummaryWriter

from predictor import forecast
from learning_rate import TransformerSchedule, apply_lr_schedule
from trainer.trainer import Trainer
from utils.visualization import samples_to_board, attn_weights_to_board

import os
import time
import mxnet as mx


class TransformerTrainer(Trainer):

    def __init__(self, opts, net, ctx):
        super(TransformerTrainer, self).__init__(opts, net)

        self._ctx = ctx

        self._loss = self._net.projector.get_loss()
        self._loss_metric = metric.Loss('loss')
        self._rmse_metric = metric.RMSE()
        self._best_rmse = 10000

        self._optimizer_params = {
            'beta1': self._opts.beta1,
            'beta2': self._opts.beta2,
            'epsilon': self._opts.eps,
        }

        self._scheduler = TransformerSchedule(self._opts.dim_model)
        self._optimizer = mx.optimizer.Adam(**self._optimizer_params, lr_scheduler=self._scheduler)
        self._trainer = gluon.Trainer(self._net.collect_params(), self._optimizer)

    def model_name(self):
        return 'transformer'

    def train(self, train_data, *args):
        val_data = args[0]

        with SummaryWriter(logdir=self._outlogs, flush_secs=5, verbose=False) as writer:
            num_iter = len(train_data)
            self._visualize_lr_schedule(writer)
            self.t_tick()
            for epoch in range(self._epochs):
                self.e_tick()

                self._loss_metric.reset()

                for i, batch in enumerate(train_data):
                    self.b_tick()

                    self._visualize_graphs(epoch, i)
                    self._visualize_weights(writer, epoch, i)

                    # start profiler once every thing is inferred
                    if self._profile and epoch == 0 and i == 1:
                        profiler.set_state('run')

                    # split data across gpus
                    X, X_len, Y, Y_len = self._split_and_load(batch)
                    with autograd.record():
                        targets = []
                        for x, x_len, y, y_len in zip(X, X_len, Y, Y_len):
                            dec_input = x.slice_axis(begin=-self._opts.forecast_window, end=None, axis=-1)
                            targets.append(self._net(x, x_len, dec_input, y_len))

                        outputs, attn_w_1, attn_w_2 = map(list, zip(*targets))
                        losses = [self._loss(o, y) for o, y in zip(outputs, Y)]

                    # backpropagate
                    autograd.backward(losses)
                    self._trainer.step(self._opts.batch_size)
                    self._loss_metric.update(0, [l * self._opts.batch_size for l in losses])
                    self._rmse_metric.update(Y, outputs)

                    # visualize generated image each x epoch over each gpus
                    if (i + 1) == num_iter and (epoch + 1) % self._opts.thumb_interval == 0:
                        t0 = time.time()
                        self._log('Plotting samples...')
                        samples_to_board(writer, [X, Y, outputs], epoch + 1, 'current_samples')
                        attn_weights_to_board(writer, attn_w_1, epoch + 1, 'current_attn_weights_block_1')
                        attn_weights_to_board(writer, attn_w_2, epoch + 1, 'current_attn_weights_block_2')
                        self._log('Plots generated in {:.2f} second(s)'.format(time.time() - t0))

                    # per x iter logging
                    if i % self._log_interval == 0:
                        b_time = self.b_duration()
                        speed = self._opts.batch_size / b_time
                        iter_stats = 'exec time: {:.2f} second(s) speed: {:.2f} samples/s'.format(b_time, speed)
                        self._log('[Epoch {}] --[{}/{}]-- {}'.format(epoch + 1, i + 1, num_iter, iter_stats))

                # per epoch logging
                _, loss = self._loss_metric.get()
                _, rmse_train = self._rmse_metric.get()

                rmse_val = self._validate(epoch, val_data)
                rmse_val_str = 'rmse-val = {:.6f}'.format(rmse_val) if rmse_val is not None else ''

                global_step = epoch + 1
                self._log('[Epoch {}] exec time: {:.2f}'.format(global_step, self.e_duration()))
                self._log('\tloss = {:.6f}'.format(loss))
                self._log('\trmse-train = {:.6f} {}'.format(rmse_train, rmse_val_str))

                # per epoch reporting
                writer.add_scalar(tag='loss', value=loss, global_step=global_step)
                rmse_err = {'train': rmse_train, 'validation': rmse_val} if rmse_val is not None else {'train': rmse_train}
                writer.add_scalar(tag='rmse', value=rmse_err, global_step=global_step)

                if rmse_val is not None:
                    if rmse_val < self._best_rmse:
                        self._best_rmse = rmse_val

                        if self._opts.chkpt_interval is None:
                            self._log('RMSE has decreased ! Checkpointing.')
                            self._do_checkpoint(global_step)

                # save model each x epochs
                if self._chkpt_interval is not None and (global_step % self._chkpt_interval == 0):
                    self._do_checkpoint(global_step)

            self._log('Exporting model...')
            nd.waitall()
            self._save_profile()
            self._export_model(self._epochs)
            self._log('Training complete. Done in {:.2f}s.'.format(self.t_duration()))

    def _validate(self, cur_epoch, val_data):
        if cur_epoch % self._val_interval != 0:
            return None

        rmse_metric = metric.RMSE()
        self.v_tick()

        for i, batch in enumerate(val_data):
            self._log('VALIDATION --[{}/{}]--'.format(i + 1, len(val_data)))
            X, X_len, Y, Y_len = self._split_and_load(batch)

            targets = [forecast(self._net, x, x_len, self._opts.dec_data_size) for x, x_len in zip(X, X_len)]
            preds, _, _ = map(list, zip(*targets))
            rmse_metric.update(Y, [p[:, 1, :] for p in preds])

        _, rmse = rmse_metric.get()
        self._log('Validation complete. Done in {:.2f}s.'.format(self.v_duration()))

        return rmse

    def _export_model(self, num_epoch):
        if not self._opts.no_hybridize:
            outfile = os.path.join(self._outdir, '{}'.format(self.model_name()))
            self._net.export(outfile, epoch=num_epoch)

    def _do_checkpoint(self, cur_epoch):
        outfile = os.path.join(self._outchkpts, '{}-{:04d}.chkpt'.format(self.model_name(), cur_epoch))
        self._net.save_parameters(outfile)

    def _visualize_lr_schedule(self, writer):
        lr_values = apply_lr_schedule(self._scheduler)
        for iteration, lr in enumerate(lr_values):
            writer.add_scalar(tag='lr/schedule', value=lr, global_step=iteration)

    def _visualize_graphs(self, cur_epoch, cur_iter):
        if not self._opts.no_hybridize and cur_epoch == 0 and cur_iter == 1:
            with SummaryWriter(logdir=self._outlogs, flush_secs=5, verbose=False) as writer:
                writer.add_graph(self._net)

    def _visualize_weights(self, writer, cur_epoch, cur_iter):
        if self._viz_interval > 0 and cur_iter == 0 and cur_epoch % self._viz_interval == 0:
            # to visualize gradients each x epochs
            params = [p for p in self._net.collect_params().values() if type(p) == Parameter and p._grad]
            for p in params:
                name = '{}/{}/{}'.format(self._net._name, '_'.join(p.name.split('_')[:-1]), p.name.split('_')[-1])
                aggregated_grads = nd.concat(*[grad.as_in_context(mx.cpu()) for grad in p._grad], dim=0)
                writer.add_histogram(tag=name, values=aggregated_grads, global_step=cur_epoch + 1, bins=1000)

    def _split_and_load(self, batch):
        conf = {'ctx_list': self._ctx, 'batch_axis': 0}
        if type(batch) == list or type(batch) == tuple:
            x_tuple = batch[0]
            x = utils.split_and_load(x_tuple[0], **conf)
            x_lengths = utils.split_and_load(x_tuple[1], **conf)

            y_tuple = batch[1]
            y = utils.split_and_load(y_tuple[0], **conf)
            y_lengths = utils.split_and_load(y_tuple[1], **conf)

            return x, x_lengths, y, y_lengths
        else:
            x = utils.split_and_load(batch, **conf)
            return x, x
