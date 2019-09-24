from mxnet.gluon import utils
from predictor import forecast, reconstruct
from utils.visualization import samples_to_plot, plot, fill_between, new_fig, agg_dist_tensor

import csv


class Predictor:

    def __init__(self, opts, net, ctx):
        self._opts = opts

        self._anomalies = {}

        self._net = net
        self._ctx = ctx

    def reconstruct(self, data, window=34):
        for i, (X, X_len) in enumerate(data):
            X = utils.split_and_load(X, ctx_list=self._ctx, batch_axis=0, even_split=False)
            X_len = utils.split_and_load(X_len, ctx_list=self._ctx, batch_axis=0, even_split=False)

            recons = [reconstruct(self._net, x, window, self._opts.forecast_window) for x, x_len in zip(X, X_len)]
            self.handle_reconstruct(i, X, recons, window)

    def handle_reconstruct(self, batch_num, batch_x, batch_r, start_idx_recon):
        batch_x, batch_r = agg_dist_tensor(batch_x), agg_dist_tensor(batch_r)
        for j, (x, r) in enumerate(zip(batch_x, batch_r)):
            serie_num = batch_num * self._opts.batch_size + j

            fig, ax = new_fig()
            plot(ax, x, 0)
            fill_between(ax, r, start_idx_recon, color='teal')
            fig.suptitle('serie_{}'.format(serie_num))
            fig.show()

    def forecast(self, data):
        for i, (X, X_len) in enumerate(data):
            Y = X.slice_axis(begin=-self._opts.forecast_window, end=None, axis=2)
            X = X.slice_axis(begin=0, end=-self._opts.forecast_window, axis=2)
            X_len -= self._opts.forecast_window

            X = utils.split_and_load(X, ctx_list=self._ctx, batch_axis=0, even_split=False)
            X_len = utils.split_and_load(X_len, ctx_list=self._ctx, batch_axis=0, even_split=False)
            forecasts = [forecast(self._net, x, x_len, self._opts.forecast_window) for x, x_len in zip(X, X_len)]
            self.handle_forecast(i, X, [Y], forecasts)

    def handle_forecast(self, batch_num, batch_x, batch_y, batch_o):
        batch_x, batch_y, batch_o = agg_dist_tensor(batch_x), agg_dist_tensor(batch_y), agg_dist_tensor(batch_o)
        for j, (x, y, o) in enumerate(zip(batch_x, batch_y, batch_o)):
            serie_num = batch_num * self._opts.batch_size + j

            fig = samples_to_plot(x, y, o)
            fig.suptitle('serie_{}'.format(serie_num))
            fig.show()

    def _write_csv(self, num_series, outfile):
        rows = [['seqID', 'anomaly']]
        for i in range(num_series):
            is_anomalous = 1 if i in self._anomalies else 0
            rows.append([i, is_anomalous])

        with open(outfile, 'w', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerows(rows)
