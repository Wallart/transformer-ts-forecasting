from mxnet import nd, gluon


class QuantileLoss(gluon.HybridBlock):
    def __init__(self, quantiles, quantile_weights=None, **kwargs):
        super().__init__(**kwargs)

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.quantile_weights = nd.ones(self.num_quantiles) / self.num_quantiles if not quantile_weights else quantile_weights

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, y_pred, y_true, sample_weight=None):
        y_pred_all = F.split(y_pred, axis=1, num_outputs=self.num_quantiles)

        qt_loss = []
        for i, y_pred_q in enumerate(y_pred_all):
            q = self.quantiles[i]
            weighted_qt = (self.compute_quantile_loss(F, y_true, y_pred_q, q) * self.quantile_weights[i].asscalar())
            qt_loss.append(weighted_qt)

        avg_qt_losses = F.concat(*qt_loss, dim=1).mean(axis=1)  # avg across quantiles

        if sample_weight is None:
            sample_weight = 1

        avg_qt_losses = avg_qt_losses * sample_weight
        return avg_qt_losses.mean(axis=0, exclude=True)

    @staticmethod
    def compute_quantile_loss(F, y_true, y_pred_p, p):
        under_bias = p * F.maximum(y_true - y_pred_p, 0)
        over_bias = (1 - p) * F.maximum(y_pred_p - y_true, 0)

        qt_loss = 2 * (under_bias + over_bias)

        return qt_loss
