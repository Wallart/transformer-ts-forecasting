from mxnet import nd

import mxnet as mx


def reconstruct(net, x, reconstruct_window, forecast_window):
    reconstructions = []
    for i in range(0, x.shape[2], forecast_window):
        end = min(i + reconstruct_window, x.shape[2])
        enc_input = x.slice_axis(begin=i, end=end, axis=-1)
        enc_input, enc_len = pad_input(enc_input, reconstruct_window)

        dec_input = enc_input.slice_axis(begin=-forecast_window, end=None, axis=-1)
        dec_len = nd.full((x.shape[0], x.shape[1]), forecast_window, ctx=x.context)

        outputs = net(enc_input, enc_len, dec_input, dec_len)
        reconstructions.append(outputs)

    reconstructions = nd.concat(*reconstructions, dim=-1)
    return reconstructions.slice_axis(begin=0, end=x.shape[2] - reconstruct_window, axis=-1)


def forecast(net, x, x_len, forecast_window):
    input, input_len = pad_input(x.slice_axis(begin=-forecast_window, end=None, axis=-1), forecast_window)
    output = net(x, x_len, input, input_len)
    # for i in range(forecast_window):
    #     context = x.slice_axis(begin=0, end=-forecast_window, axis=-1)
    #     observed_values = x.slice_axis(begin=-forecast_window + i, end=None, axis=-1)
    #     observed_values = nd.concat(observed_values, preds[:, 1, :].expand_dims(1), dim=-1) if preds is not None else observed_values
    #
    #     _, input_len = pad_input(observed_values, forecast_window)
    #
    #     output = net(context, past_len, observed_values, input_len)
    #     pred = output.slice_axis(begin=-1, end=None, axis=-1)
    #     preds = pred if preds is None else nd.concat(preds, pred, dim=-1)

    return output


def build_input(forecast_window, *tensors):
    raw_input = nd.concat(*tensors, dim=2)
    return pad_input(raw_input, forecast_window)


def pad_input(data, max_size):
    output_len = nd.full((data.shape[0], data.shape[1]), data.shape[2], ctx=data.context)
    output = pad_output(data, max_size)
    return output, output_len


def pad_output(output, max_size):
    pad_width = (0, 0, 0, 0, 0, 0, 0, max_size - output.shape[2])
    return output.expand_dims(0).pad('constant', pad_width).squeeze(axis=0)
