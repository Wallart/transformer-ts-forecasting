from mxnet import nd

import random
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt


# *** TRAINING FUNCTIONS *** #

def samples_to_board(writer, tensors, epoch, tag):
    x, y, out = tensors
    x, y, out = agg_dist_tensor(x), agg_dist_tensor(y), agg_dist_tensor(out)

    plots = samples_to_plot_of_plots(x, y, out)

    writer.add_image(image=plots.transpose((0, 3, 1, 2)), global_step=epoch, tag=tag)
    writer.flush()


def samples_to_plot_of_plots(*tensors):
    x, y, out = tensors
    plots = [fig_as_numpy(samples_to_plot(x, y, out)) for x, y, out in zip(x, y, out)]
    plots = np.stack(plots, axis=0)  # stack plots into one big plot
    return plots


def samples_to_plot(*tensors):
    x, y, out = tensors

    fig, ax = new_fig()
    plot(ax, x, 0)
    plot(ax, y, len(x[0]))
    plot(ax, out.slice_axis(begin=1, end=2, axis=0), len(x[0]), color='teal', linestyle='--')
    fill_between(ax, out, len(x[0]), color='teal')

    return fig


def attn_weights_to_board(writer, weights, epoch, tag, layer=-1):
    if type(weights) == list or type(weights) == tuple:
        weights = nd.concat(*weights, dim=0)

    rand_idx = random.randint(0, weights.shape[0] - 1)
    plots = attn_to_plot_of_plots(weights[rand_idx], layer).transpose((2, 0, 1))

    writer.add_image(image=plots, global_step=epoch, tag=tag)
    writer.flush()


def attn_to_plot_of_plots(attn, layer):
    attn = attn[layer]
    num_heads = attn.shape[0]

    fig = plt.figure(figsize=(15, 15))

    for i in range(num_heads):
        ax = fig.add_subplot(2, 4, i + 1)

        head = attn[i]
        ax.matshow(head.asnumpy(), cmap='viridis')
        ax.set_xlabel('head {}'.format(i + 1))

    fig.tight_layout()
    return fig_as_numpy(fig)


# *** GENERIC FUNCTIONS *** #


def agg_dist_tensor(tensor, ctx=mx.cpu()):
    assert type(tensor) == list, 'Tensor is not distributed across contexes'
    return nd.concat(*[t.as_in_context(ctx) for t in tensor], dim=0)


def split_n_squeeze(tensor):
    splitted_tensor = tensor.split(axis=0, num_outputs=tensor.shape[0])
    if type(splitted_tensor) != list:
        splitted_tensor = [splitted_tensor]

    return [s.squeeze() for s in splitted_tensor]


def plot(ax, tensors, begin, color=None, linestyle=None):
    """
    :param ax: matplotlib axes
    :param tensors: NDArray of shape (Channels, Width)
    :param begin: index
    :param color: str
    :param linestyle: str
    """
    tensors = split_n_squeeze(tensors)
    _ = [ax.plot(range(begin, begin + len(t)), t.asnumpy(), color=color, linestyle=linestyle) for t in tensors]


def fill_between(ax, tensors, begin, color=None):
    """
    :param ax: matplotlib axes
    :param tensors: NDArray of shape (Channels, Width)
    :param begin: index
    :param color: str
    """
    t = split_n_squeeze(tensors)
    bottom = t[0].asnumpy()
    top = t[-1].asnumpy()
    ax.fill_between(range(begin, begin + len(t[0])), bottom, top, color=color, alpha='0.5')


def new_fig(y_lim=(-.2, 1.2)):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim(y_lim)
    return fig, ax


def fig_as_numpy(fig):
    fig.canvas.draw()

    graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return graph
