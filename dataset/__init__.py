from mxnet import nd
from mxnet.gluon import data
from dataset.transforms import *
from dataset.dataset_builder.walk_forward_builder import WalkForwardBuilder
from dataset.unsupervised.splitter_dataset import SplitterDataset
from dataset.unsupervised.synthetic_dataset import SyntheticDataset

import os
import shutil
import random
import matplotlib.pyplot as plt

datasets = {
    'synth': SyntheticDataset
}


def plot_one_file(dataset, outdir, samples=10, serie_idx=0, overwrite=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_ylim((dataset.min(), dataset.max()))

    for _ in range(samples):
        idx = random.randint(0, len(dataset) - 1)
        print('INFO: plotting serie {}'.format(idx))

        serie = dataset[idx][serie_idx].squeeze()
        ax.plot(serie.asnumpy())

    outdir = os.path.expanduser(outdir)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    graph_path = os.path.join(outdir, '{}_{}-samples.png'.format(dataset.__class__.__name__, samples))
    if os.path.isfile(graph_path) and overwrite:
        print('WARNING: overwriting previous file.')
        os.remove(graph_path)
    elif os.path.isfile(graph_path) and not overwrite:
        print('ERROR: {} already exists.'.format(graph_path))
        exit(1)

    fig.savefig(graph_path)
    ax.cla()

    print('INFO: plot saved at: {}'.format(graph_path))


def plot_multi_files(dataset, outdir, samples=10, serie_idx=0, overwrite=False):
    outdir = os.path.expanduser(outdir)

    if os.path.isdir(outdir) and overwrite:
        shutil.rmtree(outdir, ignore_errors=True)
    elif os.path.isdir(outdir) and not overwrite:
        print('ERROR: {} already exists.'.format(outdir))

    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for _ in range(samples):
        ax.cla()
        #ax.set_ylim((dataset.min(), dataset.max()))

        idx = random.randint(0, len(dataset) - 1)
        print('INFO: plotting serie {}'.format(idx))

        serie = dataset[idx][serie_idx].squeeze()
        ax.plot(serie.asnumpy())

        graph_path = os.path.join(outdir, '{}_{}-sample-{}.png'.format(dataset.__class__.__name__, samples, idx))
        fig.savefig(graph_path)

    print('INFO: plots saved at: {}'.format(outdir))


def collate_fn_gt(batch):
    x, y = map(list, zip(*batch))

    x_lengths, y_lengths = [e.shape[1] for e in x], [e.shape[1] for e in y]
    # TODO Apply padding
    #x_max_len, y_max_len = max(x_lengths), max(y_lengths)

    return (nd.stack(*x), nd.array(x_lengths).expand_dims(-1)), (nd.stack(*y), nd.array(y_lengths).expand_dims(-1))


def collate_fn(x):
    x_lengths = [e.shape[1] for e in x]
    return nd.stack(*x), nd.array(x_lengths).expand_dims(-1)


def get_opts(batch_size):
    return {
        'batch_size': batch_size,
        'last_batch': 'discard',
        'thread_pool': True
    }


def get_walk_forward_dataloaders(dataset, batch_size, forecast_window, val_split, for_training=True):
    assert len(dataset) >= batch_size, 'Dataset is smaller than batch size'

    # TODO Refactor
    dataset.add_transform(lambda d, x: x.clip(max=np.percentile(x, 95)), lazy=False)
    dataset.add_transform(lambda d, x: x.clip(min=np.percentile(x, 5)), lazy=False)
    dataset.add_transform(normalize)

    train_data = SplitterDataset(dataset, forecast_window)
    val_data = train_data

    #walker = WalkForwardBuilder(dataset, forecast_window, val_split)
    #train_data, val_data = walker.build()
    x_size, y_size = train_data.width()

    train_dataloader = data.DataLoader(train_data, shuffle=for_training, batchify_fn=collate_fn_gt, **get_opts(batch_size))
    val_dataloader = data.DataLoader(val_data, shuffle=False, batchify_fn=collate_fn_gt, **get_opts(batch_size))
    return train_dataloader, val_dataloader, x_size, y_size, train_data.num_features()


def get_dataloader(dataset, batch_size, for_training=True):
    assert len(dataset) >= batch_size, 'Dataset is smaller than batch size'

    # TODO Refactor
    dataset.add_transform(lambda d, x: x.clip(max=np.percentile(x, 95)), lazy=False)
    dataset.add_transform(lambda d, x: x.clip(min=np.percentile(x, 5)), lazy=False)
    dataset.add_transform(normalize)

    dataloader = data.DataLoader(dataset, shuffle=for_training, batchify_fn=collate_fn, **get_opts(batch_size))
    return dataloader, dataset.width(), dataset.num_features()


def get_dataset(dataset_type, path):
    path = os.path.expanduser(path)
    dataset = datasets[dataset_type](path)
    return dataset
