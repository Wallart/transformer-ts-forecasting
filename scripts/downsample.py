from scipy.signal import decimate

import os
import h5py
import time
import logging
import numpy as np


def hamming_downsample(h5py_input, h5py_output, closest_size):
    downsample(h5py_input, h5py_output, closest_size, 'fir')


def chebyshev_downsample(h5py_input, h5py_output, closest_size):
    downsample(h5py_input, h5py_output, closest_size, 'iir')


def find_factors(serie_size, closest_size):
    cur_serie_size = serie_size
    downsample_factors = []
    while cur_serie_size > closest_size:
        factor = min(13, int(cur_serie_size / closest_size))
        if factor <= 1:
            break
        cur_serie_size /= factor
        downsample_factors.append(factor)

    return downsample_factors


def downsample(h5py_input, h5py_output, closest_size, lowpass_filter):
    t0 = time.time()

    h5py_input = os.path.expanduser(h5py_input)
    h5py_output = os.path.expanduser(h5py_output)

    with h5py.File(h5py_input, 'r') as input_dataset:
        keys = list(input_dataset.keys())
        assert len(keys) == 1, 'Multiple entries in h5py file.'

        input_dataset = input_dataset[keys[0]]
        series = input_dataset['block0_values']

        with h5py.File(h5py_output, 'w') as output_dataset:
            output_dataset = output_dataset.create_group(keys[0])

            num_series = series.shape[0]
            serie_size = series.shape[1]

            downsample_factors = find_factors(serie_size, closest_size)
            data = None
            for i, serie in enumerate(series):
                logging.info('Downsampling serie {}/{}'.format(i + 1, num_series))
                for factor in downsample_factors:
                    serie = decimate(serie, factor, ftype=lowpass_filter)
                serie = serie[np.newaxis, :]
                data = serie if data is None else np.concatenate((data, serie), axis=0)

            output_dataset.create_dataset('block0_values', data=data)
            logging.info('Dataset downsampled from {} to {} in {:.3f} sec(s)'.format(serie_size, data.shape[1], time.time() - t0))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    #hamming_downsample('~/datasets/dftrain.h5', '~/datasets/dftrain-downsampled-hamming.h5', 300)
    chebyshev_downsample('~/datasets/dftrain.h5', '~/datasets/dftrain-chebyshev.h5', 300)
