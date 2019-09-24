from dataset import *

import os
import scipy
import argparse
import numpy as np


def prepare_outdir(outdir, overwrite=True):
    outdir = os.path.expanduser(outdir)

    if os.path.isdir(outdir) and overwrite:
        shutil.rmtree(outdir, ignore_errors=True)
    elif os.path.isdir(outdir) and not overwrite:
        print('ERROR: {} already exists.'.format(outdir))

    os.makedirs(outdir, exist_ok=True)
    return outdir


def serie_envelope(fig, serie, amplitude_envelope):
    ax0 = fig.add_subplot(221)
    ax0.plot(serie, label='signal')
    ax0.plot(amplitude_envelope, label='envelope')
    ax0.set(xlabel='time')
    ax0.legend()


def instant_freq(fig, instantaneous_frequency):
    ax1 = fig.add_subplot(223)
    ax1.plot(instantaneous_frequency)
    ax1.set(xlabel='time', ylabel='frequency')


def power_spectral_density(fig, f, Pxx_den):
    ax2 = fig.add_subplot(222)
    ax2.semilogy(f, Pxx_den)
    ax2.set(xlabel='frequency [Hz]', ylabel='PSD [V**2/Hz]')


def linear_spectrum(fig, f, Pxx_spec):
    ax3 = fig.add_subplot(224)
    ax3.semilogy(f, np.sqrt(Pxx_spec))
    ax3.set(xlabel='frequency [Hz]', ylabel='Linear spectrum [V RMS]')


def explore(fs, outdir, overwrite):
    outdir = prepare_outdir(outdir, overwrite)

    for i in range(0, len(dataset)):
        serie = dataset[i].squeeze().asnumpy()
        analytic_signal = scipy.signal.hilbert(serie)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
        f1, Pxx_den = scipy.signal.periodogram(serie, fs)
        f2, Pxx_spec = scipy.signal.periodogram(serie, fs, 'flattop', scaling='spectrum')

        fig = plt.figure(figsize=(20, 6))
        fig.suptitle('sample {}'.format(i))

        serie_envelope(fig, serie, amplitude_envelope)
        instant_freq(fig, instantaneous_frequency)
        power_spectral_density(fig, f1, Pxx_den)
        linear_spectrum(fig, f2, Pxx_spec)

        fig.savefig(os.path.join(outdir, 'sample-{}.png'.format(i)))
        plt.close(fig)
        print('Sample {} saved.'.format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset explorer')
    parser.add_argument('dataset_type', choices=datasets.keys(), type=str, help='dataset type')
    parser.add_argument('dataset', type=str, help='path of the dataset')
    parser.add_argument('sampling_rate', type=int, help='data sampling rate')
    parser.add_argument('outdir', type=str, help='output directory')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite output directory if it already exists')

    args = parser.parse_args()

    dataset = get_dataset(args.dataset_type, os.path.expanduser(args.dataset))
    explore(args.sampling_rate, args.outdir, args.overwrite)
