from scipy import signal

import numpy as np


def log1p(d, x):
    return np.log1p(x)


def normalize(d, x):
    return (x - d.min()) / (d.max() - d.min())


def standardize(d, x):
    return (x - d.mean()) / d.std()


def spec(d, x):
    _, specs = signal.periodogram(x, fs=1024, nfft=512)
    return specs


def de_meaning(mean):
    def subtract_mean(d, x):
        return x - mean
    return subtract_mean
