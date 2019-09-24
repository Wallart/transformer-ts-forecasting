from dataset.abstract_dataset import AbstractDataset

import numpy as np


class SyntheticDataset(AbstractDataset):

    def __init__(self, *args, t0=192, num_samples=4500, **kwargs):
        super(SyntheticDataset, self).__init__(*args, **kwargs)

        series = []
        for i in range(0, num_samples // 4):
            a_variables = [np.random.uniform(0, 60) for _ in range(0, 3)]
            a_variables.append(max(a_variables[0], a_variables[1]))
            nx = np.random.normal(0, 1)

            parts = list()
            parts.append(a_variables[0] * np.sin(np.pi * np.arange(0, 12) / 6) + 72 + nx)
            parts.append(a_variables[1] * np.sin(np.pi * np.arange(12, 24) / 6) + 72 + nx)
            parts.append(a_variables[2] * np.sin(np.pi * np.arange(24, t0) / 6) + 72 + nx)
            parts.append(a_variables[3] * np.sin(np.pi * np.arange(t0, t0 + 24) / 12) + 72 + nx)
            series.append(np.concatenate(parts))

        self._arr_x = np.stack(series, axis=0)[:, np.newaxis, :]
