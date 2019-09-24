from abc import ABC, abstractmethod
from datetime import datetime
from mxnet import profiler

import os
import time
import json
import logging


class TrainerException(Exception):
    pass


class Trainer(ABC):

    def __init__(self, opts, net):
        self._net = net

        self._opts = opts
        self._epochs = opts.epochs

        self._log_interval = opts.log_interval
        self._chkpt_interval = opts.chkpt_interval
        self._val_interval = opts.val_interval
        self._viz_interval = opts.viz_interval
        self._profile = opts.profile

        self._start_time = 0
        self._val_time = 0
        self._epoch_tick = 0
        self._batch_tick = 0

        self._outdir = opts.outdir or os.path.join(os.getcwd(), '{}-{}e-{}'.format(self._net.model_name(), self._epochs, datetime.now().strftime('%y_%m_%d-%H_%M')))
        self._outlogs = os.path.join(self._outdir, 'logs')
        self._outchkpts = os.path.join(self._outdir, 'checkpoints')
        self._prepare_outdir()

        if self._profile:
            self._outprofile = os.path.join(self._outdir, 'profile.json')
            profiler.set_config(profile_all=True, aggregate_stats=True, filename=self._outprofile)

    def _prepare_outdir(self):
        os.makedirs(self._outlogs)
        os.makedirs(self._outchkpts)

        params_dump = os.path.join(self._outdir, 'parameters_dump.json')
        with open(params_dump, 'w') as f:
            json.dump(vars(self._opts), f, indent=4,  skipkeys=True, default=lambda x: str(x))

    def b_tick(self):
        self._batch_tick = time.time()

    def e_tick(self):
        self._epoch_tick = time.time()

    def t_tick(self):
        self._start_time = time.time()

    def v_tick(self):
        self._val_time = time.time()

    def b_duration(self):
        return time.time() - self._batch_tick

    def e_duration(self):
        return time.time() - self._epoch_tick

    def t_duration(self):
        return time.time() - self._start_time

    def v_duration(self):
        return time.time() - self._val_time

    def _log(self, message, level=logging.INFO):
        logging.log(level, message)

    def _save_profile(self):
        if self._profile:
            print(profiler.dumps())
            profiler.dump()

    @abstractmethod
    def train(self, train_data):
        pass

    @abstractmethod
    def _export_model(self, num_epoch):
        pass

    @abstractmethod
    def _do_checkpoint(self, cur_epoch):
        pass
