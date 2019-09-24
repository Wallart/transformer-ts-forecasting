import os
import sys
import logging


class Logger:
    def __init__(self, opts):
        self._opts = opts
        self._foreground = os.getpgrp() == os.tcgetpgrp(sys.stdout.fileno())

        conf = {}
        if not self._foreground:
            if hasattr(self._opts, 'outdir') and self._opts.outdir:
                conf['filename'] = os.path.expanduser(os.path.join(self._opts.outdir, 'output.log'))
            else:
                raise Exception('Cannot run in background. No output directory defined.')

        logging.basicConfig(**conf)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)


