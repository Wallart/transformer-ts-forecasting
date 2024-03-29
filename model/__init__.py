from mxnet import gluon, MXNetError
from model.layers.output_projectors import projectors
from model.transformer import Transformer

import os
import logging
import mxnet as mx


def get_model(args, ctx, model_class, model_path=None, symbol_path=None):
    nn = None
    if model_path is not None:
        model_path = os.path.expanduser(model_path)

        try:
            symbol_path = symbol_path or get_symbol(model_path)
            symbol_path = os.path.expanduser(symbol_path)

            nn = gluon.nn.SymbolBlock.imports(symbol_path, ['data'], model_path)
            nn.collect_params().reset_ctx(ctx=ctx)
        except Exception as e:
            logging.info('{}. Trying to load model as a checkpoint'.format(e))

        if nn is None:
            try:
                nn = compose_model(args, model_class)
                nn.load_parameters(model_path, ctx=ctx)
            except MXNetError as _:
                logging.error('Cannot load model. Invalid file.')
                exit(1)
    else:
        nn = compose_model(args, model_class)
        nn.initialize(ctx=ctx)

    if not args.no_hybridize:
        nn.hybridize()

    return nn


def compose_model(args, model_class):
    logging.info('Output projector: {}.'.format(args.output_projector))
    projector = projectors[args.output_projector]()
    nn = model_class(args, projector)
    return nn


def get_symbol(weights_path):
    weights_file = os.path.basename(weights_path)
    symbol_file = '{}-symbol.json'.format(weights_file.split('-')[0])
    symbol_path = os.path.join(os.path.dirname(weights_path), symbol_file)
    if not os.path.isfile(symbol_path):
        raise Exception('Cannot find symbol file')

    return symbol_path


def get_ctx(args):
    try:
        devices_id = [int(i) for i in args.gpus.split(',') if i.strip()]
        if len(devices_id) == 0:
            devices_id = mx.test_utils.list_gpus()

        ctx = [mx.gpu(i) for i in devices_id if i >= 0]
        ctx = ctx if len(ctx) > 0 else [mx.cpu()]
    except Exception:
        logging.error('Cannot access GPU.')
        ctx = [mx.cpu()]

    args.batch_per_gpu = args.batch_size // len(ctx)
    logging.info('Used context: {}'.format(', '.join([str(x) for x in ctx])))
    return ctx

