#!/usr/bin/env python
from dataset import *
from model import get_ctx, get_model, Transformer, projectors
from trainer.module_trainer import ModuleTrainer
from utils.logger import Logger
from trainer.transformer_trainer import TransformerTrainer
from predictor.predictor import Predictor

import os
import sys
import shutil
import argparse


def memonger(args, net):
    from mxnet import sym
    from utils import memonger

    dshapes = {
        'x': (args.batch_size, 1, args.enc_data_size),
        'x_len': (args.batch_size, 1),
        'dec_input': (args.batch_size, 1, args.forecast_window),
        'y_len': (args.batch_size, 1)
    }
    graph = net(sym.Variable('x'), sym.Variable('x_len'), sym.Variable('dec_input'), sym.Variable('y_len'))

    net_mem_planned = memonger.search_plan(graph, **dshapes)
    old_cost = memonger.get_cost(graph, **dshapes)
    new_cost = memonger.get_cost(net_mem_planned, **dshapes)

    print('Old feature map cost={} MB'.format(old_cost))
    print('New feature map cost={} MB'.format(new_cost))


def prepare_outdir(opts):
    if hasattr(opts, 'outdir') and opts.outdir:
        opts.outdir = os.path.expanduser(opts.outdir)
        outdir_exists = os.path.isdir(opts.outdir)
        if outdir_exists and not opts.overwrite:
            raise Exception('Output directory already exists.')
        elif outdir_exists and opts.overwrite:
            shutil.rmtree(opts.outdir)

        os.makedirs(opts.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformers playground')
    sub_parsers = parser.add_subparsers(dest='action')

    ##-- TRAIN --##
    train_parser = sub_parsers.add_parser('train')
    train_parser.add_argument('dataset_type', choices=datasets.keys(), type=str, help='training dataset type')
    train_parser.add_argument('dataset', type=str, help='path of the train dataset')
    train_parser.add_argument('forecast_window', type=int, help='forecasting window')
    train_parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=32, help='batch size')
    train_parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=200, help='learning epochs')
    train_parser.add_argument('-o', '--output', dest='outdir', type=str, help='model output directory')
    train_parser.add_argument('--overwrite', action='store_true', help='overwrite output directory if it already exists')
    train_parser.add_argument('--gpus', type=str, default='', help='gpus id to use, for example 0,1, etc. -1 to use cpu')
    train_parser.add_argument('--val-split', type=float, default=0.2, help='percentage of dataset to reuse for validation')

    # Adam optimizer params
    train_parser.add_argument('-b1', '--beta1', type=float, default=0.9, help='beta1 value for Adam optimizer')
    train_parser.add_argument('-b2', '--beta2', type=float, default=0.98, help='beta2 value for Adam optimizer')
    train_parser.add_argument('--eps', type=float, default=1e-9, help='epsilon value for Adam optimizer')
    # Model pre-trained weights
    train_parser.add_argument('--pre-trained', dest='model', type=str, help='path of the pre-trained weights.')
    train_parser.add_argument('-s', '--symbol', type=str, help='symbol file path')
    # Trainer interval opts
    train_parser.add_argument('--log-interval', type=int, default=5, help='iterations log interval')
    train_parser.add_argument('--chkpt-interval', type=int, help='model checkpointing interval (epochs)')
    train_parser.add_argument('--val-interval', type=int, default=10, help='model validation interval (epochs)')
    train_parser.add_argument('--viz-interval', type=int, default=2, help='model visualization interval (epochs)')
    train_parser.add_argument('--thumb-interval', type=int, default=2, help='thumbnail generation interval (epochs)')
    # Debug opts
    train_parser.add_argument('--profile', action='store_true', help='enable profiling')
    train_parser.add_argument('--no-hybridize', action='store_true', help='disable mxnet hybridize network (debug purpose)')
    # Model hyperparams
    train_parser.add_argument('--dim-model', default=512, type=int, help='model dimension')
    train_parser.add_argument('--dim-ff', default=2048, type=int, help='model hidden size')
    train_parser.add_argument('--dropout-rate', default=.1, type=float, help='model dropout rate')
    train_parser.add_argument('--num-heads', default=8, type=int, help='model num heads')
    train_parser.add_argument('--num-layers', default=6, type=int, help='model num layers')
    train_parser.add_argument('--projector', dest='output_projector', default='quantile', type=str, choices=projectors, help='network output projector')

    #-- FORECASTING --##
    forecast_parser = sub_parsers.add_parser('forecast')
    forecast_parser.add_argument('dataset_type', choices=datasets.keys(), type=str, help='training dataset type')
    forecast_parser.add_argument('dataset', type=str, help='path of the train dataset')
    forecast_parser.add_argument('model', type=str, help='path of the pre-trained weights.')
    forecast_parser.add_argument('forecast_window', type=int, help='forecasting window')
    forecast_parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=32, help='batch size')
    forecast_parser.add_argument('-t', '--threshold', type=float, default=.9, help='anomaly threshold [0, 1]')
    forecast_parser.add_argument('-o', '--outfile', type=str, help='predictions output path')
    forecast_parser.add_argument('--gpus', type=str, default='', help='gpus id to use, for example 0,1, etc. -1 to use cpu')
    forecast_parser.add_argument('--val-split', type=float, default=0.2, help='percentage of dataset to reuse for validation')
    # Model symbol path
    forecast_parser.add_argument('-s', '--symbol', type=str, help='symbol file path')
    # Debug opts
    forecast_parser.add_argument('--no-hybridize', action='store_true', help='disable mxnet hybridize network (debug purpose)')
    # Model hyperparams
    forecast_parser.add_argument('--dim-model', default=512, type=int, help='model dimension')
    forecast_parser.add_argument('--dim-ff', default=2048, type=int, help='model hidden size')
    forecast_parser.add_argument('--dropout-rate', default=.1, type=float, help='model dropout rate')
    forecast_parser.add_argument('--num-heads', default=8, type=int, help='model num heads')
    forecast_parser.add_argument('--num-layers', default=6, type=int, help='model num layers')
    forecast_parser.add_argument('--projector', dest='output_projector', default='quantile', type=str, choices=projectors, help='network output projector')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        exit(1)

    args = parser.parse_args()
    prepare_outdir(args)
    logger = Logger(args)

    ctx = get_ctx(args)
    dataset = get_dataset(args.dataset_type, args.dataset)

    if args.action == 'train':
        conf = get_walk_forward_dataloaders(dataset, args.batch_size, args.forecast_window, args.val_split)
        train_data, val_data, x_size, y_size, num_feats = conf

        args.enc_data_size = x_size
        args.dec_data_size = y_size
        args.num_features = num_feats

        net = get_model(args, ctx, Transformer, model_path=args.model, symbol_path=args.symbol)

        trainer = TransformerTrainer(args, net, ctx)
        #trainer = ModuleTrainer(args, net, ctx)
        trainer.train(train_data, val_data)

    elif args.action == 'forecast':
        data, x_size, num_feats = get_dataloader(dataset, args.batch_size, for_training=False)
        # WARNING Don't use for_training because of shuffle

        args.enc_data_size = x_size - args.forecast_window
        args.dec_data_size = args.forecast_window
        args.num_features = num_feats

        net = get_model(args, ctx, Transformer, model_path=args.model, symbol_path=args.symbol)

        predictor = Predictor(args, net, ctx)
        predictor.forecast(data)
