# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Flow Training Script

Instantiates and fits the flow Pytorch Lightning Module.
"""
import os
import numpy as np
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


from metavision_ml.flow.lightning_model import FlowModel
from metavision_ml.flow.data_module import FlowDataModule
from metavision_ml.flow.feature_extractor import AVAILABLES_ARCHS
from metavision_ml.flow.callbacks import FlowCallback
from metavision_ml.utils.main_tools import search_latest_checkpoint, infer_preprocessing

torch.manual_seed(0)
np.random.seed(0)


def _parse_args(argv=None):
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Trains a model for flow regression.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('output_dir', default='train_log', help='directory where logs, checkpoint and demo videos '
                        'will be be saved')
    parser.add_argument('dataset_path', help='Path of the directory containing the train, test and val split'
                        'folders.')

    # train params
    parser.add_argument('--fast-dev-run', action='store_true', help='run one batch for testing purposes.')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--accumulate-grad-batches', type=int, default=4, help='Average n batches gradient before '
                        'performing backward pass. This allows to use virtually bigger batches without overloading '
                        'gpu memory')

    parser.add_argument('--val-every', type=int, default=2, help='validate every X epochs')

    parser.add_argument('--demo-every', type=int, default=2,
                        help='run demo, i.e. produces a video with flow visualization on the test set every X epochs.')

    parser.add_argument(
        '--show-flow-everywhere', dest="mask_by_input", action="store_false",
        help="if set this will "
             "show flow arrows everywhere and not just when there are input events.")
    parser.add_argument('--save-every', type=int, default=1, help='Saves a checkpoint every X epochs.')
    parser.add_argument('--max-epochs', type=int, default=20, help='Number of epochs in training.')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default='', help='Resume from specific checkpoint')
    parser.add_argument('--precision', type=int, default=16, help='Mixed precision training.')
    parser.add_argument('--cpu', action='store_true', help='Uses cpu, changes precision to float32.')
    parser.add_argument('--just-val', action='store_true', help='run validation on val set')
    parser.add_argument('--just-demo', action='store_true', help='run demo video with trained network')
    parser.add_argument('--limit-train-batches', type=float, default=1.0,
                        help='if set between 0 and 1, limits the number of train batches to fraction of dataset.'
                             ' If set to a positive integer superior to 1, it sets explicitly the number of batches.')
    parser.add_argument('--limit-val-batches', type=float, default=1.0,
                        help='if set between 0 and 1, limits the number of validation batches to fraction of dataset.'
                        ' If set to a positive integer superior to 1, it sets explicitly the number of batches.')
    parser.add_argument('--limit-test-batches', type=float, default=1.0,
                        help='if set between 0 and 1, limits the number of test batches to fraction of dataset.'
                        ' If set to a positive integer superior to 1, it sets explicitly the number of batches.')

    # demo params
    parser.add_argument('--no-window', action='store_true', help='just-demo do not show video')
    parser.add_argument('--demo-num-batches', type=int, default=100, help='just-demo num batches')
    parser.add_argument(
        '--draw-dense',
        action='store_true',
        help='draws the flow in dense format, otherwise we draw arrows')

    # data params
    parser.add_argument('--height-width', nargs=2, default=None, type=int,
                        help="if set downsize the feature tensor to the corresponding resolution using interpolation")
    parser.add_argument('--num-tbins', type=int, default=4, help="timesteps per batch for truncated backprop")
    parser.add_argument('--num-workers', type=int, default=2, help='number of processes using for the data loading.')

    parser.add_argument('--no-data-aug', dest="data_aug", action="store_false",
                        help='flag to remove any data augmentation')
    parser.add_argument('--val-dataset-path', default="", help='Path of the directory containing Labeled flow data'
                                                               ' as DAT events files and HDF5 flow files.')

    # model params
    parser.add_argument('--feature-extractor', type=str, default='eminet',
                        choices=AVAILABLES_ARCHS, help='feature extractor name to use')
    parser.add_argument('--depth', type=int, default=1,
                        help='depth parameter in the feature extractor if available')
    parser.add_argument('--feature-base', type=int, default=16,
                        help='Multiplier for the number of filter at each layer')
    parser.add_argument('--rnn-cell', type=str, default='lstm',
                        choices=("lstm", "gru"), help='type of cell used for the convolutioanl RNN')

    return parser.parse_args(argv) if argv is not None else parser.parse_args()


def main(params):
    """This trainer trains a flow regression network on event-based data."""
    train_dir = os.path.join(params.dataset_path, "train")
    assert os.path.isdir(train_dir), "dataset_path must contain a train folder!"

    params.preprocess_dim, params.preprocess, params.delta_t, _, _, params.preprocess_kwargs = infer_preprocessing(params)
    params.array_dim = (params.num_tbins, *params.preprocess_dim)

    flow_model = FlowModel(
        delta_t=params.delta_t,
        preprocess=params.preprocess,
        learning_rate=params.lr,
        array_dim=params.array_dim,
        loss_weights={
            "data": 1,
            "smoothness": 0.4,
            "smoothness2": 4 * 0.15,
            "l1": 2 * 0.15,
            'time_consistency': 1,
            "bw_deblur": 1},
        feature_extractor_name=params.feature_extractor,
        network_kwargs={
            "base": params.feature_base,
            "depth": params.depth,
            "separable": True},
        draw_dense=params.draw_dense)
    flow_data = FlowDataModule(params, data_dir=params.dataset_path,
                               test_data_dir=params.val_dataset_path)

    if params.resume:
        ckpt = search_latest_checkpoint(params.output_dir)
    elif params.checkpoint != "":
        ckpt = params.checkpoint
    else:
        ckpt = None

    if ckpt is not None and params.just_demo or params.just_val:
        print('Latest Checkpoint: ', ckpt)
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu') if params.cpu else torch.device("cuda"))
        flow_model.load_state_dict(checkpoint['state_dict'])

    if params.just_demo:
        flow_data.setup()
        log_dir = os.path.join(params.output_dir, "logs")
        if not params.cpu:
            flow_model.cuda()
        flow_model.demo_video(
            flow_data.test_dataloader(),
            log_dir,
            epoch=-1,
            num_batches=params.demo_num_batches,
            write_video=True,
            show_video=not params.no_window)
    elif params.just_val:
        flow_data.setup()
        pl.Trainer(
            gpus=0 if params.cpu else 1,
            precision=32 if params.cpu else params.precision).test(
            flow_model, flow_data.test_dataloader())
    else:
        # Define callbacks
        # save checkpoint every two epochs
        tmpdir = os.path.join(params.output_dir, 'checkpoints')
        checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, every_n_epochs=params.save_every)
        tqdm_callback = TQDMProgressBar(refresh_rate=20)

        callbacks = [FlowCallback(flow_data, 1e9, video_result_every_n_epochs=params.demo_every,
                                  mask_flow_by_input=params.mask_by_input),
                     checkpoint_callback,
                     tqdm_callback]

        logger = TensorBoardLogger(save_dir=os.path.join(params.output_dir, 'logs/'))

        trainer = pl.Trainer(
            default_root_dir=params.output_dir,
            logger=logger,
            enable_checkpointing=True,
            callbacks=callbacks,
            gpus=0 if params.cpu else 1,
            precision=32 if params.cpu else params.precision,
            check_val_every_n_epoch=params.val_every,
            resume_from_checkpoint=ckpt,
            max_epochs=params.max_epochs,
            fast_dev_run=params.fast_dev_run,
            accumulate_grad_batches=params.accumulate_grad_batches,
            limit_train_batches=params.limit_train_batches,
            limit_val_batches=params.limit_val_batches)
        trainer.fit(flow_model, flow_data)


if __name__ == '__main__':
    main(_parse_args())
