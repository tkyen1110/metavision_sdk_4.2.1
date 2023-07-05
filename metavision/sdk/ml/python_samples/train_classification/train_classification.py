# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Classification Training Script

Instanciates and fits the classification Pytorch Lightning Module.
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

try:
    import mlflow
    from pytorch_lightning.loggers import MLFlowLogger
    use_mlflow_logger = True
except ImportError:
    use_mlflow_logger = False

from pytorch_lightning.callbacks.progress import TQDMProgressBar

from metavision_ml.classification import get_model_names, get_model_class, get_data_module, is_rnn
from metavision_ml.detection.data_factory import get_classes_from_label_map_rnn, get_classes_from_label_map_fnn
from metavision_ml.utils.main_tools import search_latest_checkpoint, infer_preprocessing
from metavision_ml.data.label_loading import get_label_backward_map_dict, get_label_forward_map_dict

import os
import glob
import h5py
import warnings
import argparse
from types import SimpleNamespace


def autocomplete_params(params: argparse.Namespace):
    """
    Fills in the blank of the config,
    for default configuration.

    Args:
        params: argparse.Namespace

    Returns:
        params: argparse.Namespace (completed)
    """
    params = SimpleNamespace(**vars(params))

    if is_rnn(params.models):
        all_classes = get_classes_from_label_map_rnn(params.dataset_path)
        if not params.classes:
            params.classes = all_classes
        else:
            c1 = set(params.classes)
            c2 = set(all_classes)
            inter = c1.difference(c2)
            assert len(inter) == 0, "some classes are not part of this dataset: " + str(inter) + " from: " + str(c2)
    else:
        assert len(params.classes) == 0, "The classes shouldn't be provided in the case of training feedforward "
        "networks. All the classes in the .json file will be used."
        params.classes = get_classes_from_label_map_fnn(params.dataset_path)
        # put it here
        # label maps
        label_map_path = os.path.join(params.dataset_path, 'label_map_dictionary_fnn.json')
        assert os.path.isfile(label_map_path), f"label file {label_map_path} not found!!!"
        backward_label_dict = get_label_backward_map_dict(label_map_path)
        assert "ignore" in backward_label_dict.keys(), "The labels should include 'ignore'!!!"
        assert backward_label_dict["ignore"] == 255, "The key of the 'ignore' should be set as 255!!!"
        # will be used for demo video
        # self.forward_label_dict = get_label_forward_map_dict(label_map_path)
        params.forward_label_dict = get_label_forward_map_dict(label_map_path)
        params.backward_label_dict = backward_label_dict

    preprocess_dim, preprocess, delta_t, mode, n_events, preprocess_kwargs = infer_preprocessing(params)

    if preprocess_dim is None:
        preprocess_dim, preprocess, delta_t = (6, 240, 320), "event_cube", params.delta_t

    params.preprocess_channels = preprocess_dim[0]
    if params.height_width is None:
        params.height = preprocess_dim[-2]
        params.width = preprocess_dim[-1]
    else:
        assert len(params.height_width) == 2
        params.height, params.width = params.height_width
    params.delta_t = delta_t
    params.preprocess = preprocess
    params.preprocess_kwargs = preprocess_kwargs

    params = argparse.Namespace(**params.__dict__)
    if params.cpu and params.precision != 32:
        print('Warning: in cpu mode precision must be 32, overriding')
        params.precision = 32

    if params.just_demo or params.just_test or params.just_val:
        params.resume = True

    if not is_rnn(params.models):
        assert params.precision > 16, "FFN training pipeline only supports precisions higher than 16."

    return params


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=False,
    mode='min'
)


class TrainCallback(pl.callbacks.Callback):
    def __init__(self, data_module, demo_every, show_video=False):
        super().__init__()
        self.data_module = data_module
        self.demo_every = demo_every
        self.show_video = show_video

    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(trainer.train_dataloader, "dataset") and is_rnn(pl_module.hparams.models):
            print("Shuffling train dataset")
            trainer.train_dataloader.loaders.dataset.shuffle()

        if trainer.current_epoch == 0:
            pl_module.demo_video(
                self.data_module.train_dataloader(),
                epoch=trainer.current_epoch,
                num_batches=50,
                show_video=self.show_video,
                show_pred=False,
                fps=5)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.current_epoch % self.demo_every:
            pl_module.demo_video(
                self.data_module.val_dataloader(),
                epoch=trainer.current_epoch,
                num_batches=50,
                show_video=False)


def train_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dir params
    parser.add_argument('root_dir', type=str, default='train_log', help='logging directory')
    parser.add_argument('dataset_path', type=str, default='toy_problem', help='path')

    # train params
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_scheduler_step_gamma', type=float, default=None,
                        help='learning rate scheduler step param (disabled if None)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--delta-t', type=int, default=10000, help="timeslice duration in us")
    parser.add_argument('--wd', type=float, default=1e-05, help='weight decay')
    parser.add_argument('-w', '--weights', nargs='+', default=[], type=float, help='list of weights for each class')
    parser.add_argument('--val_every', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--demo_every', type=int, default=2, help='run demo every X epoch')
    parser.add_argument('--save_every', type=int, default=1, help='save checkpoint every X epochs')
    parser.add_argument('--max_epochs', type=int, default=200, help='train for X epochs')
    parser.add_argument('--just_val', action='store_true', help='run validation on val set')
    parser.add_argument('--just_test', action='store_true', help='run validation on test set')
    parser.add_argument('--just_demo', action='store_true', help='run demo video with trained network')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default='', help='resume from specific checkpoint')
    parser.add_argument('--precision', type=int, default=32, help='mixed precision training default to float16')

    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='limit train batches to fraction of dataset')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='limit val batches to fraction of dataset')
    parser.add_argument('--limit_test_batches', type=float, default=1.0,
                        help='limit test batches to fraction of dataset')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='accumulate gradient for more than a single batch')
    parser.add_argument('--fast_dev_run', action='store_true', help='run a single batch')
    parser.add_argument('--finetune', action='store_true', help='finetune from checkpoint')

    # data params
    parser.add_argument('--height_width', nargs=2, default=None, type=int,
                        help="if set, downsize the feature tensor to the corresponding resolution using interpolation")
    parser.add_argument('--num_tbins', type=int, default=1, help="timesteps per batch for truncated backprop")
    parser.add_argument('--num_ev_reps', type=int, default=1, help="number of event frames to form a tensor"
                                                                   "as input for a single prediction.")
    parser.add_argument('--classes', nargs='+', default=[], help='subset of classes to use')
    parser.add_argument('--num_workers', type=int, default=2, help='number of threads')
    parser.add_argument('--skip_us', type=int, default=0, help='skip this amount of microseconds')
    parser.add_argument('--label_delta_t', type=int, default=10000, help='delta_t of annotation in (us)')
    parser.add_argument("--use_nonzero_labels", dest="use_label_freq", action="store_false",
                        help="if set, only use labels which are non-zero")
    parser.add_argument("--disable_data_augmentation", dest="use_data_augmentation", action="store_false",
                        help="Disable data augmentation during training")
    parser.add_argument("--train_plus_val", action="store_true",
                        help="if set, train using train+val, test on test")
    parser.add_argument("--shuffle", action="store_true",
                        help="shuffle the input dataset")

    # model params
    parser.add_argument('--models', default='ConvRNNClassifier', type=str, choices=get_model_names(),
                        help='model architecture type')
    parser.add_argument('--feature_base', type=int, default=16, help='no. of base features for RNN type of model')
    parser.add_argument('--width_mul', type=float, default=1.,
                        help='growth factor of feature extractor for MobileNet2')
    parser.add_argument('--feature_channels_out', type=int, default=128, help='number of channels per feature-map')

    # display params
    parser.add_argument("--no_window", action="store_true",
                        help="Disable output window during demo (only write a video)")
    parser.add_argument("--show_plots", nargs='+', default=['cm', 'pr', 'roc'], choices=['cm', 'pr', 'roc'],
                        help="select evaluation plots on test datasets")
    parser.add_argument("--inspect_result", action="store_true",
                        help="Inspect sequential results together with their recording frames")

    parser.add_argument('--experiment_name', type=str, default='train_ref', help='name of the experiment')
    parser.add_argument('--allow_labels_interpolation', action="store_true", help='interpolate labels if label_delta_t too small compared to frame delta_t')

    return parser


def compare_hparams(params, hparams):
    """
    Compare the params and verify if they give the same values for the same key

    Args:
        params (Namespace): parameters given as input
        hparams (Namespace): parameters saved during training
    """
    params, hparams = vars(params), vars(hparams)
    set1 = set(params.keys())
    set2 = set(hparams.keys())
    common_keys = set1.intersection(set2)
    diff_keys = []
    for key in common_keys:
        if params[key] != hparams[key] and key != 'batch_size':
            diff_keys.append(key)
            assert key not in ["batch_size", "num_tbins", "width", "height"], \
                "input '{}': {} is different from the loaded checkpoint, which is {}".format(key, params[key],
                                                                                             hparams[key])
    if diff_keys:
        warnings.warn("Note that the following input parameters are inconsistent with the trained model: {}"
                      .format(diff_keys))

    hparams.update(params)
    return argparse.Namespace(**hparams)


def check_params(params):

    #check if dataset is compued with n_events mode, in this case only feedforward models can be used
    train_files = glob.glob(os.path.join(params.dataset_path, "train", '*.h5'))
    assert len(train_files) > 0
    with h5py.File(train_files[0], "r") as f:
        precomputed_dataset_mode =  f["data"].attrs.get("mode", "delta_t")
        assert precomputed_dataset_mode in ["delta_t", "n_events"], "only n_events and delta_t mode are supported."
        if precomputed_dataset_mode == "n_events":
            assert not is_rnn(params.models), "only feed forward models can be trained in n_events mode"
            print("Training in n_events mode!")
        else:
            if not params.allow_labels_interpolation:
                assert params.label_delta_t <= f["data"].attrs["delta_t"], "label frequency smaller than frame frequency! \
                             Consider using option --allow_labels_interpolation." 

def train(params: argparse.Namespace):
    """Using Pytorch Lightning to train our model

    This trainer trains a Recurrent or a Feed Forward Classifier on your data.
    Every X epochs it creates a demonstration video and computes
    Accuracy metrics.

    Otherwise, you need to indicate a valid dataset path with format compatible
    with metavision_ml/data/sequential_dataset.py (.h5 or .dat files)

    You can visualize logs with tensorboard:

    %tensorboard --log_dir=.
    """
    model_class = get_model_class(params.models)
    model = model_class(params)
    data_module = get_data_module(params.models)
    classif_data = data_module(params, data_dir=params.dataset_path)

    #check params compatbility
    check_params(params)
    gpu_nums = 0 if params.cpu else 1

    if params.finetune:
        assert params.checkpoint is not None
        model.load_pretrained(params.checkpoint)
        ckpt = None
    elif params.checkpoint != "":
        ckpt = params.checkpoint
    elif params.resume:
        ckpt = search_latest_checkpoint(params.root_dir)
    else:
        ckpt = None

    tmpdir = os.path.join(params.root_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, every_n_epochs=params.save_every)
    tqdm_callback = TQDMProgressBar(refresh_rate=20)

    tb_logger = TensorBoardLogger(save_dir=os.path.join(params.root_dir, 'logs'))

    if use_mlflow_logger:
        mlflow_logger = MLFlowLogger(tracking_uri="file:{}".format(os.path.join(params.root_dir, 'mlruns')),
                                     experiment_name=params.experiment_name)

    if ckpt is not None and params.just_demo or params.just_val or params.just_test:
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu') if params.cpu else torch.device("cuda"))
        hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
        hparams = compare_hparams(params, hparams)  # merge two params
        model_class = get_model_class(params.models)
        model = model_class(params)
        model.load_state_dict(checkpoint['state_dict'])

    if params.just_demo:
        if not params.cpu:
            model = model.cuda()
        model.demo_video(classif_data.test_dataloader(), -1, num_batches=100, show_video=not params.no_window)
    elif params.just_val:
        pl.Trainer(gpus=gpu_nums, precision=params.precision, limit_test_batches=params.limit_val_batches).test(
            model, test_dataloaders=classif_data.val_dataloader())
    elif params.just_test:
        pl.Trainer(gpus=gpu_nums, precision=params.precision, limit_test_batches=params.limit_test_batches).test(
            model, test_dataloaders=classif_data.test_dataloader())
    else:

        loggers = [tb_logger]
        if use_mlflow_logger:
            loggers.append(mlflow_logger)
        trainer = pl.Trainer(
            default_root_dir=params.root_dir, logger=loggers,
            callbacks=[checkpoint_callback,
                       early_stop_callback,
                       TrainCallback(classif_data,
                                     params.demo_every,
                                     show_video=not params.no_window),
                       tqdm_callback],
            enable_checkpointing=True,
            gpus=gpu_nums,
            precision=params.precision,
            max_epochs=params.max_epochs,
            check_val_every_n_epoch=params.val_every,
            resume_from_checkpoint=ckpt,
            accumulate_grad_batches=params.accumulate_grad_batches,
            log_every_n_steps=1,
            fast_dev_run=params.fast_dev_run,
            # Fast Run
            limit_train_batches=params.limit_train_batches,
            limit_val_batches=params.limit_val_batches,
            limit_test_batches=params.limit_test_batches)
        trainer.fit(model, classif_data)


if __name__ == '__main__':
    parser = train_parser()
    params = parser.parse_args()
    params = autocomplete_params(params)
    train(params)
