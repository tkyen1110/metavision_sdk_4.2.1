# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Tools common to training main functions.
"""
import os
from glob import glob, iglob

import h5py
from metavision_core.event_io import EventDatReader, EventNpyReader
from metavision_ml.preprocessing import PREPROCESSING_DICT, get_preprocess_kwargs_keys
from metavision_core_ml.utils.train_utils import search_latest_checkpoint


def get_original_size_file(path):
    """Returns the couple (height, width) of a file.

    Args:
        path (string): File path, either a DAT file of an hdf5 file path.
    """
    file_extension = os.path.splitext(path)[1].lower()
    if file_extension in (".dat", ".npy"):
        cls = [EventDatReader, EventNpyReader][file_extension == ".npy"]
        return cls(path).get_size()
    elif file_extension in ('.hdf5', ".h5"):
        with h5py.File(path, "r") as f:
            dset = f["data"]
            if "event_input_width" in dset.attrs:
                return dset.attrs['event_input_height'], dset.attrs['event_input_width']
            height, width = dset.shape[-2:]
            if "downsampling_factor" not in dset.attrs:
                downsampling_factor = dset.attrs.get('shift', 0)
            else:
                downsampling_factor = dset.attrs['downsampling_factor']
            return height << downsampling_factor, width << downsampling_factor
    else:
        raise ValueError("can't measure size of  ", os.path.basename(path))


def check_input_power_2(event_input_height, event_input_width, height=None, width=None):
    """Checks that the provided height and width are indeed a negative power of two of the input.

    Args:
        event_input_height (int): height of the sensor in pixels.
        event_input_width (int): width of the sensor in pixels.
        height (int): desired height of features after rescaling
        width (int): desired width of features after rescaling
    """
    def _check(event_input_dim, dimension, dim_name):
        if height is not None:
            msg = f"{dim_name} {dimension:d} must be a negative power of two of input {dim_name} : {event_input_dim:d}"
            quotient = event_input_dim / dimension
            assert quotient.is_integer(), msg
            n = int(quotient)
            assert (n & (n - 1) == 0) and n != 0, msg
    _check(event_input_height, height, "height")
    _check(event_input_width, width, "width")


def infer_preprocessing(params, h5path=None):
    """
    Infer the preprocessing parameters via reading attributes of first found HDF5 file.

    Args:
        params: struct containing training parameters.
        h5path (string): optional path of an HDF5 file from the dataset,
            its attributes are used to override the preprocessing parameters.
    Returns:
        array_dim (int List): tensor shape of a single item of a batch (num time bins, num channels, height, width)
        preprocess (string): name of the preprocessing used.
        delta_t (int): duration of a temporal bin in us.
        mode (string): mode to generate the HDF5 data ("delta_t" or "n_events")
        n_events (int): number of events in the slice (if mode=="n_events"), 0 otherwise
        preprocess_kwargs : preprocessing args used to generate the HDF5 data 
    """
    # if there is a precomputed HDF5 file it should override certain command line params.
    input_files = os.path.join(params.dataset_path, "**/*.h5")
    if not h5path:
        h5path = next(iglob(input_files, recursive=True))
        if h5path is None:
            raise FileNotFoundError(f"No HDF5 file found in {params.dataset_path}")

    with h5py.File(h5path, "r") as f:
        dset = f['data']
        mode = f["data"].attrs.get("mode", "delta_t")
        n_events = f["data"].attrs.get("n_events", 0)
        delta_t = dset.attrs["delta_t"]
        preprocess_name = dset.attrs["events_to_tensor"].decode("utf-8")
        preprocess_kwargs_keys = get_preprocess_kwargs_keys(preprocess_name)
        preprocess_kwargs = {}
        for preprocess_kwargs_key in preprocess_kwargs_keys:
            if preprocess_kwargs_key == "preprocess_dtype":
                continue
            preprocess_kwargs[preprocess_kwargs_key] = dset.attrs[preprocess_kwargs_key]
        assert preprocess_name in PREPROCESSING_DICT, f"stored preprocessing {preprocess_name} doesn't exist ! (" + ", ".join(
            PREPROCESSING_DICT.keys()) + ")"
        channels = dset.shape[1]
        network_height, network_width = dset.shape[-2:]

        if params.height_width is not None:
            [height, width] = params.height_width
            assert network_height >= height, f"height must be <= {dset.shape[2]}"
            assert network_width >= width, f"width must be <= {dset.shape[3]}"
            network_height, network_width = height, width

    preprocess_dim = (channels, network_height, network_width)
    return preprocess_dim, preprocess_name, delta_t, mode, n_events, preprocess_kwargs
