# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

from .event_to_tensor import histo, diff, timesurface, event_cube, multi_channel_timesurface, diff_quantized, histo_quantized  # pylint:disable-all
from .viz import viz_histo, viz_diff, viz_timesurface, viz_event_cube_rgb, viz_histo_filtered  # pylint: disable-all
from .viz import viz_histo_binarized, viz_diff_binarized
from .viz import viz_multichannel_timesurface
import numpy as np
from functools import partial
from copy import deepcopy

PREPROCESSING_DICT = {
    'diff': {'cin': 1, 'events_to_tensor': diff,
             'kwargs': {'preprocess_dtype': np.dtype('float32'), 'max_incr_per_pixel': 5},
             "viz": viz_diff_binarized},
    'diff_quantized': {'cin': 1, 'events_to_tensor': diff_quantized,
                       'kwargs': {'preprocess_dtype': np.dtype('int8'), 'negative_bit_length': 8,
                                  'normalization': False},
                       "viz": viz_diff_binarized},
    'histo': {'cin': 2, 'events_to_tensor': histo,
              'kwargs': {'preprocess_dtype': np.dtype('float32'), 'max_incr_per_pixel': 5},
              "viz": viz_histo_binarized},
    'histo_quantized': {'cin': 2, 'events_to_tensor': histo_quantized,
                        'kwargs': {'preprocess_dtype': np.dtype('uint8'), 'negative_bit_length': 4,
                                   'total_bit_length': 8, 'normalization': False},
                        "viz": viz_histo_binarized},
    'timesurface': {'cin': 2, 'events_to_tensor': timesurface,
                    'kwargs': {'preprocess_dtype': np.dtype('float32')},
                    "viz": viz_timesurface},
    'event_cube': {'cin': 6, 'events_to_tensor': event_cube,
                   'kwargs': {'preprocess_dtype': np.dtype('float32'), 'split_polarity': True,
                              'max_incr_per_pixel': 5},
                   "viz": viz_event_cube_rgb},
    'event_cube_paper': {'cin': 10, 'events_to_tensor': event_cube,
                         'kwargs': {'preprocess_dtype': np.dtype('float32'), 'split_polarity': True,
                                    'max_incr_per_pixel': 255./4},
                         "viz": viz_event_cube_rgb},
    'multi_channel_timesurface': {'cin': 6, 'events_to_tensor': multi_channel_timesurface,
                                  'kwargs': {'preprocess_dtype': np.dtype('float32')},
                                  "viz": viz_multichannel_timesurface}
}


def register_new_preprocessing(preproc_name, n_input_channels, function, viz_function,
                               kwargs={'preprocess_dtype': np.dtype('float32'), 'max_incr_per_pixel': 5}):
    """
    Registers a new preprocessing function to be available across the package.

    This must be done each time the python interpreter is invoked.

    Args:
        preproc_name (string): Name of the preprocessing function, has to be unique.
        n_input_channels (int): Number of channels in the resulting tensor.
        function (function): Preprocessing function. Its signature must be
            function(events, tensor, **kwargs) -> None
        viz_function (function): Visualization function. Its signature must be
            viz_function(tensor) -> img (nd.array H x W x 3)
        kwargs (dict): Dictionary of optional arguments to be passed to the function.
    """
    assert preproc_name not in PREPROCESSING_DICT, f"preprocessing {preproc_name} already registered ! (" + ", ".join(
        PREPROCESSING_DICT) + ")"

    PREPROCESSING_DICT[preproc_name] = {'cin': n_input_channels, 'events_to_tensor': function, 'viz': viz_function,
                                        'kwargs': kwargs}


def get_preprocess_dict(preproc_name):
    """Returns a mutable dict describing a preprocessing function.

    preproc_name (string): Name of the preprocessing function, already existing one or previously registered.
    """

    assert preproc_name in PREPROCESSING_DICT, f"preprocessing {preproc_name} does not exist ! (" + ", ".join(
        PREPROCESSING_DICT) + ")"
    return deepcopy(PREPROCESSING_DICT[preproc_name])


def get_preprocess_function_names():
    """Returns the names of all existing and registered preprocessing functions."""
    return list(PREPROCESSING_DICT)


def get_preprocess_kwargs_keys(preproc_name):
    """Returns the kwargs dict of a certain perprocess method."""
    assert preproc_name in PREPROCESSING_DICT, f"preprocessing {preproc_name} does not exist ! (" + ", ".join(
        PREPROCESSING_DICT) + ")"
    return PREPROCESSING_DICT[preproc_name]['kwargs'].keys()


class CDProcessor(object):
    """Wrapper that simplifies the preprocessing.

    Args:
        height: image height
        width: image width
        num_tbins: number of time slices
        preprocessing: name of method, must be registered first
        downsampling_factor (int): Parameter used to reduce the spatial dimension of the obtained feature.
            Actually multiply the coordinates by 2**(-downsampling_factor).
    """

    def __init__(self, height, width, num_tbins=5, preprocessing='histo', downsampling_factor=0,
                 preprocess_kwargs={}):
        self.preprocess_dict = get_preprocess_dict(preprocessing)

        self.preprocess_kwargs = self.preprocess_dict['kwargs']
        self.preprocess_kwargs.update(preprocess_kwargs)
        self.dtype = self.preprocess_kwargs.pop(
            'preprocess_dtype') if "preprocess_dtype" in self.preprocess_kwargs else np.uint8

        if downsampling_factor>0 and "quantized" in preprocessing:
            #we need float 32 when we downsample quantized tensors
            self.dtype = np.dtype('float32')
        in_channels = self.preprocess_dict['cin']
        self.array = np.zeros((num_tbins, in_channels, height, width), self.dtype)

        self.fun = partial(
            self.preprocess_dict['events_to_tensor'], downsampling_factor=downsampling_factor, reset=False,
            **self.preprocess_kwargs)
        self.viz = self.preprocess_dict['viz']

    def __call__(self, events, delta_t=None):
        if delta_t is None:
            delta_t = events['t'][-1]-events['t'][0] if len(events) else 0
        self.array[...] = 0
        self.fun(events, self.array, delta_t)
        return self.array

    def show(self, array):
        return self.viz(array)

    def get_array_dim(self):
        """Returns the shape of the preprocessing tensor.

        Returns:
            shape (int tuple): shape of the tensor to be preprocessed."""
        return self.array.shape

    def get_preprocess_dtype(self):
        return self.array.dtype
