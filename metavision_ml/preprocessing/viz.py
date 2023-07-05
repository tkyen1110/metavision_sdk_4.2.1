# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
A collection of visualization utilities for preprocessings

Examples:
    >>> delta = 100000
    >>> initial_ts = record.current_time
    >>> events = record.load_delta_t(delta)  # load 100 milliseconds worth of events
    >>> events['t'] -= int(initial_ts)  # events timestamp should be reset
    >>> output_array = np.zeros((1, 2, height, width))  # preallocate output array
    >>> histo(events, output_array, delta)
    >>> img = viz_diff("histo")(output_array[0])
    >>> cv2.imshow('img', img)
    >>> cv2.waitKey()
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import cv2
from metavision_core_ml.preprocessing.viz import BG_COLOR, POS_COLOR, NEG_COLOR


def viz_timesurface(im):
    """
    Visualize timesurface

    Note: In order to generate a timesurface you need to call event_to_tensor.timesurface
    Typically, if you want to see an exponential decay timesurface you don't set the arg "reset" to True
    in order to keep the latest event's timestamp at each pixel.

    Here we assume the timesurface has already been normalized between [0,1] either by min-max normalization
    or with exponential time-decay.

    Args:
        im (np.ndarray): Array of shape (2,H,W) or (H,W)

    Returns:
        output_array (np.ndarray): Array of shape (H,W,3)
    """
    if len(im) == 2:
        im = np.maximum(im[0], im[1])
    img = (im * 255).astype(np.uint8)
    show = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return show


def gray_to_rgb(im):
    """
    Just Repeat image 3 times

    Args:
        im (np.ndarray): Array of shape (H,W)

    Returns:
        output_array: array of shape (H,W,3)
    """
    return im[..., None].repeat(3, 2)


def viz_histo_rgb(im):
    """
    visualize histo image with 3 channels

      Args:
          im (np.ndarray): Array of shape (H,W)

      Returns:
          output_array: array of shape (H,W,3)
    """
    im = viz_histo(im)
    return gray_to_rgb(im)


def viz_histo_filtered(im, val_max=0.5):
    """
    visualize strongly filtered histo image with 3 channels

      Args:
          im (np.ndarray): Array of shape (2,H,W)

      Returns:
          output_array: array of shape (H,W,3)
    """
    im = im = im.astype(np.float32)
    im = im[1] - im[0]
    im = np.clip(im, -val_max, val_max)
    im = ((im + val_max) / (2 * val_max) * 255).astype(np.uint8)

    return gray_to_rgb(im)


def viz_diff(im):
    """
    Visualize difference of histogram

    Args:
        im (np.ndarray): Array of shape (H,W)

    Returns:
        output_array (np.ndarray): Array of shape (H,W)
    """
    im = im.astype(np.float32)
    im = filter_outliers(im, 3)
    im = normalize(im) * 255
    im = np.uint8(im)

    if im.max() == 0:
        im[...] = 127
    return im


def viz_histo(im):
    """
    Visualize difference of histogram

    Args:
        im (np.ndarray): Array of shape (2,H,W)

    Returns:
        output_array (np.ndarray): Array of shape (H,W)
    """
    im = im.astype(np.float32)
    im = im[1] - im[0]
    return viz_diff(im)


def viz_histo_binarized(im):
    """
    Visualize binarized histogram of events

    Args:
        im (np.ndarray): Array of shape (2,H,W)

    Returns:
        output_array (np.ndarray): Array of shape (H,W,3)
    """
    img = np.full(im.shape[-2:] + (3,), BG_COLOR, dtype=np.uint8)
    y, x = np.where(im[0] > 0)
    img[y, x, :] = POS_COLOR
    y, x = np.where(im[1] > 0)
    img[y, x, :] = NEG_COLOR
    return img


def viz_diff_binarized(im):
    """
    Visualize binarized difference of events ("ON"-"OFF")

    Args:
        im (np.ndarray): Array of shape (H,W)

    Returns:
        output_array (np.ndarray): Array of shape (H,W,3)
    """
    img = np.full(im.shape[-2:] + (3,), BG_COLOR, dtype=np.uint8)
    y, x = np.where(im[0] > 0)
    img[y, x, :] = POS_COLOR
    y, x = np.where(im[0] < 0)
    img[y, x, :] = NEG_COLOR
    return img


def viz_event_cube_rgb(im, split_polarity=True):
    """
    Visualize 3 out of 6 channels in RGB mode.

    Args:
        im (np.ndarray): Array of shape (T,H,W) T images or T//2 group of images with 2 channels
        split_polarity: whether each image is single-channel (ON-OFF) or (ON, OFF) 2 channel images.

    Returns:
        output_array (np.ndarray): Array of shape (H,W,3)
    """
    t, h, w = im.shape
    if split_polarity:
        im = im[:6]
        im = im.reshape(3, 2, h, w)
        rgb = np.concatenate([viz_histo(im[i])[..., None] for i in range(3)], axis=2)
    else:
        im = im[:3]
        im = im.reshape(3, 1, h, w)
        rgb = np.concatenate([viz_diff(im[i])[..., None] for i in range(3)], axis=2)

    return rgb


def viz_multichannel_timesurface(tensor, blend=False):
    """
    Visualizes three channels of a multi channel timesurface.

    Args:
        tensor (np.ndarray): array of shape (T,H,W) T images or T//2 group of images with 2 channels
        blend (boolean): whether to blend different channels to visualize them as one.

    Returns:
        output_array (np.ndarray): array of shape (H,W,3)
    """

    channel_index = 0
    n_micro_tbins = tensor.shape[0] // 2

    if blend:
        range_value = 1. / n_micro_tbins
        im = range_value * tensor
        im[::2] = im[::2] + np.arange(0, n_micro_tbins)[:, None, None]
        im[1::2] = im[::2] + np.arange(0, n_micro_tbins)[:, None, None]
        im = np.maximum(im[::2].sum(axis=0), im[1::2].sum(axis=0))
        return viz_timesurface(im)
    else:
        return viz_event_cube_rgb(tensor, split_polarity=True)


def normalize(im):
    """
    Normalizes image by min-max

    Args:
        im (np.ndarray): Array of any shape

    Returns:
        output_array (np.ndarray): Normalized array of same shape as input
    """
    low, high = im.min(), im.max()
    return (im - low) / (high - low + 1e-6)


def filter_outliers(input_val, num_std=2):
    """
    Filter outliers in an array

    Args:
        input_val (np.ndarray): Array of any shape

    Returns:
        output_array (np.ndarray): Normalized array of same shape as input
    """
    val_range = num_std * input_val.std()
    img_min = input_val.mean() - val_range
    img_max = input_val.mean() + val_range
    normed = np.clip(input_val, img_min, img_max)  # clamp
    return normed
