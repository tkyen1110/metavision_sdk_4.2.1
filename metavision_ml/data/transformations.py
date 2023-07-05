# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

import random

import torch
import numpy as np
import PIL


def transform_sequence(sequence, metadata, transforms, base_seed=0):
    """Applies a series of 2d transformations to each frame and each channel of a sequence.

    The metadata of the sequence is used to provide a seed.

    Args:
        sequence (torch.tensor): feature tensor of shape (num_time_bins, num_channels, height, width)
        metadata (FileMetadata): object describing the metadata of the sequence to which the tensor belongs.
        transforms (torchvision.transforms): transform to be applied to each channel of each frame.
        base_seed (int): base_seed to add to the sequence in order to have additionnal randomness.
            However it needs to be the constant within an epoch.

    Returns:
        sequence (torch.tensor): feature tensor of shape (num_time_bins, num_channels, height, width)
    """
    if transforms is None:
        return sequence

    seed = hash(hash(metadata) + base_seed)

    for t, tensor in enumerate(sequence):
        for c, channel in enumerate(tensor):
            random.seed(seed)
            torch.random.manual_seed(seed)
            sequence[t, c] = torch.from_numpy(np.array(transforms(PIL.Image.fromarray(channel.numpy()))))

    return sequence


def transform_ev_tensor(ev_tensor, file_path, transforms, base_seed=0):
    """Applies a series of 2d transformations to each frame and each channel of a ev_tensor.

    Args:
        ev_tensor (torch.tensor): feature tensor of shape (num_ev_reps, num_channels, height, width).
        file_path (string): it will be used to calculate the seed.
        transforms (torchvision.transforms): transform to be applied to each channel of each frame.
        base_seed (int): base_seed to add to the sequence in order to have additionnal randomness.
            However it needs to be the constant within an epoch.

    Returns:
        ev_tensor (torch.tensor): feature tensor of shape (num_ev_reps, num_channels, height, width).
    """

    seed = hash(hash(file_path) + base_seed)

    for frame_idx, tensor in enumerate(ev_tensor):
        for c_idx, channel in enumerate(tensor):
            random.seed(seed)
            torch.random.manual_seed(seed)
            ev_tensor[frame_idx, c_idx] = torch.from_numpy(np.array(transforms(PIL.Image.fromarray(channel.numpy()))))

    return ev_tensor
