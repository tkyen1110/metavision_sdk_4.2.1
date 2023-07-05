# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Subclassing Torch dataset to load DAT files and labels from events and wrapping them using the dataloader
class. It supports currently DAT and HDF5 files, although we recommend to use the latter.

This class is generic to any type of labels. A function should be provided to load them.
"""

import os
import math

from functools import partial

import h5py
import cv2
import numpy as np
import torch
import torch.utils.data as data

from metavision_core.event_io import EventDatReader
from ..preprocessing import get_preprocess_dict

from .scheduler import Scheduler
from .transformations import transform_sequence

# pylint: disable=no-member


def load_labels_stub(metadata, start_time, duration, tensor):
    """This is a stub implementation of a function to load label data.

    This function doesn't actually load anything and should be passed to the SequentialDataset for
        self-supervised training when no actual labelling is required.

    Args:
        metadata (FileMetadata): This class contains information about the sequence that is being read.
            Ideally the path for the labels should be deducible from `metadata.path`.
        start_time (int): Time in us in the file at which we start reading.
        duration (int): Duration in us of the data we need to read from said file.
        tensor (torch.tensor): Torch tensor of the feature for which labels are loaded.
            It can be used for instance to filter out the labels in area where there is no events.
    Returns:
        labels should be indexable by time bin (to differentiate the labels of each time bin). It could
            therefore be a list of length *num_tbins*.
        (boolean nd array): This boolean mask array of *length* num_tbins indicates
            whether the frame contains a label. It is used to differentiate between time_bins that actually
            contain an empty label (for instance no bounding boxes) from time bins that weren't labeled due
            to cost constraints. The latter timebins shouldn't contribute to supervised losses used during
            training.
    """
    labels = [torch.empty(0) for _ in np.arange(0, duration, metadata.delta_t)]
    frame_is_labeled = np.zeros((len(labels)), dtype=bool)

    return labels, frame_is_labeled


def _downsample_clip(frames, output_array, max_val=None):
    """Applies downsampling and clipping to a list of features frames."""
    num_tbins = output_array.shape[0]

    h_src, w_src = frames.shape[-2:]
    h_dst, w_dst = output_array.shape[-2:]
    assert frames.dtype == output_array.dtype
    assert len(frames) == num_tbins, f"number of preprocessed features incorrect {len(frames)}, should be {num_tbins}"
    if h_src != h_dst or w_src != w_dst:
        for tbin in range(num_tbins):
            for channel in range(output_array.shape[1]):
                output_array[tbin, channel] = cv2.resize(frames[tbin, channel], (w_dst, h_dst),
                                                         output_array[tbin, channel], interpolation=cv2.INTER_AREA)

    else:
        output_array[...] = frames  # frames is exactly what we want
    if max_val is not None:
        np.clip(output_array, -max_val, max_val, out=output_array)


class SequentialDataset(data.Dataset):
    def __init__(self, files, delta_t, preprocess_function_name, array_dim,
                 load_labels=load_labels_stub, durations=[], batch_size=8, preprocess_kwargs={},
                 padding=False, transforms=None):
        """
        Subclass of torch.data.dataset designed to stream batch of sequences chronologically.

        It will read data sequentially from the same file until it jumps to
        another file which will also be read sequentially.

        Usually it is used in conjunction with the SequentialDataLoader, in which case this object is
        directly initialized by the SequentialDataLoader itself.

        Args:
            files (list): List of input files. Can be either DAT files or HDF5 files.
            delta_t (int): Timeslice delta_t in us.
            preprocess_function_name (string): Name of the preprocessing function used to turn events into
                features. Can be any of the functions present in metavision_ml.preprocessing or one registered
                by the user.
            array_dim (int list): Dimension of feature tensors:
                (num_tbins, channels, sensor_height * 2^-k, sensor_width * 2^-k)
            load_labels (function):
            batch_size (int): Number of sequences being read concurrently. This can affect the loading time
                of the batch and has effect on the gradient statistics.
            preprocess_kwargs: dictionary of optional arguments to the preprocessing function.
            padding (boolean): If True, at the end of an epoch the Dataset will run with incomplete batches
                when it can't read a complete one until all data is read. The last incomplete batches
                will contain FileMetadata object, with padding = True so that no loss is computed on them.
                If False, the epoch stops after the last complete batch. This can be used to make sure that
                evaluation is computed on the whole test set for example.
            transforms (torchvision Transforms): Transformations to be applied to each frame of a sequence.

        Attributes:
            downsampling_factor (int): Parameter used to reduce the spatial dimension of the obtained feature.
                Actually multiply the coordinates by 2**(-downsampling_factor).
        """
        self.array_dim = array_dim
        self.num_tbins = self.array_dim[0]
        self.delta_t = delta_t
        self.total_tbins_delta_t = self.delta_t * self.num_tbins

        self.scheduler = Scheduler.create_schedule(files, durations, self.delta_t, self.num_tbins, batch_size,
                                                   max_consecutive_batch=None, shuffle=False, padding=padding)
        self.transforms = transforms
        self.files = self.scheduler.files
        self.batch_size = self.scheduler.batch_size  # the scheduler might have a smaller batch size (too few files)

        if len(self.files):
            self.height_orig, self.width_orig = self.files[0].get_original_size()

            h_shifted, w_shifted = array_dim[-2:]

            assert_error_msg = f"array_dim[-2]: {h_shifted} should be {self.height_orig}*2^(-k)"
            assert (float(self.height_orig) / h_shifted).is_integer(), assert_error_msg
            assert_error_msg = f"array_dim[-1]: {w_shifted} should be {self.width_orig}*2^(-k)"
            assert (float(self.width_orig) / w_shifted).is_integer(), assert_error_msg
            assert self.width_orig / w_shifted == self.height_orig / h_shifted, "wrong aspect ratio in array_dim"

            self.downsampling_factor = int(np.log2(self.width_orig / w_shifted))
        else:
            self.downsampling_factor = 0

        self.preprocess_dict = get_preprocess_dict(preprocess_function_name)
        self.preprocess_kwargs = self.preprocess_dict['kwargs']
        self.preprocess_kwargs.update(preprocess_kwargs)
        if "preprocess_dtype" in self.preprocess_kwargs:
            # if there are preprocessed files their dtype overrides the option
            hdf5_metadata = [metadata for metadata in self.files if metadata.path.endswith("h5")]
            if len(hdf5_metadata):
                with h5py.File(hdf5_metadata[0].path, "r") as f:
                    self.dtype = f['data'].dtype if not f['data'].attrs.get('store_as_uint8', False) else np.float32
            else:
                self.dtype = self.preprocess_kwargs.pop("preprocess_dtype")
        else:
            self.dtype = np.uint8

        self.events_to_tensor = partial(
            self.preprocess_dict['events_to_tensor'], total_tbins_delta_t=array_dim[0] * delta_t,
            downsampling_factor=self.downsampling_factor, reset=False, **self.preprocess_kwargs)

        self.load_labels = load_labels

    def get_size(self):
        """
        Returns height and width of histograms/features, i.e. size after downsampling_factor.
        """
        return self.array_dim[-2:]

    def get_size_original(self):
        """
        Returns height and width of input events before downscaling.
        """
        return (self.height_orig, self.width_orig)

    def get_unique_files(self):
        """
        Returns a unique list of FileMetadata.
        It is useful in case of a curriculum learning (launch using reschedule) where there is several
        occurrences of the same file with different start_ts.
        """
        paths = set()
        unique_files = []
        for filematadatum in self.files:
            if filematadatum.path in paths:
                continue
            else:
                paths.add(filematadatum.path)
                unique_files.append(filematadatum)
        return unique_files

    def get_batch_metadata(self, batch_idx):
        """
        Gets the metadata information of the batch obtained from the batch indices.

        Return:
            List of tuple composed of (FileMetadata, start  list time of sequence in us, duration of sequence in us).
        """
        indices = np.arange(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)
        return [self.scheduler[index] for index in indices]

    def shuffle(self, seed=None):
        """
        Shuffles the list of input files.
        """
        self.scheduler.shuffle(seed=seed)

    def reschedule(self, max_consecutive_batch, shuffle=True):
        """
        Recomputes a new schedule corresponding to the same files but a different *max_consecutive_batch* parameter.

        This is useful when trying to do curriculum learning when you want to feed your model with sequence
        of increasing duration. Alternatively if you don't want to change any parameters you can simply
        use the `shuffle` function.

        Args:
            max_consecutive_batch (int): Maximum number of consecutive batches allowed in a sequence. If a
                file is longer than *max_consecutive_batch* x *num_tbins* x *delta_t* the rest will be
                considered as part of another sequence. If None, the full length of the sequence will be used.
            shuffle (boolean): Whether to apply a random shuffle to the list of files. Setting it to True,
                is recommended.
        """
        self.scheduler = self.scheduler.reschedule(max_consecutive_batch, self.num_tbins, self.delta_t,
                                                   shuffle=shuffle)
        self.files = self.scheduler.files

    def _load_sequences(self, metadata, batch_start_time, duration):
        """
        Either load events and compute features or load precomputed ones from HDF5.
        """
        assert batch_start_time >= metadata.start_ts
        assert duration % self.num_tbins == 0, "Error : duration is not a multiple of num_tbins.\n\tbatch_start_time: {},  duration: {},  self.num_tbins: {},  metadata.delta_t: {}".format(
            batch_start_time, duration, self.num_tbins, metadata.delta_t)

        array = np.zeros(self.array_dim, dtype=self.dtype)

        file_extension = os.path.splitext(metadata.path)[1].lower()
        if file_extension in (".dat",):
            events = self._load_events(metadata, batch_start_time, duration)
            self.events_to_tensor(events, array)
        elif file_extension in ('.hdf5', ".h5"):
            precomputed_frames = self._load_hdf5(metadata, batch_start_time, duration)
            _downsample_clip(precomputed_frames, array, max_val=None)
        else:
            raise ValueError("can't load ", metadata)

        return array

    def _load_events(self, metadata, batch_start_time, duration):
        """Fetches events from filename, batch_start_time & total_tbins_delta_t."""

        video = EventDatReader(metadata.path)

        video.seek_time(batch_start_time)
        events = video.load_delta_t(duration)

        if len(events) > 0:
            events['t'] -= batch_start_time

        return events

    def _load_hdf5(self, metadata, batch_start_time, duration):
        """Fetches precomputed tensors from filename, batch_start_time & total_tbins_delta_t."""
        with h5py.File(metadata.path, "r") as f:
            assert f["data"].attrs.get("mode", "delta_t") == "delta_t", "only delta_t mode supported"
            assert f["data"].attrs["delta_t"] > 0
            array = f['data'][
                batch_start_time // f['data'].attrs["delta_t"]:
                (batch_start_time + duration) // f['data'].attrs["delta_t"]]
            if f['data'].attrs.get('store_as_uint8', False):
                array = array.astype(np.float32) / 255

            return array

    def __getitem__(self, index):
        """This function maps an iteration index to a part of a file in the dataset, so that a
        Pytorch DataLoader constructed with the `shuffle` parameter set to False produces chronological batches.

        Returns:
            A tuple, composed of:

             - sequence (torch.tensor): A feature of shape `self.array_dim`
             - labels (list): A list of labels of length *num_tbins* i.e. `self.array_dim.shape[0]`
             - keep_memory (float): This value is either:
                - 1. if the sequence was already being read during a previous iteration (in which case the
                    memory of the model should be kept intact).
                - 0. if this is the beginning of the sequence. In this case the memory of the model should be
                    reset.
             - frame_is_labeled (boolean nd array): This mask vector of length num_tbins indicates
                 whether the frame contains a label. It is used to differentiate between time_bins that
                 actually contain an empty label (for instance no bounding boxes) from time bins that
                 weren't labeled due to cost constraints. The latter timebins shouldn't contribute to
                 supervised losses used during training.
             - video_infos (tuple of FileMetadata int int): Scheduling information for the item.
        """
        metadata, batch_start_time, duration = self.scheduler[index]
        if metadata.is_padding():
            seq = torch.from_numpy(np.zeros(self.array_dim, dtype=self.dtype))
            keep_memory = 1.0
            frame_is_labeled = np.zeros((self.array_dim[0],), dtype=bool)
            return (seq, [np.nan for _ in range(self.num_tbins)],
                    keep_memory, frame_is_labeled, (metadata, batch_start_time, duration))

        seq = self._load_sequences(metadata, batch_start_time, duration)

        # keep_memory is false (i.e. we must reset) only at the beginning of a new
        # sequence : when batch_start_time == metadata.start_ts
        keep_memory = float(batch_start_time > metadata.start_ts)
        seq = torch.from_numpy(seq)

        # data aug
        seq = transform_sequence(seq, metadata, self.transforms, base_seed=self.scheduler.base_seed)

        labels, frame_is_labeled = self.load_labels(metadata, batch_start_time, duration, seq)

        video_infos = (metadata, batch_start_time, duration)
        return (seq, labels, keep_memory, frame_is_labeled, video_infos)

    def __len__(self):
        return len(self.scheduler)


def collate_fn(data_list):
    """
    Builds a batch from the result of the different `__getitem__` calls of the Dataset. This function
    helps define the DataLoader behaviour.

    By doing so it puts the temporal dimensions (each time bin) as the first dimension and
    the batch dimension becomes second.

    Args:
        data_list (tuple list): List where each item is a tuple composed of a tensor, the labels,
            the keep memory mask and the frame_is_labeled mask.

    Returns:
        dictionary: see SequentialDataLoader
    """
    tensors, labels, mask_keep_memory, frame_is_labeled, file_metadata = zip(*data_list)
    tensors = torch.stack(tensors, 1)

    # if some of the file metadata have a valid labelling_delta_t and some don't, this stack operation will fail
    frame_is_labeled = np.stack(frame_is_labeled, 1)
    frame_is_labeled = torch.from_numpy(frame_is_labeled)
    mask_keep_memory = torch.FloatTensor(mask_keep_memory).view(-1, 1, 1, 1)

    t, n = tensors.shape[:2]

    tn_labels = [[labels[i][t] for i in range(n)] for t in range(t)]

    return {"inputs": tensors, "labels": tn_labels, "mask_keep_memory": mask_keep_memory,
            "frame_is_labeled": frame_is_labeled, "video_infos": file_metadata}


class SequentialDataLoader(object):
    """
    SequentialDataLoader uses `a pytorch DataLoader`_ to read batches chronologically.

    It is used simply as an iterator and returns a dictionary containing the following keys:

     * `inputs` a torch.tensor of shape `num_tbins` x `batch_size` x `channel` x `height` x `width`.
        Note that it is normalized to 1.
        The dtype depends on the preprocessing function used but can by specifying the `preprocess_kwargs`.

     * `labels` is the list of labels provided by the `load_labels` function.
     * `mask_keep_memory` a float array of shape `batch_size`, with values in (0., 1.) indicating
        whether memory is kept or reset at the beginning of the sequence.
     * `frame_is_labeled` a boolean array of shape `num_tbins` x `batch_size`, indicating whether the
        corresponding labels can be used for loss computation. (id est if the labels are valid or not).
     * `video_infos` is a list of (FileMetadata, batch_start_time, duration) of size batch_size containing
         infos about each recording in the batch.

    Attributes:

        batch_size (int): Number of sequences being read concurrently. This can affect the loading time
            of the batch and has effect on the gradient statistics.
        num_workers (int): Number of processes being used by the DataLoader, 0 means it uses Python's main
            process. More processes help with speed but up to a point: too many processes can actually hurt
            loading times.
        max_consecutive_batch (int): Maximum number of consecutive batches allowed in a sequence. If a
            file is longer than *max_consecutive_batch* x *num_tbins* x *delta_t* the rest will be
            considered as part of another sequence. If None, the full length of the sequence will be used.
        device (torch.device): Indicates on which device (cpu or cuda for instance) the data will be put.
        dataset (SequentialDataset): Instance of SequentialDataset that is used to load the data, or
            possibly change the scheduling. *Note* that if the dataset is changed, that change won't take
            effect until the next iteration of the DataLoader.

    Args:
        files (list): List of input files. Can be either DAT files or HDF5 files.
        delta_t (int): Timeslice delta_t in us.
        preprocess_function_name (string): Name of the preprocessing function used to turn events into
            features. Can be any of the functions present in metavision_ml.preprocessing or one registered by the user.
        array_dim (int list): Dimension of feature tensors:
            (num_tbins, channels, sensor_height // 2^k, sensor_width >> 2^k)
        load_labels: function providing labels (see load_labels_stub).
        durations (int list): Optionally you can provide the durations in us to all the files in input.
            This allows to save a bit of time when there are many of them. If you provide a duration that is
            shorter than the actual duration of a sequence, only part of it will be read.
        batch_size (int): Number of sequences being read concurrently. This can affect the loading time
            of the batch and has effect on the gradient statistics.
        num_workers (int): Number of processes being used by the DataLoader, 0 means it uses Python's main
            process. More processes help with speed but up to a point: too many processes can actually hurt
            loading times.
        preprocess_kwargs: dictionary of optional arguments to the preprocessing function. This can be used
            to override the default value of `max_incr_per_pixel` for instance. `{"max_incr_per_pixel": 20}` to clip and
            normalize tensors by 20.
        shuffle (boolean): If True, breaks the temporal continuity between batches. This should be only used
            when training a model **without** memory.
        padding (boolean): If True, at the end of an epoch the Dataset will run with incomplete batches
            when it can't read a complete one until all data is read. The last incomplete batches
            will contain FileMetadata object, with padding = True so that no loss is computed on them.
            If False, the epoch stops after the last complete batch. This can be used to make sure that
            evaluation is computed on the whole test set for example.
        transforms (torchvision Transforms): Transformations to be applied to each frame of a sequence.

    Examples:
        >>> array_dim = [5, 2, 480, 640]
        >>> dataloader = SequentialDataLoader(['train/file1.dat', 'train/file1.dat'], 50000, "histo", array_dim)
        >>> for ind, data_dic in enumerate(dataloader):
        >>>     batch = data_dic["inputs"]
        >>>     targets = data_dic["labels"]
        >>>     mask = data_dic["mask_keep_memory"]
        >>>     frame_is_labeled = data_dic["frame_is_labeled"]

    .. _a pytorch DataLoader:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    def __init__(self, files, delta_t, preprocess_function_name, array_dim,
                 load_labels=load_labels_stub, durations=[], batch_size=8, num_workers=2,
                 preprocess_kwargs={}, shuffle=False, padding=False, transforms=None):

        self.num_workers = num_workers
        self.shuffle = shuffle

        self.device = torch.device('cpu')
        self.max_consecutive_batch = None
        self.dataset = SequentialDataset(files, delta_t, preprocess_function_name, array_dim,
                                         durations=durations, batch_size=batch_size, load_labels=load_labels,
                                         preprocess_kwargs=preprocess_kwargs, padding=padding, transforms=transforms)
        self.batch_size = self.dataset.batch_size

    def cuda(self, device=torch.device('cuda')):
        """Sets the SequentialDataLoader to copy tensors to GPU memory before returning them.

        Args:
            device (torch.device): The destination GPU device. Defaults to the current CUDA device.
        """
        assert torch.cuda.is_available()
        self.device = device
        return self

    def to(self, device):
        """Sets the SequentialDataLoader to copy tensors to the given device before returning them.

        Args:
            device (torch.device): The destination GPU device. For instance `torch.device('cpu')`
                or torch.device('cuda').
        """
        assert isinstance(device, torch.device)
        self.device = device
        return self

    def cpu(self):
        """Sets the SequentialDataLoader to leave tensors on CPU."""
        self.device = torch.device('cpu')
        return self

    def __len__(self):
        return len(self.dataset) // self.dataset.batch_size

    def __iter__(self):
        # actual batch size might be smaller or even 0 in which case it is set to 1 (for the DataLoader)
        self.dataloader = iter(data.DataLoader(self.dataset, batch_size=max(1, self.dataset.batch_size),
                                               num_workers=self.num_workers, shuffle=self.shuffle,
                                               collate_fn=collate_fn, pin_memory=False))
        self.step = 0
        return self

    def __next__(self):
        if self.step == len(self):
            raise StopIteration

        self.step += 1
        data = next(self.dataloader)

        data['inputs'] = data['inputs'].to(device=self.device)
        data['mask_keep_memory'] = data['mask_keep_memory'].to(device=self.device)

        data['inputs'] = data['inputs'].float()

        return data

    def get_vis_func(self):
        """Returns the visualization function corresponding to the preprocessing being used."""
        return self.dataset.preprocess_dict['viz']

    def show(self, viz_labels=None):
        """
        Visualizes batches of the DataLoader in parallel with open cv.

        This returns a generator that draws the input and also the labels if
        a "viz_labels" function is provided.

        Args:
            viz_labels (function): Optionally take a visualization function for labels. Its signature is
                - img (np.ndarray) a image of size (height, width, 3) and of dtype np.uint8
                - labels as defined in your load_labels function.
        """
        height_scaled, width_scaled = self.dataset.array_dim[-2:]
        size_x = int(math.ceil(math.sqrt(self.dataset.batch_size)))
        size_y = int(math.ceil(float(self.dataset.batch_size) / size_x))

        frame = np.zeros((size_y * height_scaled, width_scaled * size_x, 3), dtype=np.uint8)

        for ind, data_dic in enumerate(self):
            batch = data_dic["inputs"]
            mask = data_dic["mask_keep_memory"]

            metadata = self.dataset.get_batch_metadata(ind)

            for t in range(batch.shape[0]):
                for index, im in enumerate(batch[t]):
                    im = im.cpu().numpy()

                    y, x = divmod(index, size_x)
                    img = self.get_vis_func()(im)

                    # if a function was provided display the labels
                    if viz_labels is not None:
                        labels = data_dic["labels"][t][index]
                        img = viz_labels(img, labels)

                    if t <= 1 and not mask[index]:
                        # mark the beginning of a sequence with a red square
                        img[:10, :10, 0] = 222
                    # put the name of the file
                    name = metadata[index][0].path.split('/')[-1]
                    cv2.putText(img, name, (int(0.05 * (width_scaled)), int(0.94 * (height_scaled))),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, (50, 240, 12))

                    frame[y * (height_scaled):(y + 1) * (height_scaled),
                          x * (width_scaled): (x + 1) * (width_scaled)] = img

                yield frame
