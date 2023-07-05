# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Scheduler is a File agnostic class that does the scheduling of sequence for a dataloader.
"""

import random
import os
import math

import numpy as np
import h5py

from metavision_core.event_io import EventDatReader, RawReader, EventNpyReader, get_raw_info


def get_duration(path):
    """Returns duration of a file"""
    duration = None
    file_extension = os.path.splitext(path)[1].lower()
    if file_extension in (".dat", ".npy"):
        cls = [EventDatReader, EventNpyReader][file_extension == ".npy"]
        duration = int(1e6 * cls(path).duration_s)
    elif file_extension == ".raw":
        duration = get_raw_info(path)['duration']
    elif file_extension in ('.hdf5', ".h5"):
        with h5py.File(path, "r") as f:
            mode = f["data"].attrs.get("mode", "delta_t")
            if mode == "delta_t":
                duration = int(f['data'].shape[0] * f["data"].attrs["delta_t"])
            elif mode == "n_events":
                #NB for mode=n_events we assume start_ts is always =0 
                duration = f['last_timestamps'][-1]
            else:
                raise ValueError(f"unknown mode {mode}")
    return duration


def _get_durations(files):
    """Returns a list of durations in us."""
    durations = []
    for path in files:
        duration = get_duration(path)
        if duration is None:
            raise ValueError("can't measure durations of  ", os.path.basename(path))
        else:
            durations.append(duration)
    return durations


def _get_original_size_file(path):
    """Returns the couple (height, width) of a file.

    Args:
        path (string): File path, either a DAT file of an hdf5 file path.
    """
    file_extension = os.path.splitext(path)[1].lower()
    if file_extension in (".dat", ".npy"):
        cls = [EventDatReader, EventNpyReader][file_extension == ".npy"]
        return cls(path).get_size()
    elif file_extension == ".raw":
        cls = RawReader(path)
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


class FileMetadata(object):
    """
    Metadata class describing a sequence.

    Args:
        file (str):    Path to the sequence file.
        duration (int):    Sequence duration in us.
        delta_t (int):    Duration of a time bin in us.
        num_tbins (int):    Number of time bins together.
        labels (str):   Path to the label file for the sequence.
        start_ts  (int):    Timestamps at which we start reading the sequence. effectively cuts it.
        padding (boolean): Whether the object is padding (i.e. the FileMetadata is associated to no file
            or labels and is just here in case of incomplete batches.)

    Attributes:
        path (str):    Path to the sequence file
        duration (int):    Sequence duration in us
        delta_t (int):    Duration of a time bin in us
        num_tbins (int):    Number of time bins together
        labels (str):   Path to the label file for the sequence
        start_ts  (int):    Timestamps at which we start reading the sequence. effectively cuts it
        padding (boolean): Whether the object is padding (i.e. the FileMetadata is associated to no file
            or labels and is just here in case of incomplete batches.)
    """

    def __init__(self, file, duration, delta_t, num_tbins,
                 labels=None, start_ts=0, padding=False):
        self.padding = padding
        if padding:
            return  # when padding FileMetadata has only the padding attribute

        self.path = file

        self.start_ts = int(start_ts)
        self.duration = duration

        self.delta_t = delta_t
        self.num_tbins = num_tbins
        self.labels = labels

        self._height_orig, self._width_orig = None, None

    def get_original_size(self):
        """Returns the couple (height, width) of a file before any downsampling was optionally done.

        This corresponds to the resolution of the imager used to record the original data.
        """
        if self._width_orig is None:
            self._height_orig, self._width_orig = _get_original_size_file(self.path)
        return self._height_orig, self._width_orig

    def __repr__(self):
        if self.padding:
            return "FileMetadata object PADDING"
        repr = "FileMetadata object {}\n".format(self.path)
        repr += "\tstart_ts {}us delta_t {}us, num_tbins {}, total duration {}us\n".format(
            self.start_ts, self.delta_t, self.num_tbins, self.duration)
        if self.labels:
            repr += "labels {}".format(self.labels)
        return repr

    def __hash__(self):
        # required to be able to put FileMetadata in a set
        return hash(repr(self))

    def __eq__(self, other):
        """Returns True if both FileMetadata are equal.

        Two FilesMetadata with identical parameters including starts are considered equal.
        """
        same_params = self.delta_t == other.delta_t and self.num_tbins == other.num_tbins
        same_file = self.path == other.path and self.start_ts == other.start_ts and self.labels == other.labels
        return same_file and same_params

    def get_remaining_duration(self):
        """Returns the duration left considering the starting point."""
        return self.duration - self.start_ts

    def get_ending(self):
        for ending in ('_td.dat', ".h5"):
            if self.path.endswith(ending):
                return ending
        raise ValueError('Unknown path type for FileMetadata')

    def is_precomputed(self):
        """Is the data in a HDF5 File."""
        return os.path.splitext(self.path)[1].lower() in ('.h5', '.hdf5')

    def is_padding(self):
        """Is padding data."""
        return self.padding


def _rounding(duration, total_tbins_delta_t, max_batch=None, rounding_tolerance=1000):
    """Rounds down a duration so that it is a multiple of total_tbins_delta_t

    If max_batch is specified, it makes sure that duration is no more than max_batch*total_tbins_delta_t
    input can be scalars or numpy ndarrays.

    Args:
        duration (int or numpy array): Duration in us of a file.
        total_tbins_delta_t (int): Total duration of multiple tbins fed to a network all together.
        max_batch (int): If not None, limits the duration to max_batch * total_tbins_delta_t, i.e. the
            duration of that many batches.
        rounding_tolerance (int): Rounds up instead of down if we are less from rounding_tolerance from
            the upper number.
    """
    excess_duration = duration % total_tbins_delta_t
    if isinstance(excess_duration, np.ndarray):
        mask = total_tbins_delta_t - (excess_duration) <= rounding_tolerance
        excess_duration[mask] = duration[mask] % - total_tbins_delta_t
    elif total_tbins_delta_t - (excess_duration) <= rounding_tolerance:
        # if we only need a few microseconds to round up we actually extend the duration with negative excess
        excess_duration = duration % - total_tbins_delta_t
    rounded_duration = duration - excess_duration

    if max_batch is None:
        return rounded_duration
    else:
        return np.minimum(rounded_duration, max_batch * total_tbins_delta_t)


class Scheduler(object):
    """
    File agnostic class that does the scheduling of sequence for a dataloader.
    Assumes a dataloader in non shuffle mode for temporal continuity.

    Args :
        filesmetadata (FileMetadata list): List of FileMetadata objects describing the dataset.
        total_tbins_delta_t (int): Duration in us of a sequence inside a minibatch.
        batch_size (int): Number of sequences being read concurrently.
        max_consecutive_batch (int): Maximum number of consecutive batches allowed in a sequence. If a
            file is longer than *max_consecutive_batch* x *total_tbins_delta_t* the rest will be
            considered as part of another sequence. If None, the full length of the sequence will be used.
            This is used for curriculum learning to vary how long sequences are.
        padding (boolean): If True, the Scheduler will run with incomplete batches when it can't
            read a complete one until all data is read. The last incomplete batches will contain
            FileMetadata object, with padding = True so that no loss is computed on them.
            If False, the Scheduler stops at the last complete batch
        base_seed (int): consistent random seed associated with each epoch.
    """

    def __init__(self, filesmetadata, total_tbins_delta_t, batch_size,
                 max_consecutive_batch=None, padding=False, base_seed=0):
        self.padding = padding
        self.files = filesmetadata

        self.max_consecutive_batch = max_consecutive_batch

        self.batch_size = batch_size
        self.total_tbins_delta_t = total_tbins_delta_t

        if len(filesmetadata):
            assert batch_size <= len(filesmetadata)
        else:
            print("warning: empty schedule!")

        self._construct_schedule()
        self.base_seed = base_seed

    @classmethod
    def create_schedule(cls, files, durations, delta_t, num_tbins, batch_size,
                        labels=None, max_consecutive_batch=None, shuffle=True,
                        padding=False):
        """
        Alternate way of constructing a Scheduler with paths and duration instead of FileMetadata list
        create a full schedule where everything is read
        """
        total_tbins_delta_t = delta_t * num_tbins
        if labels is None or not labels:
            labels = [None for f in files]

        if len(durations) != len(files):
            assert len(durations) == 0
            durations = _get_durations(files)

        if max_consecutive_batch is None:
            files_metadata = [FileMetadata(fpath, duration, delta_t, num_tbins, labels=label)
                              for fpath, duration, label in zip(files, durations, labels)]
        else:
            # we create more FileMetadata with different beginning
            files_metadata = [
                FileMetadata(fpath, duration, delta_t, num_tbins, start_ts=start_ts, labels=label)
                for fpath, duration, label in zip(files, durations, labels)
                for start_ts in range(0,
                                      int(_rounding(duration - total_tbins_delta_t, total_tbins_delta_t) + 1),
                                      int(max_consecutive_batch * total_tbins_delta_t))]
        if shuffle:
            random.shuffle(files_metadata)
            base_seed = random.randint(0, int(1e9))
        else:
            base_seed = 0
        # filter out non existing files
        files_metadata = [f for f in files_metadata if os.path.exists(f.path)]

        if len(files_metadata) < len(files):
            assert len(files_metadata)
            print("Warning in Scheduler:  {} non existing files!".format(len(files) - len(files_metadata)))
        if len(files_metadata) < batch_size:
            print("Warning in Scheduler: batch_size reduced to {}".format(len(files_metadata)))
            batch_size = len(files_metadata)
        return cls(files_metadata, total_tbins_delta_t, batch_size,
                   max_consecutive_batch=max_consecutive_batch, padding=padding, base_seed=base_seed)

    def reschedule(self, max_consecutive_batch, num_tbins, delta_t, shuffle=True):
        """
        Returns a new schedule corresponding to the same files but some different parameters.

        This is useful when trying to do curriculum learning when you want to feed your model with sequence
        of increasing duration. Alternatively if you don't want to change any parameters you can simply
        use the `shuffle` function.

        Args:
            max_consecutive_batch (int): Maximum number of consecutive batches allowed in a sequence. If a
                file is longer than *max_consecutive_batch* x *num_tbins* x *delta_t* the rest will be
                considered as part of another sequence. If None, the full length of the sequence will be used.
            num_tbins (int): Number of time bins in each batch (also the first dimension of the input tensor)
            delta_t (int): In us duration of a single time bin.
            shuffle (boolean): Whether to apply a random shuffle to the list of files.
                If *max_consecutive_batch* is not None, this is heavily recommended.

        Returns:
            scheduler, a new Scheduler object.
        """

        # retrieve all unique files with duration
        files = []
        durations = []
        labels = []
        for f in self.files:
            if f.path not in files:
                files.append(f.path)
                durations.append(f.duration)

        return Scheduler.create_schedule(files, durations, delta_t, num_tbins, self.batch_size, labels=labels,
                                         max_consecutive_batch=max_consecutive_batch, shuffle=shuffle)

    def remove_files(self, files_to_remove):
        """
        Removes some files from the scheduler and reinitialize the schedule.
        """
        files_kept = []
        for f in enumerate(self.files):
            if f.path not in files_to_remove:
                files_kept.append(f)

        self.files = files_kept

        assert self.batch_size <= len(self.files)

        self._construct_schedule()

    def shuffle(self, seed=None):
        """Shuffles the FileMetadata list held by the Scheduler and reconstructs a schedule.

        Args:
            seed (int): seed value to make shuffling deterministic.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.files)
        self.base_seed = random.randint(0, int(1e9))
        self._construct_schedule()

    def _construct_schedule(self):
        """Constructs the cumulative_time_table_per_batch that __getitem__ is going to use."""

        self.durations = np.array([f.get_remaining_duration() for f in self.files])

        # initial scheduling
        max_batch = None if self.max_consecutive_batch is None else self.max_consecutive_batch
        # actual_durations takes into account max_consecutive_batch
        # note that duration is changed so that files appear to be read at normal speed
        self.actual_durations = _rounding(self.durations, self.total_tbins_delta_t, max_batch=max_batch)
        assert (self.actual_durations % self.total_tbins_delta_t == 0).all()
        # iterates through all files to put them in batches and add their cumulative durations to a timetable
        # in order to know which file should be read where
        self.cumulative_time_table_per_batch = [np.cumsum(self.actual_durations[i::self.batch_size])
                                                for i in range(self.batch_size)]

    def __getitem__(self, index):
        """Returns the metadata necessary to load a file at the right time during training.

        Given an index from the pytorch DataLoader in non random mode, determine the file_index
        and all parameters relevant to data loading: which file, when to start and how long a time
        slice is.
        Each file is assigned to its place in the batch beforehand *batch_index* then a table of
        cumulative durations is used to find the right file -> *file_index*
        then it is simply a matter of finding the time elapsed in the file -> *batch_start_time*

        Args:
            index (int): Index of the dataset as called by the DataLoader

        Returns:
            FileMetadata metadata class about the file
            start_time in us, indicates where in the file to start loading data from.
            duration in us, indicates how much data to load from the file
        """
        if index == self.__len__():
            raise IndexError
        # time at the end of the batch since the beginning of the epoch in us
        iteration_time = ((index // self.batch_size) + 1) * self.total_tbins_delta_t
        batch_index = index % self.batch_size
        if iteration_time > self.cumulative_time_table_per_batch[batch_index][-1]:
            assert self.padding
            return FileMetadata(None, None, None, None, padding=True), 0, 0
        # find which file correspond to this date in the right column of the time_table
        rel_file_index = np.searchsorted(self.cumulative_time_table_per_batch[batch_index], iteration_time)
        # the actual index of this table
        file_index = batch_index + self.batch_size * rel_file_index

        if rel_file_index:
            time_elapsed = self.cumulative_time_table_per_batch[batch_index][rel_file_index - 1]
        else:
            time_elapsed = 0

        batch_start_time = int(math.ceil((iteration_time - self.total_tbins_delta_t - time_elapsed)))

        curfile_total_tbins_delta_t = self.files[file_index].delta_t * self.files[file_index].num_tbins
        assert self.total_tbins_delta_t == curfile_total_tbins_delta_t

        curfile_duration = self.files[file_index].num_tbins * self.files[file_index].delta_t

        return self.files[file_index], self.files[file_index].start_ts + batch_start_time, curfile_duration

    def __len__(self):
        """
        Returns number of iterations of the Scheduler i.e. the number of batches in the epoch.
        """
        if not self.files:
            return 0
        if self.padding:
            duration = max(cum_time[-1] for cum_time in self.cumulative_time_table_per_batch)
        else:
            duration = min(cum_time[-1] for cum_time in self.cumulative_time_table_per_batch)

        return int(duration // self.total_tbins_delta_t) * self.batch_size
