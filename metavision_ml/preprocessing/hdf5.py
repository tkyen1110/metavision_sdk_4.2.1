# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Write HDF5 feature files from event files
"""

import numpy as np
import os
import math
from multiprocessing import Pool
from tqdm import tqdm

from metavision_ml.preprocessing import CDProcessor
from metavision_core.event_io import EventsIterator, EventNpyReader
from metavision_ml.utils.main_tools import check_input_power_2
from metavision_ml.utils.h5_writer import HDF5Writer


def _check_start_ts(start_ts, paths):
    """"
    checks that there is only one start_ts or one per path and convert it to integer.
    """
    # start_ts should either be an integer or a list of them
    if isinstance(start_ts, (int, float, np.number)):
        starts_ts = [int(start_ts)]
    else:
        starts_ts = [int(s) for s in start_ts]

    if len(starts_ts) == 1:
        starts_ts = [starts_ts[0] for p in paths]
    assert len(starts_ts) == len(paths), "either provide one *start_ts* or one per input path in *paths* !"
    return starts_ts


def split_label(box_path, output_folder, start_ts=0, max_duration=None):
    """
    Copy the labels according to split and max_durations
    """
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, os.path.basename(box_path))

    reader = EventNpyReader(box_path)
    reader.seek_time(start_ts)
    events = reader.load_n_events(reader.event_count())

    events['t'] -= start_ts
    if max_duration is None:
        np.save(out_path, events)
    else:
        i_file = 0
        name, suffix = out_path.rsplit("_b", maxsplit=1)
        out_path = name + "_{:d}_b" + suffix
        while len(events):
            last_index = np.searchsorted(events['t'], max_duration)
            np.save(out_path.format(i_file), events[:last_index])
            events = events[last_index:]
            events['t'] -= max_duration
            i_file += 1


def _write(dic):
    """
    Wrapper of write_to_hdf5 for the pool process.
    """
    write_to_hdf5(
        dic["path"],
        dic['start_ts'],
        output_folder=dic['output_folder'],
        delta_t=dic['delta_t'],
        preprocess=dic['preprocess'],
        height=dic['height'],
        width=dic['width'],
        max_duration=dic["max_duration"],
        store_as_uint8=dic['store_as_uint8'],
        mode=dic['mode'],
        n_events=dic['n_events'],
        preprocess_kwargs=dic['preprocess_kwargs'])


def write_to_hdf5(path, start_ts=0, output_folder=".", delta_t=50000,
                  preprocess="histo", height=None, width=None, max_duration=None,
                  store_as_uint8=False, mode="delta_t", n_events=0, preprocess_kwargs={}):
    """
    Generates a single HDF5 file of frames at a regular interval for dataset caching.

    It is meant to produce a dataset with a smaller memory footprint, that is therefore easier to load from disk.
    Generated files share the name of the input RAW file but have an HDF5 extension.

    The naming of the output_file follows this pattern:
        "{output_folder}/{path basename}_{start_ts}.h5"

    Args:
        path (string): Path of input files.
        output_folder (string): Folder path where data will be written.
        preprocess (string): Name of the preprocessing function to use.
        delta_t (int): Period at which tensors are computed (in us).
        height (int): if None the features are not downsampled, however features are downsampled to *height*
            which must be the sensor's height divided by a power of 2.
        width (int): if None the features are not downsampled, however features are downsampled to *width*
            which must be the sensor's width divided by a power of 2.
        start_ts (int): Timestamp (in microseconds) from which the computation begins. Either a single int for all
            files or a list containing exactly one int per file.
        max_duration (int): If not None, limit the duration of the file to *max_duration* us.
        store_as_uint8 (boolean): if True, casts to byte before storing to save space.
            Only supports 0-1 normalized data.
        mode (string): Load by timeslice or number of events. Either "delta_t" or "n_events".
        n_events (int): Number of events in the timeslice.
        preprocess_kwargs (dictionary): A dictionary contains the kwargs used by different preprocessing method
    """
    num_tbins = 1
    dataset_size_increment = 100  # dataset size is increased by this much until we know the actual size
    total_tbins_delta_t = num_tbins * delta_t  # this should be 0 if mode = "n_events"
    if mode == "n_events":
        assert delta_t == 0, "For mode = n_events, delta_t should be 0!"

    extension = ".h5" if max_duration is None else "_{:d}.h5"
    basename = os.path.basename(path).split('.')[0]
    if basename.endswith("_td") or basename.endswith("_cd"):
        basename = basename[:-3]
    hfilename = basename + extension

    hfilename = os.path.join(output_folder, hfilename)
    os.makedirs(output_folder, exist_ok=True)

    if os.path.exists(hfilename):
        print(f"{hfilename} already exists, skip further processing!")
        return

    mv_iterator = EventsIterator(path, start_ts=start_ts, delta_t=total_tbins_delta_t, relative_timestamps=True,
                                 mode=mode, n_events=n_events)

    event_input_height, event_input_width = mv_iterator.get_size()
    height = height if height is not None else event_input_height
    width = width if width is not None else event_input_width

    cd_proc = CDProcessor(height, width, num_tbins=num_tbins, preprocessing=preprocess,
                          downsampling_factor=int(math.log2(event_input_height // height)),
                          preprocess_kwargs=preprocess_kwargs)

    array_dim = cd_proc.get_array_dim()
    preprocess_dtype = cd_proc.get_preprocess_dtype()
    last_ts = 0
    last_n_ev = 0

    if "preprocess_dtype" in preprocess_kwargs:
        preprocess_kwargs.pop('preprocess_dtype')

    attrs = {"events_to_tensor": np.string_(preprocess), "delta_t": np.uint32(delta_t),
             "event_input_width": event_input_width, "event_input_height": event_input_height, "shape": array_dim[1:],
             "mode": mode, "n_events": n_events}
    attrs.update(preprocess_kwargs)

    n_file = 0
    print(f"Processing {os.path.basename(path)}, storing results in {hfilename}")
    h5w = HDF5Writer(hfilename if max_duration is None else hfilename.format(n_file), "data", array_dim[1:],
                     dtype=preprocess_dtype, attrs=attrs, store_as_uint8=store_as_uint8)
    h5w.dataset_size_increment = dataset_size_increment

    for index, events in tqdm(enumerate(mv_iterator), position=1):

        if mode == "n_events" and len(events) == 0:  # this can happen at the end of the buffer if n_events = len(events)
            break
        if mode == "delta_t" and max_duration is not None and index * total_tbins_delta_t - n_file * max_duration >= max_duration:
            # the current file is finished, if data has been produced, we close it
            if h5w.index:
                # finally we reshape the dataset to its exact value
                last_ts = last_ts % max_duration
                total_num_tbins = math.ceil(float(last_ts) / delta_t)
                h5w.set_cursor(max(total_num_tbins, 0))
                h5w.close()

                n_file += 1
                h5w = HDF5Writer(hfilename.format(n_file), "data",
                                 array_dim[1:], dtype=preprocess_dtype, attrs=attrs,
                                 store_as_uint8=store_as_uint8)
                h5w.dataset_size_increment = dataset_size_increment
        elif mode == "n_events" and max_duration is not None:
            # we check if we will exceed max_duration
            if len(events) and last_ts + events['t'][-1] >= max_duration:
                if last_n_ev < n_events:
                    h5w.set_cursor(h5w.index - 1)  # drop the last frame since we don't have enough events
                h5w.close()

                n_file += 1
                h5w = HDF5Writer(hfilename.format(n_file), "data",
                                 array_dim[1:], dtype=preprocess_dtype, attrs=attrs,
                                 store_as_uint8=store_as_uint8)
                h5w.dataset_size_increment = dataset_size_increment
                last_ts = 0
                if events['t'][-1] >= max_duration:
                    print("WARNING: too large slice! reduce n_events or increase max_duration")
                    continue  # we skip this buffer, since it is longer than max_duration

        proc_dt = total_tbins_delta_t if mode == "delta_t" else None
        array = cd_proc(events, delta_t=proc_dt)

        # for n_events mode we need to keep track of timestamps
        last_timestamps = None if mode == "delta_t" else np.array([last_ts + events['t'][-1]])
        h5w.write(array, last_timestamps)

        if index % 100 == 0:
            h5w.flush()  # once in a while we flush
        if events.size:
            if mode == "delta_t":
                last_ts = index * num_tbins * delta_t + events['t'][-1]
            else:
                last_ts = last_ts + events['t'][-1]
            last_n_ev = len(events)

    # finally we reshape the dataset to its exact value
    if mode == "delta_t":
        last_ts = last_ts if max_duration is None else last_ts % max_duration
        total_num_tbins = math.ceil(float(last_ts) / delta_t)
        h5w.set_cursor(max(total_num_tbins, 0))
    elif mode == "n_events":
        if last_n_ev < n_events:
            h5w.set_cursor(h5w.index - 1)  # drop the last frame since we don't have enough events
    h5w.close()


def generate_hdf5(paths, output_folder, preprocess, delta_t, height=None, width=None,
                  start_ts=0, max_duration=None, n_processes=2, box_labels=[],
                  store_as_uint8=False, mode="delta_t", n_events=0, preprocess_kwargs={}):
    """
    Generates HDF5 files of frames at a regular interval for dataset caching.

    It is meant to produce a dataset with a smaller memory footprint, that is therefore easier to load from disk.
    Generated files share the name of the input file but have an HDF5 extension.

    If max_duration is not None the naming of the output_file follows this pattern:
        `"{output_folder}/{path basename}_{index:d}.h5"` where index allows you to distinguish the cut.
    otherwise `"{output_folder}/{path basename}.h5"`

    Example:
        >>> python3 metavision_ml/preprocessing/hdf5.py src_path/test/*.dat --o dst_path/test/ --height_width 480 640 --preprocess histo --box-labels src_path/test/*bbox.npy

    Args:
        paths (string list): Paths of input files.
        output_folder (string): Folder path where data will be written.
        preprocess (string): Name of the preprocessing function to use.
        delta_t (int): Period at which tensor are computed (in us).
        height (int): if None the features are not downsampled, however features are downsampled to *height*
            which must be the sensor's height divided by a power of 2.
        width (int): if None the features are not downsampled, however features are downsampled to *width*
            which must be the sensor's width divided by a power of 2.
        start_ts (int): Timestamp (in microseconds) from which the computation begins. Either a single int for all
            files or a list containing exactly one int per file.
        max_duration (int): If not None, limit the duration of the file to *max_duration* (in us).
        n_processes (int): Number of processes writing files simultaneously.
        box_labels (string list): path to npy box label files that matches each input file. if start_ts or max_duration
            are specified, these files will be cut accordingly.
        store_as_uint8 (boolean): if True, casts to byte before storing to save space.
            Only supports 0-1 normalized data.
        mode (string): Load by timeslice or number of events. Either "delta_t" or "n_events".
        n_events (int): Number of events in the timeslice.
        preprocess_kwargs (dictionary): A dictionary contains the kwargs used by different preprocessing method
    """
    assert not store_as_uint8 or preprocess != "diff", "option store_as_uint8 doesn't support diff"
    if preprocess.endswith("_quantized"):
        assert not store_as_uint8, "option store_as_uint8 shouldn't be used for quantized preprocessing!!!"
    # CHECK INPUT PARAMETERS
    delta_t = 0 if mode == "n_events" else delta_t
    n_events = 0 if mode == "delta_t" else n_events
    if max_duration is not None and delta_t > 0:
        assert max_duration % delta_t == 0, f"max_duration ({max_duration}) must be a multiple of delta_t ({delta_t})"

    paths = [paths] if isinstance(paths, str) else paths
    event_input_height, event_input_width = EventsIterator(paths[0]).get_size()
    for i, path in enumerate(paths):
        assert path.split('.')[-1].lower() in ('dat', 'raw'), f"input file nb{i}: {path} must be .raw or .dat"
        msg = f"All inputs must have the same resolution, input file nb{i}: {path} differs."
        assert [event_input_height, event_input_width] == list(EventsIterator(path).get_size()), msg
    check_input_power_2(event_input_height, event_input_width, height=height, width=width)

    starts_ts = _check_start_ts(start_ts, paths)
    # HANDLE BOX LABELS
    if box_labels:
        box_labels = [box_labels] if isinstance(box_labels, str) else box_labels
        assert len(box_labels) == len(
            paths), f"Error {len(paths)} files were provided but {len(box_labels)} label files."

        for box_path, start_ts in zip(box_labels, starts_ts):
            split_label(box_path, output_folder, start_ts=start_ts, max_duration=max_duration)

    # PRECOMPUTING
    args_list = [{"path": path, "start_ts": start_ts, "output_folder": output_folder, "delta_t": delta_t,
                  "preprocess": preprocess, "height": height, "width": width, "max_duration": max_duration,
                  "store_as_uint8": store_as_uint8, "mode": mode, "n_events": n_events,
                  "preprocess_kwargs": preprocess_kwargs.copy()}
                 for path, start_ts in zip(paths, starts_ts)]

    if n_processes > 1 and len(args_list) > 1:
        pool = Pool(n_processes)
        pool.map(_write, args_list)
    else:
        for di in args_list:
            _write(di)
