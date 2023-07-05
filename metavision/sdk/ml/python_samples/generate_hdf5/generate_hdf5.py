# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Script to write HDF5 feature files from event files
"""
import argparse
import numpy as np

from metavision_ml.preprocessing import get_preprocess_function_names, get_preprocess_dict
from metavision_ml.preprocessing.hdf5 import generate_hdf5


def parse_args(argv=None, only_default_values=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Convert one or multiple RAW or DAT files to precomputed tensor'
                                     ' features in HDF5 datasets.')

    parser.add_argument('path', nargs="+",
                        help='RAW or DAT filenames. You can use shell wildcards to select more than one file.')
    parser.add_argument('-o', '--output-folder', required=True, help='where the hdf5 is going to be written')
    parser.add_argument('--delta-t', type=int, default=50000,
                        help='duration of timeslice (in us) in which events are accumulated'
                        ' to compute features.')
    parser.add_argument('--start-ts', type=int, default=0, nargs="+",
                        help='timestamp (in us) from which the computation begins. '
                        'Either a single int for all files or exactly one int per input file.')
    parser.add_argument(
        '--max-duration-ms', type=int, default=None,
        help='maximum duration of the hdf5 file in ms. if the input file exceeds this duration multiple'
        'files will be produced.')
    parser.add_argument('-n', '--num-workers', type=int, default=2,
                        help='Number of processes used for precomputation')
    parser.add_argument('--preprocess', default='histo', help='name of the preprocessing function used',
                        choices=get_preprocess_function_names())
    parser.add_argument('--height_width', nargs=2, default=None, type=int,
                        help="if set, downscale the feature tensor to the requested resolution using interpolation"
                        " Possible values are only power of two of the original resolution.")
    parser.add_argument(
        '--box-labels', nargs="*", default=[], type=str, help="Optional box label files for the ground truth "
        "that goes along the input files. if `start_ts` or `max_duration` are specified, these files will"
        " be cut accordingly. You can use shell wildcards to select more than one file.")
    parser.add_argument('--store_as_uint8', action="store_true", help="Use quantization to store the underlying data "
                        "as 8bit integer and therefore save space. This will reduce precision of the features.")

    parser.add_argument('--max_val', default=5., type=float, help="maximum number of increments per pixel")
    parser.add_argument(
        '--neg_bit_len_quantized', default=4, type=int,
        help="Use only for Diff3D and Histo3D: the negative bits length set by the camera")
    parser.add_argument(
        '--total_bit_len_quantized', default=8, type=int,
        help="Use only for Histo3D: total bits length used by the camera")
    parser.add_argument(
        '--normalization_quantized', action="store_true",
        help="Use only for Diff3D and Histo3D: If not used, the preprocess dtype shoule be set as int8 for Diff3D mode "
        "and uint8 for Histo3D mode. If used, the output will be normalized and the preprocess dtype "
        "shoule be set as one of the floating types, such as float32.")
    parser.add_argument('--n_events', type=int, default=0,
                        help='Number of events in the timeslice, if "mode" is "n_events".')
    parser.add_argument('--mode', type=str, default="delta_t",
                        help='Load by timeslice or number of events. Either "delta_t" or "n_events".',
                        choices=("delta_t", "n_events"))

    parser.add_argument(
        '--simu_sensor_output_quantized', action="store_true",
        help="Simulate the Diff3D and Histo3D event frames output directly from the sensor. "
        "In this mode, event frames are saved in integer values and in original resolution. Thus, "
        "parameter --height_width should not be set")

    return parser.parse_args(argv) if argv is not None else parser.parse_args()


if __name__ == '__main__':

    ARGS = parse_args()
    [height, width] = ARGS.height_width if ARGS.height_width is not None else [None, None]

    if ARGS.simu_sensor_output_quantized:
        assert height is None and width is None, "Downsampling is not allowed if we simulate sensor ouput!!!"
        assert not ARGS.normalization_quantized, "Normalization is not allowed if we simulate sensor ouput!!!"

    preprocess_kwargs = get_preprocess_dict(ARGS.preprocess)['kwargs']
    if "preprocess_dtype" in preprocess_kwargs:
        preprocess_kwargs.pop('preprocess_dtype')
    if ARGS.preprocess == "diff_quantized":
        preprocess_kwargs.update({"negative_bit_length": ARGS.neg_bit_len_quantized,
                             "normalization": ARGS.normalization_quantized,
                             "preprocess_dtype": np.int8 if ARGS.simu_sensor_output_quantized else np.float32})
    elif ARGS.preprocess == "histo_quantized":
        preprocess_kwargs.update({"negative_bit_length": ARGS.neg_bit_len_quantized,
                             "total_bit_length": ARGS.total_bit_len_quantized,
                             "normalization": ARGS.normalization_quantized,
                             "preprocess_dtype": np.uint8 if ARGS.simu_sensor_output_quantized else np.float32})
    else:
        preprocess_kwargs.update({"max_incr_per_pixel": ARGS.max_val})


    generate_hdf5(ARGS.path, ARGS.output_folder, ARGS.preprocess, ARGS.delta_t,
                  height=height, width=width, start_ts=ARGS.start_ts,
                  max_duration=ARGS.max_duration_ms * 1000 if ARGS.max_duration_ms else None,
                  box_labels=ARGS.box_labels, n_processes=ARGS.num_workers,
                  store_as_uint8=ARGS.store_as_uint8, mode=ARGS.mode, n_events=ARGS.n_events,
                  preprocess_kwargs=preprocess_kwargs)
