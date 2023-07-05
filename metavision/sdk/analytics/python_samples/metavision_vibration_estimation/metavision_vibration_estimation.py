# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to estimate the frequency of vibrating objects.
It will display a window with a visualization of the events, and
another with the frequency for pixels with periodic motion.
You can use it for example, with the file monitoring_40_50hz.raw
from Metavision Datasets that can be downloaded from our documentation.
"""

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm
from metavision_sdk_ui import EventLoop

from vibration_gui import VibrationGUI


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Vibration estimation sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base options
    base_options = parser.add_argument_group('Base options')
    base_options.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
             "If it's a camera serial number, it will try to open that camera instead.")
    base_options.add_argument('--process-from', dest='process_from', type=int, default=0,
                              help='Start time to process events (in us).')
    base_options.add_argument('--process-to', dest='process_to', type=int, default=None,
                              help='End time to process events (in us).')

    # Estimation mode options
    estimation_options = parser.add_argument_group('Estimation mode options')
    estimation_options.add_argument('--min-freq', dest='min_freq', type=float, default=10,
                                    help='Minimum detected frequency (in Hz).')
    estimation_options.add_argument('--max-freq', dest='max_freq', type=float, default=150,
                                    help='Minimum detected frequency (in Hz).')
    estimation_options.add_argument('--filter-length', dest='filter_length', type=int, default=7,
                                    help='Number of successive periods to detect a vibration.')
    estimation_options.add_argument(
        '--max-period-diff', dest='max_period_diff', type=int, default=1500,
        help='Period stability threshold - the maximum difference (in us) between two periods to be considered the same.')
    estimation_options.add_argument('--update-freq', dest='update_freq_hz', default=25,
                                    type=float, help='Update frequency of the algorithm (in Hz).')
    estimation_options.add_argument(
        '--freq-precision', dest='freq_precision_hz', type=float, default=1,
        help='Precision of frequency calculation - Width of frequency bins in histogram (in Hz).')
    estimation_options.add_argument(
        '--min-pixel-count', dest='min_pixel_count', type=int, default=25,
        help='Minimum number of pixels to consider a frequency "real", i.e not coming from noise')

    # Outcome Options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting video. A frame is generated every time the frequency measurement callback is called.")

    # Replay Option
    replay_options = parser.add_argument_group('Replay options')
    replay_options.add_argument(
        '-f', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")

    args = parser.parse_args()

    if args.process_to and args.process_from > args.process_to:
        print(f"The processing time interval is not valid. [{args.process_from,}, {args.process_to}]")
        exit(1)

    return args


def main():
    """ Main """
    args = parse_args()

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, start_ts=args.process_from,
                                 max_duration=args.process_to - args.process_from if args.process_to else None,
                                 delta_t=1e3)
    if args.replay_factor > 0 and not is_live_camera(args.input_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Graphical User Interface (Display vibration results and process mouse/keyboard events)
    vibration_gui = VibrationGUI(width=width, height=height,
                                 min_freq=args.min_freq,
                                 max_freq=args.max_freq,
                                 freq_precision=args.freq_precision_hz,
                                 min_pixel_count=args.min_pixel_count,
                                 out_video=args.out_video)

    # Frequency algorithm
    frequency_algo = FrequencyMapAsyncAlgorithm(width=width,
                                                height=height,
                                                filter_length=args.filter_length,
                                                min_freq=args.min_freq,
                                                max_freq=args.max_freq,
                                                diff_thresh_us=args.max_period_diff)
    frequency_algo.update_frequency = args.update_freq_hz

    # Output callback of the frequency algorithm
    def freq_map_cb(ts, freq_map):
        vibration_gui.show(freq_map)

    frequency_algo.set_output_callback(freq_map_cb)

    # Process events
    for evs in mv_iterator:
        # Dispatch system events to the window
        EventLoop.poll_and_dispatch()

        # Process events
        frequency_algo.process_events(evs)

        if vibration_gui.should_close():
            break

    # Important: make sure the window is deleted inside the main thread
    vibration_gui.destroy_window()


if __name__ == "__main__":
    main()
