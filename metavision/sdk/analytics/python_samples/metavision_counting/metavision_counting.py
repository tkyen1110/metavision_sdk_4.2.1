# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to count small objects.
In offline mode it will display a window with a visualization of the events and the line counters.
You can use it, for example, with the reference file 80_balls.raw.
"""


from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import CountingAlgorithm, CountingCalibration
from metavision_sdk_core import PolarityFilterAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TransposeEventsAlgorithm
from metavision_sdk_ui import EventLoop

from counting_gui import CountingGUI


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Counting sample.',
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

    # Filtering options
    filter_options = parser.add_argument_group('Filtering options')
    filter_options.add_argument('--activity-ths', dest='activity_ths', type=int, default=0,
                                help='Length of the time window for activity filtering (in us).')
    filter_options.add_argument('--polarity', dest='polarity', type=str, default="OFF", choices=("OFF", "ON", "ALL"),
                                help='Which event polarity to process. By default it uses only OFF events.')
    filter_options.add_argument(
        '-r', '--rotate', dest='rotate', default=False, action='store_true',
        help='Rotate the camera 90 degrees clockwise in case of particles moving horizontally in FOV.')

    # Calibration options
    calib_options = parser.add_argument_group('Calibration options')
    calib_options.add_argument('--object-min-size', dest='object_min_size', type=float, default=6.,
                               help='Approximate minimum size of an object to count (its largest dimension in mm).')
    calib_options.add_argument('--object-average-speed', dest='object_average_speed', type=float, default=5.,
                               help='Approximate average speed of an object to count in meters per second.')
    calib_options.add_argument(
        '--distance-object-camera', dest='distance_object_camera', type=float, default=300.,
        help='Average distance between the flow of objects to count and the camera (distance in mm).')

    # Replay Option
    replay_options = parser.add_argument_group('Replay options')
    replay_options.add_argument(
        '-f', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")

    # Algorithm options
    algo_options = parser.add_argument_group('Algorithm options')
    algo_options.add_argument('-n', '--num-lines', dest='num_lines', type=int, default=4,
                              help='Number of lines for counting between min-y and max-y.')
    algo_options.add_argument('--min-y', dest='min_y_line', type=int, default=150,
                              help='Ordinate at which to place the first line counter.')
    algo_options.add_argument('--max-y', dest='max_y_line', type=int, default=330,
                              help='Ordinate at which to place the last line counter.')

    # Outcome options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
        '--no-display', dest='no_display', default=False, action='store_true',
        help="Disable the GUI when reading a RAW (no effect with a live camera where GUI is already disabled).")
    outcome_options.add_argument('--notification-sampling', dest='notification_sampling', type=int, default=1,
                                 help='Minimal number of counted objects between each notification.')
    outcome_options.add_argument('--inactivity-time', dest='inactivity_time', type=int, default=1000000,
                                 help='Time of inactivity in us (no counter increment) to be notified.')
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting slow motion video. A frame is generated after each process of the algorithm. The video "
        "will be written only for processed events. When the display is disabled, i.e. either with a live camera or when --no-display has "
        "been specified, frames are not generated, so the video can't be generated either.")

    args = parser.parse_args()

    if args.max_y_line <= args.min_y_line:
        print(f"The range of y-positions for the line counters is not valid: [{args.min_y_line_}, {args.max_y_line_}]")
        exit(1)

    if args.process_to and args.process_from > args.process_to:
        print(f"The processing time interval is not valid. [{args.process_from,}, {args.process_to}]")
        exit(1)

    if args.out_video and args.no_display:
        print("Try to generate an output video whereas the display is disabled")
        exit(1)

    return args


def main():
    """ Main """
    args = parse_args()

    print("Code sample for counting algorithm on a stream of events from an event-based device or recorded data.\n\n"
          "By default, this samples uses only OFF events and assumes that the objects are moving vertically in FOV.\n"
          "In case of different configuration, the default parameters should be adjusted.\n"
          "Please note that the GUI is displayed only when reading RAW files "
          "as it would not make sense to generate frames at such high frequency\n\n"
          "Press 'q' or Escape key to leave the program.\n"
          "Press 'r' to reset the counter.\n"
          "Press 'p' to increase (+1) notification sampling (number of objects to be counted between each "
          "notification).\n"
          "Press 'm' to decrease (-1) notification sampling (number of objects to be counted between each "
          "notification).\n")

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, start_ts=args.process_from,
                                 max_duration=args.process_to - args.process_from if args.process_to else None,
                                 delta_t=1e3)

    if args.replay_factor > 0 and not is_live_camera(args.input_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # List of pre-processing filters that will be applied to events
    filtering_algorithms = []
    if args.polarity != "ALL":  # Polarity Filter
        filtering_algorithms.append(PolarityFilterAlgorithm(polarity=0 if args.polarity == "OFF" else 1))

    if args.rotate:  # Transpose Filter
        filtering_algorithms.append(TransposeEventsAlgorithm())
        height, width = width, height  # Swap width and height

    if args.activity_ths != 0:  # Activity Noise Filter
        filtering_algorithms.append(ActivityNoiseFilterAlgorithm(width=width, height=height,
                                                                 threshold=args.activity_ths))

    if filtering_algorithms:
        events_buf = PolarityFilterAlgorithm.get_empty_output_buffer()

    # [COUNTING_CALIBRATION_BEGIN]
    # Counting Calibration (Get optimal algorithm parameters)
    cluster_ths, accumulation_time_us = CountingCalibration.calibrate(
        width=width, height=height, object_min_size=args.object_min_size,
        object_average_speed=args.object_average_speed, distance_object_camera=args.distance_object_camera)
    # [COUNTING_CALIBRATION_END]

    # Counting rows
    counting_rows = []
    y_line_step = int((args.max_y_line - args.min_y_line) / (args.num_lines - 1))
    counting_rows = [args.min_y_line + k * y_line_step for k in range(args.num_lines)]

    # Counting Algorithm
    counting_algo = CountingAlgorithm(width=width, height=height, cluster_ths=cluster_ths,
                                      accumulation_time_us=accumulation_time_us)
    counting_algo.add_line_counters(counting_rows)

    # Display is automatically disabled with a live camera
    if args.input_path == "":
        args.no_display = True

    if args.no_display:
        # Output callback of the counting algorithm
        def counting_cb(ts, global_counter, last_count_ts, line_mono_counters):
            print(f"At {ts} counter is {global_counter}")

        counting_algo.set_output_callback(counting_cb)

        # Process events
        for evs in mv_iterator:
            if filtering_algorithms:
                filtering_algorithms[0].process_events(evs, events_buf)
                for filter in filtering_algorithms[1:]:
                    filter.process_events_(events_buf)
                counting_algo.process_events(events_buf)
            else:
                counting_algo.process_events(evs)
    else:
        # Graphical User Interface (Display counting results and process keyboard events)
        counting_gui = CountingGUI(width=width, height=height,
                                   accumulation_time_us=accumulation_time_us,
                                   rows=counting_rows,
                                   notification_sampling=args.notification_sampling,
                                   inactivity_time=args.inactivity_time,
                                   out_video=args.out_video)

        # Output callback of the counting algorithm
        def counting_cb(ts, global_counter, last_count_ts, line_mono_counters):
            counting_gui.show(ts, global_counter, last_count_ts)

        def on_reset_cb():
            counting_algo.reset_counters()

        counting_algo.set_output_callback(counting_cb)
        counting_gui.set_on_reset_cb(on_reset_cb)

        # [MV_ITERATOR_BEGIN]
        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            # Process events
            if filtering_algorithms:
                filtering_algorithms[0].process_events(evs, events_buf)
                for filter in filtering_algorithms[1:]:
                    filter.process_events_(events_buf)
                counting_gui.process_events(events_buf)
                counting_algo.process_events(events_buf)
            else:
                counting_gui.process_events(evs)
                counting_algo.process_events(evs)

            if counting_gui.should_close():
                break
        # [MV_ITERATOR_END]

        # Important: make sure the window is deleted inside the main thread
        counting_gui.destroy_window()


if __name__ == "__main__":
    main()
