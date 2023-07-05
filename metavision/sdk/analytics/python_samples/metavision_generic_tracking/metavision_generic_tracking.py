# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to track general objects.
You can use it, for example, with the reference file traffic_monitoring.raw.
"""

import cv2
import numpy as np

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import TrackingAlgorithm, TrackingConfig, draw_tracking_results
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generic Tracking sample.',
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

    parser.add_argument('--update-frequency', dest='update_frequency', type=float,
                        default=200., help="Tracker's update frequency, in Hz.")

    # Min/Max size options
    minmax_size_options = parser.add_argument_group('Min/Max size options')
    minmax_size_options.add_argument('--min-size', dest='min_size', type=int,
                                     default=10, help='Minimal size of an object to track (in pixels).')
    minmax_size_options.add_argument('--max-size', dest='max_size', type=int,
                                     default=300, help='Maximal size of an object to track (in pixels).')

    # Filtering options
    filter_options = parser.add_argument_group('Filtering options')
    filter_options.add_argument(
        '--activity-time-ths', dest='activity_time_ths', type=int, default=10000,
        help='Length of the time window for activity filtering (Disabled if the threshold is equal to 0).')
    filter_options.add_argument('--activity-ths', dest='activity_ths', type=int, default=1,
                                help='Minimum number of events in the neighborhood.')
    filter_options.add_argument('--activity-trail-ths', dest='activity_trail_ths', type=int, default=1000,
                                help='Length of the time window for trail filtering (in us).')

    # Outcome Options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting video. A frame is generated every time the tracking callback is called.")

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

    # Noise + Trail filter that will be applied to events
    activity_noise_filter = ActivityNoiseFilterAlgorithm(width, height, args.activity_time_ths)

    trail_filter = TrailFilterAlgorithm(width, height, args.activity_trail_ths)

    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    # Tracking Algorithm
    tracking_config = TrackingConfig()  # Default configuration
    tracking_algo = TrackingAlgorithm(sensor_width=width, sensor_height=height, tracking_config=tracking_config)
    tracking_algo.update_frequency = args.update_frequency
    tracking_algo.min_size = args.min_size
    tracking_algo.max_size = args.max_size

    # Event Frame Generator
    acc_time = int(1.0e6 / args.update_frequency)
    events_frame_gen_algo = OnDemandFrameGenerationAlgorithm(width, height, acc_time)
    output_img = np.zeros((height, width, 3), np.uint8)

    # Window - Graphical User Interface (Display tracking results and process keyboard events)
    with MTWindow(title="Generic Tracking", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        window.show_async(output_img)

        if args.out_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_name = args.out_video + ".avi"
            video_writer = cv2.VideoWriter(video_name, fourcc, 20, (width, height))

        def keyboard_cb(key, scancode, action, mods):
            SIZE_STEP = 10

            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            # A: Increase minimum size of the object to track
            elif key == UIKeyEvent.KEY_A:
                if args.min_size + SIZE_STEP <= args.max_size:
                    args.min_size += SIZE_STEP
                    print("Increase min size to {}".format(args.min_size))
                    tracking_algo.min_size = args.min_size
            # B: Decrease minimum size of the object to track
            elif key == UIKeyEvent.KEY_B:
                if args.min_size - SIZE_STEP >= 0:
                    args.min_size -= SIZE_STEP
                    print("Decrease min size to {}".format(args.min_size))
                    tracking_algo.min_size = args.min_size
            # C: Increase maximum size of the object to track
            elif key == UIKeyEvent.KEY_C:
                args.max_size += SIZE_STEP
                print("Increase max size to {}".format(args.max_size))
                tracking_algo.max_size = args.max_size
            # D: Decrease maximum size of the object to track
            elif key == UIKeyEvent.KEY_D:
                if args.max_size - SIZE_STEP >= args.min_size:
                    args.max_size -= SIZE_STEP
                    print("Decrease max size to {}".format(args.max_size))
                    tracking_algo.max_size = args.max_size

        window.set_keyboard_callback(keyboard_cb)
        print("Press 'q' to leave the program.\n"
              "Press 'a' to increase the minimum size of the object to track.\n"
              "Press 'b' to decrease the minimum size of the object to track.\n"
              "Press 'c' to increase the maximum size of the object to track.\n"
              "Press 'd' to decrease the maximum size of the object to track.")

        # [GENERIC_TRACKING_TRACKER_CALLBACK_BEGIN]
        # Output callback of the tracking algorithm
        def tracking_cb(ts, tracking_results):
            nonlocal output_img
            events_frame_gen_algo.generate(ts, output_img)
            draw_tracking_results(ts, tracking_results, output_img)
            window.show_async(output_img)
            if args.out_video:
                video_writer.write(output_img)
        # [GENERIC_TRACKING_TRACKER_CALLBACK_END]

        # [GENERIC_TRACKING_SET_OUTPUT_CALLBACK_BEGIN]
        tracking_algo.set_output_callback(tracking_cb)
        # [GENERIC_TRACKING_SET_OUTPUT_CALLBACK_END]

        # [GENERIC_TRACKING_MAIN_LOOP_BEGIN]
        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            # Process events
            activity_noise_filter.process_events(evs, events_buf)
            trail_filter.process_events_(events_buf)
            events_frame_gen_algo.process_events(events_buf)
            tracking_algo.process_events(events_buf)

            if window.should_close():
                break
        # [GENERIC_TRACKING_MAIN_LOOP_END]

        if args.out_video:
            video_writer.release()
            print("Video has been saved in " + video_name)


if __name__ == "__main__":
    main()
