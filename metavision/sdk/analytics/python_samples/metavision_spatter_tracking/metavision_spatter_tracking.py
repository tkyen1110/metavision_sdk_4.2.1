# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to track simple, non colliding objects.
You can use it, for example, with the reference file sparklers.raw.
"""

import cv2
import numpy as np
import csv

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import SpatterTrackerAlgorithm, draw_tracking_results
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Spatter Tracking sample.',
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

    # Algorithm options
    algo_options = parser.add_argument_group('Algorithm options')
    algo_options.add_argument('--cell-width', dest='cell_width', type=int,
                              default=7, help='Cell width used for clustering (in pixels).')
    algo_options.add_argument('--cell-height', dest='cell_height', type=int,
                              default=7, help='Cell height used for clustering (in pixels).')
    algo_options.add_argument('--accumulation-time-us', dest='accumulation_time_us', type=int,
                              default=5000, help='Processing accumulation time (in us).')
    algo_options.add_argument('--untracked-ths', dest='untracked_ths', type=int, default=5,
                              help='Maximum number of times a cluster can stay untracked before being removed.')
    algo_options.add_argument('--activation-ths', dest='activation_ths', type=int, default=10,
                              help='Minimum number of events in a cell to consider it as active.')
    algo_options.add_argument(
        '--disable-filter', dest='disable_filter', default=False, action='store_true',
        help='If not specified, then the cell activation threshold considers only one event per pixel.')
    algo_options.add_argument('--max-dist', dest='max_distance', type=int,
                              default=50, help='Maximum distance for clusters association (in pixels).')
    algo_options.add_argument('--min-size', dest='min_size', type=int,
                              default=10, help='Minimal size of an object to track (in pixels).')
    algo_options.add_argument('--max-size', dest='max_size', type=int,
                              default=300, help='Maximal size of an object to track (in pixels).')

    # Outcome Options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to the resulting video. A frame is generated every time the tracking callback is called.")
    outcome_options.add_argument(
        '-l', '--log-result', dest='out_log', type=str, default="",
        help="File to save the output of tracking.")

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

    # Spatter Tracking Algorithm
    spatter_tracker = SpatterTrackerAlgorithm(width=width, height=height,
                                              cell_width=args.cell_width,
                                              cell_height=args.cell_height,
                                              accumulation_time_us=args.accumulation_time_us,
                                              untracked_threshold=args.untracked_ths,
                                              activation_threshold=args.activation_ths,
                                              apply_filter=not args.disable_filter,
                                              max_distance=args.max_distance,
                                              min_size=args.min_size,
                                              max_size=args.max_size
                                              )

    # Event Frame Generator
    events_frame_gen_algo = OnDemandFrameGenerationAlgorithm(width, height, args.accumulation_time_us)
    output_img = np.zeros((height, width, 3), np.uint8)

    # Window - Graphical User Interface (Display spatter tracking results and process keyboard events)
    with MTWindow(title="Spatter Tracking", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        window.show_async(output_img)

        if args.out_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_name = args.out_video + ".avi"
            video_writer = cv2.VideoWriter(video_name, fourcc, 20, (width, height))

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        log = []

        # [TRACKING_TRACKER_CALLBACK_BEGIN]
        # Output callback of the spatter tracking algorithm
        def spatter_tracking_cb(ts, clusters):
            clusters_np = clusters.numpy()
            for cluster in clusters_np:
                log.append([ts, cluster['id'], int(cluster['x']), int(cluster['y']), int(cluster['width']),
                            int(cluster['height'])])
            events_frame_gen_algo.generate(ts, output_img)
            draw_tracking_results(ts, clusters, output_img)
            window.show_async(output_img)
            if args.out_video:
                video_writer.write(output_img)
        # [TRACKING_TRACKER_CALLBACK_END]

        # [TRACKING_SET_OUTPUT_CALLBACK_BEGIN]
        spatter_tracker.set_output_callback(spatter_tracking_cb)
        # [TRACKING_SET_OUTPUT_CALLBACK_END]

        # [TRACKING_MAIN_LOOP_BEGIN]
        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            # Process events
            events_frame_gen_algo.process_events(evs)
            spatter_tracker.process_events(evs)

            if window.should_close():
                break
        # [TRACKING_MAIN_LOOP_END]

        print("Number of tracked clusters: {}".format(spatter_tracker.get_cluster_count))

        if args.out_video:
            video_writer.release()
            print("Video has been saved in " + video_name)

        if args.out_log:
            with open(args.out_log, mode='w') as f:
                data_writer = csv.writer(f)
                data_writer.writerow(['timestamp', 'id', 'x', 'y', 'width', 'height'])
                data_writer.writerows(log)


if __name__ == "__main__":
    main()
