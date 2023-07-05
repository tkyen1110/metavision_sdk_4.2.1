# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to count and estimate the size of falling particles.
In offline mode it will display a window with a visualization of the events and the Particle Size Measurement results.
Particle detections will be printed in the console.
"""


import cv2
from math import tan
import numpy as np

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_analytics import PsmAlgorithm, LineClusterTrackingConfig, LineParticleTrackingConfig, \
    LineClustersOutputView, LineParticleTrackingOutput, LineParticleTrack, \
    CountingDrawingHelper, LineParticleTrackDrawingHelper, LineClusterDrawingHelper
from metavision_sdk_core import PolarityFilterAlgorithm, OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TransposeEventsAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Particle Size Measurement sample.',
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
    base_options.add_argument(
        '--roi-line', dest='roi_line_half_width', type=int, default=None,
        help="If specified, it sets hardware lines of interest with a live camera; if not, all the events are used."
        "It corresponds to the half-width of each counting lines of interest to set, i.e. the number of rows "
        "of pixels to add on each side of a line of interest to make the ROI thicker.")

    # Filtering options
    filter_options = parser.add_argument_group('Filtering options')
    filter_options.add_argument('--activity-ths', dest='activity_ths', type=int, default=0,
                                help='Length of the time window for activity filtering (in us).')
    filter_options.add_argument('--polarity', dest='polarity', type=str, default="OFF", choices=("OFF", "ON", "ALL"),
                                help='Which event polarity to process. By default it uses only OFF events.')
    filter_options.add_argument(
        '-r', '--rotate', dest='rotate', default=False, action='store_true',
        help='Rotate the camera 90 degrees clockwise in case of particles moving horizontally in FOV.')

    # Detection options
    detection_options = parser.add_argument_group('Detection options')
    detection_options.add_argument('-a', '--accumulation-time', dest='accumulation_time', type=int, default=200,
                                   help='Accumulation time in us (temporal length of the processed event-buffers).')
    detection_options.add_argument('-p', '--precision-time', dest='precision_time', type=int, default=30,
                                   help='Precision time in us (time duration between two asynchronous processes).')
    detection_options.add_argument(
        '--cluster-ths', dest='cluster_ths', type=int, default=3,
        help='Minimum width (in pixels) below which clusters of events along the line are considered as noise.')
    detection_options.add_argument(
        '--num-clusters-ths', dest='num_clusters_ths', type=int, default=4,
        help='Minimum number of cluster measurements below which a particle is considered as noise.')
    detection_options.add_argument(
        '--min-inter-clusters-dist', dest='min_inter_clusters_dist', type=int, default=1,
        help='Once small clusters have been removed, merge clusters that are closer than this distance.')
    detection_options.add_argument(
        '--learning-rate', dest='learning_rate', type=float, default=0.8,
        help="Ratio in the weighted mean between the current x position and the observation."
        " This is used only when the particle is shrinking. 0.0 is conservative and does not take the observation"
        " into account, whereas 1.0 has no memory and overwrites the cluster estimate with the new observation.")
    detection_options.add_argument(
        '--clamping',
        dest='clamping',
        type=float,
        default=5.0,
        help='Threshold that caps x variation at this value. A negative value disables the clamping. This is used only when the particle is shrinking.')

    # Tracking options
    tracking_options = parser.add_argument_group('Tracking options')
    tracking_options.add_argument('-n', '--num-lines', dest='num_lines', type=int, default=6,
                                  help='Number of lines for counting between min-y and max-y.')
    tracking_options.add_argument('--min-y', dest='min_y_line', type=int, default=200,
                                  help='Ordinate at which to place the first line counter.')
    tracking_options.add_argument('--max-y', dest='max_y_line', type=int, default=300,
                                  help='Ordinate at which to place the last line counter.')
    tracking_options.add_argument('-u', '--objects-moving-up', dest='is_going_up', default=False, action='store_true',
                                  help='Specify if the particles are going upwards.')
    tracking_options.add_argument('--first-match-dt', dest='first_match_dt', type=int, default=100000,
                                  help='Maximum allowed duration to match the 2nd particle of a track.')
    tracking_options.add_argument(
        '--max-angle-deg', dest='max_angle_deg', type=int, default=45,
        help="Angle with the vertical beyond which two particles on consecutive lines can't be matched.")
    tracking_options.add_argument('--matching-ths', dest='matching_ths', type=float, default=0.5,
                                  help='Minimum similarity score in [0,1] needed to match two particles.')

    # Outcome Options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
        '--no-display', dest='no_display', default=False, action='store_true',
        help="Disable the GUI when reading a RAW (no effect with a live camera where GUI is already disabled).")
    outcome_options.add_argument(
        '--persistence-contour', dest='persistence_contour', type=int, default=40,
        help='Once a particle contour has been estimated, keep the drawing superimposed on the display for a given number of frames.')
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting slow motion video. A frame is generated after each process of the algorithm. The video "
        "will be written only for processed events. When the display is disabled, i.e. either with a live camera or when --no-display has "
        "been specified, frames are not generated, so the video can't be generated either.")

    # Replay Option
    replay_options = parser.add_argument_group('Replay options')
    replay_options.add_argument(
        '-f', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")

    args = parser.parse_args()

    if args.max_y_line <= args.min_y_line:
        print(f"The range of y-positions for the line counters is not valid: [{args.min_y_line_}, {args.max_y_line_}]")
        exit(1)

    if args.out_video and args.no_display:
        print("Try to generate an output video whereas the display is disabled")
        exit(1)

    return args


def main():
    """ Main """
    args = parse_args()

    print(
        "Code sample for Particle Size Measurement algorithm on a stream of events from an event-based device or "
        "recorded data.\n\n"
        "By default, this samples uses only OFF events and assumes that the particles are moving vertically in FOV.\n"
        "In case of different configuration, the default parameters should be adjusted.\n"
        "Please note that the GUI is displayed only when reading RAW files "
        "as it would not make sense to generate frames at such high frequency\n\n"
        "Press 'q' or Escape key to leave the program.\n"
        "Press 'r' to reset the counter.\n")

    # Detection_lines
    y_line_step = int((args.max_y_line - args.min_y_line) / (args.num_lines - 1))
    detection_rows = [args.min_y_line + k * y_line_step for k in range(args.num_lines)]
    print("Detection rows: " + str(detection_rows))

    # Events iterator on Camera or RAW file
    device = initiate_device(path=args.input_path)
    i_geometry = device.get_i_geometry()  # Camera Geometry
    width = i_geometry.get_width()
    height = i_geometry.get_height()
    i_roi = device.get_i_roi()
    # Set camera ROI in case of a live stream
    if not (i_roi is None or args.roi_line_half_width is None):
        expanded_detection_rows = [
            x + k for x in detection_rows for k in range(-args.roi_line_half_width, args.roi_line_half_width + 1)]
        if args.rotate:  # Transpose
            cols_rois = [0] * width
            for id in expanded_detection_rows:
                cols_rois[id] = 1
            i_roi.set_lines(cols=cols_rois, rows=[1] * height)
            i_roi.enable(True)
            print("Columns of interest: " + str(cols_rexpanded_detection_rowsois))
        else:
            rows_rois = [1] * height
            for id in expanded_detection_rows:
                rows_rois[id] = 1
            i_roi.set_lines(cols=[1] * width, rows=rows_rois)
            i_roi.enable(True)
            print("Rows of interest: " + str(expanded_detection_rows))
    mv_iterator = EventsIterator.from_device(
        device=device, start_ts=args.process_from, max_duration=args.process_to - args.process_from
        if args.process_to else None, delta_t=1e3)

    if args.replay_factor > 0 and not is_live_camera(args.input_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)

    # List of pre-processing filters that will be applied to events
    filtering_algorithms = []
    if args.polarity != "ALL":  # Polarity Filter
        filtering_algorithms.append(
            PolarityFilterAlgorithm(polarity=0 if args.polarity == "OFF" else 1))

    if args.rotate:  # Transpose Filter
        filtering_algorithms.append(TransposeEventsAlgorithm())
        height, width = width, height  # Swap width and height

    if args.activity_ths != 0:  # Activity Noise Filter
        filtering_algorithms.append(ActivityNoiseFilterAlgorithm(width=width, height=height,
                                                                 threshold=args.activity_ths))
    if filtering_algorithms:
        events_buf = PolarityFilterAlgorithm.get_empty_output_buffer()

    # Configurations
    detection_config = LineClusterTrackingConfig(precision_time_us=args.precision_time,
                                                 bitsets_buffer_size=int(args.accumulation_time / args.precision_time),
                                                 cluster_ths=args.cluster_ths,
                                                 num_clusters_ths=args.num_clusters_ths,
                                                 min_inter_clusters_distance=args.min_inter_clusters_dist,
                                                 learning_rate=args.learning_rate,
                                                 max_dx_allowed=args.clamping,
                                                 max_nbr_empty_rows=0)

    tracking_config = LineParticleTrackingConfig(is_going_down=not args.is_going_up,
                                                 dt_first_match_ths=args.first_match_dt,
                                                 tan_angle_ths=tan(args.max_angle_deg * 3.14 / 180),
                                                 matching_ths=args.matching_ths)

    # PSM Algorithm
    psm_algo = PsmAlgorithm(width=width, height=height,
                            rows=detection_rows,
                            detection_config=detection_config,
                            tracking_config=tracking_config,
                            num_process_before_matching=3)

    # Display is automatically disabled with a live camera
    if args.input_path == "":
        args.no_display = True

    if args.no_display:
        # Callback printing results of the PSM algorithm
        def psm_cb(ts, tracks, line_clusters):
            particle_sizes = []
            for track in tracks:
                particle_sizes.append("{:.1f}".format(track.particle_size))

            if particle_sizes:
                print(f"At {ts}, the counter is {tracks.global_counter}. New particle sizes (in pix): ["
                      + ", ".join(particle_sizes) + "]")

        psm_algo.set_output_callback(psm_cb)

        # Process events
        for evs in mv_iterator:
            # Process events
            if filtering_algorithms:
                filtering_algorithms[0].process_events(evs, events_buf)
                for filter in filtering_algorithms[1:]:
                    filter.process_events_(events_buf)
                psm_algo.process_events(events_buf)
            else:
                psm_algo.process_events(evs)
    else:
        # PSM Drawing Helpers
        counting_drawing_helper = CountingDrawingHelper()
        counting_drawing_helper.add_line_counters(detection_rows)
        detection_drawing_helper = LineClusterDrawingHelper()
        tracking_drawing_helper = LineParticleTrackDrawingHelper(
            width=width, height=height, persistence_time_us=args.persistence_contour * args.precision_time)

        # Event Frame Generator
        events_frame_gen_algo = OnDemandFrameGenerationAlgorithm(width, height, args.accumulation_time)
        output_img = np.zeros((height, width, 3), np.uint8)

        # Window - Graphical User Interface (Display PSM results and process keyboard events)
        with MTWindow(title="Particle Size Measurement", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
            window.show_async(output_img)

            if args.out_video:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                video_name = args.out_video + ".avi"
                video_writer = cv2.VideoWriter(video_name, fourcc, 20, (width, height))

            # Callback displaying results of the PSM algorithm
            # [PSM_CALLBACK_BEGIN]
            def psm_cb(ts, tracks, line_clusters):
                events_frame_gen_algo.generate(ts, output_img)

                counting_drawing_helper.draw(ts=ts, count=tracks.global_counter, image=output_img)
                detection_drawing_helper.draw(image=output_img, line_clusters=line_clusters)
                tracking_drawing_helper.draw(ts=ts, image=output_img, tracks=tracks)
                particle_sizes = []
                for track in tracks:
                    particle_sizes.append("{:.1f}".format(track.particle_size))

                if particle_sizes:
                    print(f"At {ts}, the counter is {tracks.global_counter}. New particle sizes (in pix): ["
                          + ", ".join(particle_sizes) + "]")

                window.show_async(output_img)
                if args.out_video:
                    video_writer.write(output_img)
            # [PSM_CALLBACK_END]

            psm_algo.set_output_callback(psm_cb)

            def keyboard_cb(key, scancode, action, mods):
                if action != UIAction.RELEASE:
                    return
                if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                    window.set_close_flag()
                elif key == UIKeyEvent.KEY_R:
                    psm_algo.reset()

            window.set_keyboard_callback(keyboard_cb)
            print("Press 'q' to leave the program.\n"
                  "Press 'r' to reste the PSM algorithm.")

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
                    events_frame_gen_algo.process_events(events_buf)
                    psm_algo.process_events(events_buf)
                else:
                    events_frame_gen_algo.process_events(evs)
                    psm_algo.process_events(evs)

                if window.should_close():
                    break
            # [MV_ITERATOR_END]

            if args.out_video:
                video_writer.release()
                print("Video has been saved in " + video_name)


if __name__ == "__main__":
    main()
