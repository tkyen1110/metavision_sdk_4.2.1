# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to detect, count and timestamp jets that are being dispensed.
In offline mode it will display a window with a visualization of the events and the detection ROI.
Jet detections will be displayed in the console. Likewise, alarms (if set) will warn you when there's something wrong
in the dispensing.
"""

import cv2
import numpy as np

from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_hal import I_ROI
from metavision_sdk_analytics import JetMonitoringAlgorithm, JetMonitoringAlarmConfig, JetMonitoringAlgorithmConfig, \
    JetMonitoringDrawingHelper, EventJet, EventJetAlarm
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


nozzle_orientation_dict = {"Down": JetMonitoringAlgorithmConfig.Orientation.Down,
                           "Up": JetMonitoringAlgorithmConfig.Orientation.Up,
                           "Left": JetMonitoringAlgorithmConfig.Orientation.Left,
                           "Right": JetMonitoringAlgorithmConfig.Orientation.Right}


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Jet Monitoring sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base options
    base_options = parser.add_argument_group('Base options')
    base_options.add_argument(
        '-i', '--input-file', dest='input_path', default="",
        help="Path to input RAW or DAT file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    base_options.add_argument('--process-from', dest='process_from', type=int, default=0,
                              help='Start time to process events (in us).')
    base_options.add_argument('--process-to', dest='process_to', type=int, default=None,
                              help='End time to process events (in us).')
    base_options.add_argument('--delta-t', dest='time_step_us', type=int, default=50,
                              help='Time interval to update the monitoring period (in us).')
    base_options.add_argument('--accumulation-time', dest='accumulation_time_us', type=int, default=500,
                              help='Period over which to accumulate events (in us).'
                                   'This must be set depending on the cycle. '
                                   'It must be slightly lower than the input cycle.')

    # Monitoring options
    monitor_options = parser.add_argument_group('Monitoring Options')
    monitor_options.add_argument('--trigger-on', dest='th_up_kevps', type=int, default=50,
                                 help='Minimum activity to trigger a jet count, in kev/s.'
                                 'If set too high, jets may be missed (activity is never reached). '
                                 )
    monitor_options.add_argument('--trigger-off', dest='th_down_kevps', type=int, default=10,
                                 help='Lower bound activity that defines the end of a jet, in kev/s. '
                                 'If set too low, jets may be missed (activity never ends). '
                                 )
    monitor_options.add_argument('--delay-start', dest='th_up_delay_us', type=int, default=100,
                                 help='Duration threshold to confirm the beginning of a jet, in us.'
                                 )
    monitor_options.add_argument('--delay-end', dest='th_down_delay_us', type=int, default=0,
                                 help='Duration threshold to confirm the end of a jet, in us.'
                                 )

    # ROI options
    roi_options = parser.add_argument_group(
        'Regions of Interest options')
    roi_options.add_argument('--nozzle-orientation', dest='_nozzle_orientation_str',
                             type=str, default="Right", choices=nozzle_orientation_dict.keys(),
                             help="Nozzle orientation in the image reference frame. Jets are moving either upwards, "
                             "downwards, leftwards or rightwards.")
    roi_options.add_argument(
        '--camera-roi', dest='camera_roi', nargs=4, type=int, default=(160, 160, 124, 93),
        help="Camera ROI [Left x, Top y, width, height]. Note that the nozzle orientation doesn't modify or rotate this ROI, "
        "it just indicates the direction in which the jets pass through this area.")
    roi_options.add_argument(
        '--detection-roi', dest='detection_roi', nargs=4, type=int, default=(177, 197, 47, 20),
        help="Detection ROI [Left x, Top y, width, height] must be large enough so that a 'nominal' jet is contained in it. "
        "But not too large so that jet unrelated activity doesn't trigger count. Note that the nozzle orientation doesn't "
        "modify or rotate this ROI, it just indicates the direction in which the jets pass through this area.")

    # Alarm options
    alarm_options = parser.add_argument_group('Alarm options')
    alarm_options.add_argument(
        '--alarm-on-count', dest='alarm_on_count', default=False, action='store_true',
        help='If specified, an alarm will be raised if jets are detected above the --max-expected-count.')
    alarm_options.add_argument('--max-expected-count', dest='max_expected_count', type=int, default=0,
                               help='Maximum expected number of jets.')
    alarm_options.add_argument(
        '--alarm-on-cycle', dest='alarm_on_cycle', default=False, action='store_true',
        help='If specified, an alarm will be raised if cycle time (time between jets) is outside the specified tolerance.')
    alarm_options.add_argument(
        '--expected-cycle-ms', dest='expected_cycle_ms', type=float, default=0,
        help='Expected cycle time (in ms). If set to 0, no alarms will be generated for jet timing.')
    alarm_options.add_argument(
        '--cycle-tol-percentage', dest='cycle_tol_percentage', type=float, default=10,
        help="Cycle tolerance, in percentage. If the time between two successive jets is off the --expected-cycle-ms "
        "by more than this percentage, an alarm will be raised.")

    # Outcome options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
        '--no-display', dest='no_display', default=False, action='store_true',
        help="Disable the GUI when reading an input file (no effect with a live camera where GUI is already disabled).")
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting slow motion video. A frame is generated after each process of the algorithm. The video "
        "will be written only for processed events. When the display is disabled, i.e. either with a live camera or when --no-display has "
        "been specified, frames are not generated, so the video can't be generated either.")

    args = parser.parse_args()

    args.nozzle_orientation = nozzle_orientation_dict[args._nozzle_orientation_str]
    args.detection_roi = tuple(args.detection_roi)
    args.camera_roi = tuple(args.camera_roi)

    if args.process_to and args.process_from > args.process_to:
        print(f"The processing time interval is not valid. [{args.process_from,}, {args.process_to}]")
        exit(1)

    if args.out_video and args.no_display:
        print("Try to generate an output video whereas the display is disabled")
        exit(1)

    return args


def roi_tuple_to_str(roi_tuple):
    return "[{2:d} x {3:d} from ({0:d}, {1:d})]".format(*roi_tuple)


def main():
    """ Main """
    args = parse_args()

    print(
        "This sample detects, counts, and timestamps the jets that are being dispensed.\n"
        "\n"
        "Please note that the GUI is displayed only when reading RAW or DAT files "
        "On the top left, you will see three lines:\n"
        "   - Time elapsed since the beginning of the app: this is the camera time in microseconds\n"
        "   - Current event rate in kEV/s. It varies depending on the activity\n"
        "   - Current jets count\n"
        "An arrow and several rectangles are displayed on the 'GUI':\n"
        "   - The arrow represents the direction in which the nozzle fires jets \n"
        "   - The largest red rectangle represents the --camera-roi, i.e. the area seen by the camera\n"
        "   - The small red rectangle represents the --detection-roi, i.e. the area where the algorithm looks for "
        "peaks in the event-rate\n"
        "   - The two blue rectangles that surround the --detection-roi are the areas used to monitor the background "
        "noise\n"
        "\n"
        "If set, alarms will warn you when there's something wrong.\n"
        "\n")

    # Camera and detection ROIs
    print("Camera ROI   : " + roi_tuple_to_str(args.camera_roi))
    print("Detection ROI: " + roi_tuple_to_str(args.detection_roi))

    # Init Events Iterator
    if args.input_path == "":
        print("Using a live camera")
        device = initiate_device(path=args.input_path)
        i_roi = device.get_i_roi()
        if i_roi is not None:  # Set camera ROI in case of a live stream
            i_roi.set_window(roi=I_ROI.Window(*args.camera_roi))
            i_roi.enable(True)
            mv_iterator = EventsIterator.from_device(device=device, start_ts=args.process_from,
                                                     max_duration=args.process_to - args.process_from
                                                     if args.process_to else None, delta_t=1e3)
    else:
        mv_iterator = EventsIterator(args.input_path, start_ts=args.process_from, delta_t=1e3,
                                     max_duration=args.process_to - args.process_from if args.process_to else None)
    height, width = mv_iterator.get_size()  # Camera Geometry
    print(height, width)

    # Algorithm configuration
    algo_config = JetMonitoringAlgorithmConfig(
        detection_roi=args.detection_roi, nozzle_orientation=args.nozzle_orientation, time_step_us=args.time_step_us,
        accumulation_time_us=args.accumulation_time_us, th_up_kevps=args.th_up_kevps, th_down_kevps=args.th_down_kevps,
        th_up_delay_us=args.th_up_delay_us, th_down_delay_us=args.th_down_delay_us)
    # Alarm configuration
    alarm_config = JetMonitoringAlarmConfig()
    if args.alarm_on_count:
        alarm_config.set_max_expected_count(args.max_expected_count)
    if args.alarm_on_cycle:
        alarm_config.set_expected_cycle_ms(expected_cycle_ms=args.expected_cycle_ms,
                                           cycle_tol_percentage=args.cycle_tol_percentage)
    # Jet Monitoring Algorithm
    jet_monitoring_algo = JetMonitoringAlgorithm(algo_config=algo_config,
                                                 alarm_config=alarm_config)

    # Callbacks printing results of the jet monitoring algorithm into the console
    last_count = 0

    def on_jet_cb(event_jet):
        nonlocal last_count
        last_count = event_jet.count
        print("Event Jet: " + str(event_jet))

    def on_alarm_cb(ev_alarm):
        print("Alarm    : " + str(ev_alarm))

    jet_monitoring_algo.set_on_jet_callback(on_jet_cb)
    jet_monitoring_algo.set_on_alarm_callback(on_alarm_cb)

    # Display is automatically disabled with a live camera
    if args.input_path == "":
        args.no_display = True

    if args.no_display:
        # Process events
        for evs in mv_iterator:
            jet_monitoring_algo.process_events(evs)
    else:
        # Jet Drawing Helper
        jet_drawing_helper = JetMonitoringDrawingHelper(camera_roi=args.camera_roi, jet_roi=args.detection_roi,
                                                        nozzle_orientation=args.nozzle_orientation)
        # Event Frame Generator
        events_frame_gen_algo = OnDemandFrameGenerationAlgorithm(width, height, args.accumulation_time_us)
        output_img = np.zeros((height, width, 3), np.uint8)

        # Window - Graphical User Interface (Display Jet Monitoring results and process keyboard events)
        with MTWindow(title="Jet Monitoring", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
            window.show_async(output_img)

            if args.out_video:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                video_name = args.out_video + ".avi"
                video_writer = cv2.VideoWriter(video_name, fourcc, 20, (width, height))

            def keyboard_cb(key, scancode, action, mods):
                nonlocal last_count
                if action != UIAction.RELEASE:
                    return
                if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                    window.set_close_flag()
                elif key == UIKeyEvent.KEY_R:
                    jet_monitoring_algo.reset_state()
                    last_count = 0

            window.set_keyboard_callback(keyboard_cb)
            print("Press 'q' or 'Escape' to leave the program.\n"
                  "Press 'r' to reset the algorithm.")

            # Callback displaying results of the jet monitoring algorithm

            def on_async_cb(processing_ts, n_processed_events):
                events_frame_gen_algo.generate(processing_ts, output_img)

                er_kevps = (1000 * n_processed_events) / args.accumulation_time_us
                jet_drawing_helper.draw(processing_ts, last_count, int(er_kevps), output_img)
                window.show_async(output_img)
                if args.out_video:
                    video_writer.write(output_img)

            jet_monitoring_algo.set_on_async_callback(on_async_cb)

            # Process events
            for evs in mv_iterator:
                # Dispatch system events to the window
                EventLoop.poll_and_dispatch()

                # Process events
                events_frame_gen_algo.process_events(evs)
                jet_monitoring_algo.process_events(evs)

                if window.should_close():
                    break

            if args.out_video:
                video_writer.release()
                print("Video has been saved in " + video_name)


if __name__ == "__main__":
    main()
