# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Metavision Jet Monitoring Calibration.
Tool to calibrate the Camera and Detection ROIs for the Jet Monitoring sample on a stream of events from an event-based device or recorded data.
"""


from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_cv import TransposeEventsAlgorithm
from metavision_sdk_ui import EventLoop

from jet_monitoring_calibration_gui import JetMonitoringCalibrationGUI


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Jet Monitoring calibration tool.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    parser.add_argument('-v', '--vertical-jets', dest='vertical_jets', default=False, action='store_true',
                        help='Rotate the camera 90 degrees clockwise in case of a nozzle firing jets vertically in the FOV.')
    parser.add_argument('-a', '--accumulation-time', dest='accumulation_time_us', type=int, default=10000,
                        help='Accumulation time (in us) to use to generate a frame.')
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    print(
        "Metavision Jet Monitoring Calibration Tool.\n\n"
        "Make sure the nozzle is in the field of view of the camera and is firing either horizontally or vertically.\n"
        "Press 'Space' when the jet is clearly visible on the display of events.\n"
        "Once ROIs have been drawn on the display, press 'Enter' to print --detection-roi and --camera-roi in the console. "
        "Then, run the Jet Monitoring sample using these two command line arguments.\n\n"
        "Press 'Space' to play/pause events\n"
        "Press 'B' to define the baseline\n"
        "Press 'C' to define the Camera ROI\n"
        "Press 'J' to define the Jet ROI\n"
        "Press 'Enter' to print ROIs\n"
        "Press 'Q' or 'Escape' to exit\n")
    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Transpose Filter
    if args.vertical_jets:
        transpose_filter = TransposeEventsAlgorithm()
        events_buf = TransposeEventsAlgorithm.get_empty_output_buffer()
        height, width = width, height  # Swap width and height

    # Graphical User Interface
    gui = JetMonitoringCalibrationGUI(width, height, args.vertical_jets)

    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, args.accumulation_time_us)

    def on_cd_frame_cb(ts, cd_frame):
        gui.swap_cd_frame_if_required(cd_frame)
        gui.update()

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    # Process events
    for evs in mv_iterator:
        # Dispatch system events to the window
        EventLoop.poll_and_dispatch()

        # Process events
        if args.vertical_jets:
            transpose_filter.process_events(evs, events_buf)
            event_frame_gen.process_events(events_buf)
        else:
            event_frame_gen.process_events(evs)

        if gui.should_close():
            break

    if args.input_path != "":
        print("The RAW file has been entirely processed. The app now uses the last saved cd frame.")

    # Wait until the closing of the window
    while not gui.should_close():
        # Dispatch system events to the window
        EventLoop.poll_and_dispatch()
        gui.update()

    # Important: make sure the window is deleted inside the main thread
    gui.destroy_window()


if __name__ == "__main__":
    main()
