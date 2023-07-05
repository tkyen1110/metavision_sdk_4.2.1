# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Code sample showing how to create a simple application testing different noise filtering strategies.
"""

from enum import Enum
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm, SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


activity_time_ths = 20000  # Length of the time window for activity filtering (in us)
activity_ths = 2  # Minimum number of events in the neighborhood for activity filtering
activity_trail_ths = 1000  # Length of the time window for trail filtering (in us)

trail_filter_ths = 1000000  # Length of the time window for activity filtering (in us)

stc_filter_ths = 10000  # Length of the time window for filtering (in us)
stc_cut_trail = True  # If true, after an event goes through, it removes all events until change of polarity


class Filter(Enum):
    NONE = 0,
    ACTIVITY = 1,
    STC = 2,
    TRAIL = 3


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Noise Filtering sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    parser.add_argument(
        '-r', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    print("Code sample showing how to create a simple application testing different noise filtering strategies.")
    print("Available keyboard options:\n"
          "  - A: Filter events using the activity noise filter algorithm\n"
          "  - T: Filter events using the trail filter algorithm\n"
          "  - S: Filter events using the spatio temporal contrast algorithm\n"
          "  - E: Show all events\n"
          "  - Q/Escape: Quit the application\n")

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)

    # Set ERC to 20Mev/s
    if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device:
        erc_module = mv_iterator.reader.device.get_i_erc_module()
        if erc_module:
            erc_module.set_cd_event_rate(20000000)
            erc_module.enable(True)

    if args.replay_factor > 0 and not is_live_camera(args.input_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # Camera Geometry

    filters = {Filter.ACTIVITY: ActivityNoiseFilterAlgorithm(width, height, activity_time_ths),
               Filter.TRAIL: TrailFilterAlgorithm(width, height, trail_filter_ths),
               Filter.STC: SpatioTemporalContrastAlgorithm(width, height, stc_filter_ths, stc_cut_trail)
               }

    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    filter_type = Filter.NONE

    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=10000)

    # Window - Graphical User Interface (Display filtered events and process keyboard events)
    with MTWindow(title="Metavision Noise Filtering", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        def keyboard_cb(key, scancode, action, mods):
            nonlocal filter_type

            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            elif key == UIKeyEvent.KEY_E:
                # Show all events
                filter_type = Filter.NONE
            elif key == UIKeyEvent.KEY_A:
                # Filter events using the activity filter algorithm
                filter_type = Filter.ACTIVITY
            elif key == UIKeyEvent.KEY_T:
                # Filter events using the trail filter algorithm
                filter_type = Filter.TRAIL
            elif key == UIKeyEvent.KEY_S:
                # Filter events using the spatio temporal contrast algorithm
                filter_type = Filter.STC

        window.set_keyboard_callback(keyboard_cb)

        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            # Process events
            if filter_type in filters:
                filters[filter_type].process_events(evs, events_buf)
                event_frame_gen.process_events(events_buf)
            else:
                event_frame_gen.process_events(evs)

            if window.should_close():
                break


if __name__ == "__main__":
    main()
