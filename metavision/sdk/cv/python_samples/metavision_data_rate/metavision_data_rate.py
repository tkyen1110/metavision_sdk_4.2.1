# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Sample code that demonstrates how to visualizes live data (CD events only) from a Prophesee sensor and estimates the event rate
"""

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_base import EventCDBuffer
from metavision_sdk_cv import AntiFlickerAlgorithm, SpatioTemporalContrastAlgorithm, ActivityNoiseFilterAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent, UIAction
import argparse
import os
import cv2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Data Rate Sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW or DAT file. If not specified, the live stream of the first available camera is used. "
             "If it's a camera serial number, it will try to open that camera instead.")

    parser.add_argument(
        '-t', '--delta-t', dest='delta_t', type=int, default=1e3,
        help="Time interval used to accumulate events")

    parser.add_argument(
        '--fps', dest='fps', type=float, default=50.0,
        help="Display FPS")

    parser.add_argument(
        '--flicker-dt', dest='time_window_flicker', type=int, default=15000,
        help='Time interval threshold for AntiFlickerAlgorithm (in us)'
    )

    parser.add_argument(
        '--st-dt', dest='time_window_stc', type=int, default=15000,
        help='Time interval threshold for SpatioTemporalContrastAlgorithm (in us)'
    )

    parser.add_argument(
        '--activity-dt', dest='time_window_activity', type=int, default=15000,
        help='Time interval threshold for ActivityNoiseFilterAlgorithm (in us)'
    )

    parser.add_argument(
        '--filter-length', type=int, default=7,
        help='Number of successive activations to detect a blinking pattern for AntiFlickerAlgorithm'
    )

    parser.add_argument(
        '--min-freq', type=int, default=70,
        help='Minimum frequency for AntiFlickerAlgorithm'
    )

    parser.add_argument(
        '--max-freq', type=int, default=130,
        help='Maximum frequency for AntiFlickerAlgorithm'
    )

    parser.add_argument(
        '--cut-trail', type=bool, default=True,
        help='whether to cut off all the following events until a change of polarity is detected'
    )

    args = parser.parse_args()
    return args


class AntiFlickerSTCFilter(object):
    """
    Apply AFK and STC filter consecutively

    Args:
        width (int): camera width
        height (int): camera height
        time_window_flicker (int): Time interval threshold for AntiFlickerAlgorithm in us
        time_window_stc (int): Time interval threshold for SpatioTemporalContrastAlgorithm in us
        filter_length (int): Number of successive activations to detect a blinking pattern for AntiFlickerAlgorithm
        min_freq (int) : Minimum frequency for AntiFlickerAlgorithm
        max_freq (int): Maximum frequency for AntiFlickerAlgorithm
        cut_trail (bool): whether to cut off all the following events until a change of polarity is detected.

    """

    def __init__(self, width, height, time_window_flicker=15000, time_window_stc=10000,
                 filter_length=7, min_freq=70, max_freq=130, cut_trail=True):
        self.anti_flicker = AntiFlickerAlgorithm(width, height, filter_length, min_freq, max_freq, time_window_flicker)
        self.spatio_temporal_filter = SpatioTemporalContrastAlgorithm(width, height, time_window_stc, cut_trail)
        self.event_buffer = EventCDBuffer()

    def process_events(self, evs, st_buffer):
        self.anti_flicker.process_events(evs, self.event_buffer)
        self.spatio_temporal_filter.process_events(self.event_buffer, st_buffer)


def main():
    """ Main """
    args = parse_args()

    # Show different keyboard options to do noise filtering
    print("Available keyboard options:\n"
          "     -C: Filter events using Anti-Flicker & Spatio-Temporal Algorithm\n"
          "     -F: Filter events using Anti-Flicker Algorithm\n"
          "     -S: Filter events using Spatio-Temporal-Contrast Filter Algorithm\n"
          "     -A: Filter events using ActivityNoise-Filter Algorithm\n"
          "     -E: Show all events\n"
          "     -Q/Escape: Quit the application\n")

    font_face = cv2.FONT_HERSHEY_SIMPLEX  # Font used for text rendering
    font_scale = 0.5  # Font scale used for text rendering
    thickness = 1  # Line thickness used for text rendering
    margin = 4  # Additional space used for text rendering
    # Colors
    color_txt = (219, 226, 228)

    # Position of text in the image
    (_, text_height), baseline = cv2.getTextSize("Data Rate", font_face, font_scale, thickness)
    help_msg_text_pos = (margin, margin + text_height)
    help_text_height = + text_height + baseline  # Maximum text height

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=args.delta_t)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Initiate the filters
    my_filters = {'activity': ActivityNoiseFilterAlgorithm(width, height, args.time_window_activity),
                  'stc': SpatioTemporalContrastAlgorithm(width, height, args.time_window_stc, args.cut_trail),
                  'anti_flicker': AntiFlickerAlgorithm(width, height, args.filter_length, args.min_freq, args.max_freq,
                                                       args.time_window_flicker),
                  'anti_flicker_stc': AntiFlickerSTCFilter(width, height, args.time_window_flicker, args.time_window_stc,
                                                           args.filter_length, args.min_freq, args.max_freq,
                                                           args.cut_trail)
                  }

    events_buf = EventCDBuffer()
    filter_type = None

    # Helper iterator to emulate realtime
    if not is_live_camera(args.input_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=args.fps)

    # Window - Graphical User Interface
    with MTWindow(title="Metavision Event-Rate Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal filter_type
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            elif key == UIKeyEvent.KEY_E:
                filter_type = None
                print("Show all events")
            elif key == UIKeyEvent.KEY_A:
                # Filter events using the activity filter algorithm
                filter_type = 'activity'
                print("Apply ActivityNoise Filter")
            elif key == UIKeyEvent.KEY_F:
                # Filter events using the AFK filter algorithm
                filter_type = 'anti_flicker'
                print("Apply AFK Filter")
            elif key == UIKeyEvent.KEY_S:
                # Filter events using the STC filter algorithm
                filter_type = 'stc'
                print("Apply STC filter")
            elif key == UIKeyEvent.KEY_C:
                # Filter events using the AFK and STC filter algorithm
                filter_type = 'anti_flicker_stc'
                print("Apply combined AFK and STC Filter")

        window.set_keyboard_callback(keyboard_cb)

        counter = 0
        ts_start = 0
        ts_end = 0

        def on_cd_frame_cb(ts, cd_frame):  # callback function
            nonlocal counter, ts_start, ts_end
            if ts_end > ts_start:
                dt = ts_end - ts_start
                kevps = 1000 * counter / dt
                if kevps > 20:
                    if kevps < 1000:
                        event_rate_txt = "{:.1f} Kev/s".format(kevps)
                    else:
                        event_rate_txt = "{:.1f} Mev/s".format(kevps / 1000)
                else:
                    event_rate_txt = "0 Kev/s"
                y_text_pos = help_msg_text_pos[1]
                cv2.putText(cd_frame, event_rate_txt, (help_msg_text_pos[0], y_text_pos), font_face,
                            font_scale, color_txt, thickness, cv2.LINE_AA)
                y_text_pos += help_text_height + margin
                ts_start = ts_end + 1
            counter = 0
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # Process events
        for evs in mv_iterator:
            if evs.size == 0:
                continue
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            ts_end = evs['t'][-1]
            if filter_type is not None:
                my_filters[filter_type].process_events(evs, events_buf)
                counter += events_buf.numpy().size
                event_frame_gen.process_events(events_buf)
            else:
                counter += evs.size
                event_frame_gen.process_events(evs)

            if window.should_close():
                break


if __name__ == "__main__":
    main()
