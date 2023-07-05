# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Utility class to generate and display an image from counting results and allow the user to interact with the keyboard.
"""

import cv2
import numpy as np

from metavision_sdk_analytics import CountingDrawingHelper
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_ui import BaseWindow, MTWindow, UIAction, UIKeyEvent


class CountingGUI:
    """
    Utility class to generate and display an image from counting results and allow the user to interact with the keyboard.

    Args:
        width (int): Sensor's width.
        height (int): Sensor's height.
        accumulation_time_us (int): Accumulation time of the event buffer before the counting algorithm processing (in us).
        rows ([int]): Line counters' ordinates.
        notification_sampling (int): Minimal number of counted objects between each notification.
        inactivity_time (int): Time of inactivity in us (no counter increment) to be notified.
        outvideo (str): Path to output AVI file to save slow motion video.
    """

    def __init__(self, width, height, accumulation_time_us, rows, notification_sampling, inactivity_time, out_video):
        # Video Writer
        self.out_video = out_video
        if self.out_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_name = out_video + ".avi"
            self.video_writer = cv2.VideoWriter(self.video_name, fourcc, 20, (width, height))

        # Event Frame Generator
        self._events_frame_gen_algo = OnDemandFrameGenerationAlgorithm(width, height, accumulation_time_us)
        self._output_img = np.zeros((height, width, 3), np.uint8)

        # Counting Drawing Helper
        self._counting_drawing_helper = CountingDrawingHelper()
        self._counting_drawing_helper.add_line_counters(rows)

        # Window
        self._window = MTWindow(title="Counting", width=width, height=height,
                                mode=BaseWindow.RenderMode.BGR, open_directly=True)

        self._window.show_async(self._output_img)

        # Key-event callback
        self._last_time_notif = 0
        self._time_notif_step = 1e6
        self._last_count_cb = 0
        self._notification_sampling = notification_sampling
        self._last_ts_cb = 0
        self._inactivity_time = inactivity_time

        self._on_reset_cb = lambda _: None

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                self._window.set_close_flag()
            elif key == UIKeyEvent.KEY_R:
                print("Reset Counter")
                self._on_reset_cb()  # Reset callback
            elif key == UIKeyEvent.KEY_P:
                self._notification_sampling += 1
                print("Setting notification sampling to {}".format(
                    self._notification_sampling))
            elif key == UIKeyEvent.KEY_M:
                if self._notification_sampling >= 2:
                    self._notification_sampling -= 1
                    print("Setting notification sampling to {}".format(
                        self._notification_sampling))

        self._window.set_keyboard_callback(keyboard_cb)

    def set_on_reset_cb(self, on_reset_cb):
        """
        Sets the callback to be called when the user wants to reset the app.

        Args:
            on_reset_cb (Function that takes no argument): Callback to be called when the user wants to reset the app.

        """
        self._on_reset_cb = on_reset_cb

    def process_events(self, events):
        """
        Processes a buffer of events with the Counting algorithm.
        The counting callback will automatically be called once a counting result is available.

        Args:
            events (np.array): Input buffer of events.
        """
        self._events_frame_gen_algo.process_events(events)

    def show(self, ts, count, last_count_ts):
        """
        Generates and displays the results in the window.

        Args:
            ts (int): Timestamp.
            count (int): Global counter.
            last_count_ts
        """
        # Generate image and display it
        self._events_frame_gen_algo.generate(ts, self._output_img)
        self._counting_drawing_helper.draw(ts, count, self._output_img)

        self._window.show_async(self._output_img)
        if self.out_video:
            self.video_writer.write(self._output_img)

        # Time
        if ts >= self._last_time_notif + self._time_notif_step:
            print("Current time: {}".format(ts))
            self._last_time_notif += self._time_notif_step

        # Count
        if count >= self._last_count_cb + self._notification_sampling:
            print("At {} counter is {}".format(ts, count))
            self._last_count_cb += int((count - self._last_count_cb) /
                                       self._notification_sampling) * self._notification_sampling

        # Inactivity
        self._last_ts_cb = max(self._last_ts_cb, last_count_ts)
        if ts >= self._last_ts_cb + self._inactivity_time:
            print("At {} inactivity period".format(ts))
            self._last_ts_cb += int((ts - self._last_ts_cb) /
                                    self._inactivity_time) * self._inactivity_time

    def should_close(self):
        """
        Indicates whether the window has been asked to close.

        Returns:
            True if the window should close, False otherwise
        """
        return self._window.should_close()

    def destroy_window(self):
        """
        Destroys the current window.
        This function must only be called from the main thread.
        """
        self._window.destroy()
        if self.out_video:
            self.video_writer.release()
            print("Video has been saved in " + self.video_name)
