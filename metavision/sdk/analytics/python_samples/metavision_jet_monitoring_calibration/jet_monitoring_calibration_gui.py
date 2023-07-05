# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Utility class acting as a Graphical User Interface for the Jet Monitoring calibration tool.
"""

import cv2
import numpy as np

from metavision_sdk_ui import BaseWindow, MTWindow, UIAction, UIMouseButton, UIKeyEvent
from enum import Enum


class JetMonitoringCalibrationGUI:
    """
    Utility class acting as a Graphical User Interface for the Jet Monitoring calibration tool.

    Args:
        width (int): Sensor's width.
        height (int): Sensor's height.
        transpose_output_rois (bool): Set to true if the nozzle is firing jets vertically in the FOV.
    """
    class State(Enum):
        NONE = 0
        BASELINE = 1
        CAMERA_ROI = 2
        JET_ROI = 3

    def __init__(self, width, height, transpose_output_rois):
        self._width = width
        self._height = height
        self._transpose_output_rois = transpose_output_rois

        # Events frames
        self._update_cd_img = True
        self._last_cd_img = np.zeros((height, width, 3), np.uint8)
        self._front_img = np.zeros((height, width, 3), np.uint8)

        # Horizontal baseline
        self._baseline_y = int(height / 2)  # Start with a baseline in the middle of the image

        # Jet ROI
        self._jet_x = -1
        self._jet_corner_offset = (-1, -1)

        # Camera ROI
        self._cam_x = -1
        self._cam_corner_offset = (-1, -1)

        # self.State
        self._is_initializing_roi_ = False
        self._state = self.State.NONE

        # Window
        self._last_mouse_pos = (-1, -1)
        self._window = MTWindow(title="Jet Monitoring Calibration", width=width, height=height,
                                mode=BaseWindow.RenderMode.BGR, open_directly=True)
        self._window.show_async(self._front_img)

        # Help message
        self._font_face = cv2.FONT_HERSHEY_SIMPLEX  # Font used for text rendering
        self._font_scale = 0.5  # Font scale used for text rendering
        self._thickness = 1  # Line thickness used for text rendering
        self._margin = 3  # Additional space used for text rendering

        # Position of the help message in the image
        (_, text_height), baseline = cv2.getTextSize("Jet Monitoring", self._font_face, self._font_scale, self._thickness)
        self._help_msg_text_pos = (self._margin, self._margin + text_height)
        self._help_text_height = + text_height + baseline  # Maximum text height

        # Colors
        self._color_txt = (219, 226, 228)
        self._color_tmp = (0, 255, 255)
        self._color_baseline = (221, 207, 193)
        self._color_roi = (118, 114, 255)
        self._color_bg_noise_roi = (201, 126, 64)

        # Cursor callback
        def cursor_cb(x, y):
            window_width, window_height = self._window.get_size()
            # The window may have been resized meanwhile. So we map the coordinates to the original window's size.
            mapped_x = x * self._width / window_width
            mapped_y = y * self._height / window_height
            self._last_mouse_pos = (int(mapped_x), int(mapped_y))

        # Mouse callback
        def mouse_cb(button, action, mods):
            if button != UIMouseButton.MOUSE_BUTTON_LEFT:
                return  # Only left click is being used

            if action == UIAction.PRESS:
                self._is_initializing_roi = False
                if self._state == self.State.JET_ROI:
                    self._jet_x = self._last_mouse_pos[0]
                elif self._state == self.State.CAMERA_ROI:
                    self._cam_x = self._last_mouse_pos[0]
            elif action == UIAction.RELEASE:
                if self._state == self.State.BASELINE:
                    self._baseline_y = self._last_mouse_pos[1]
                elif self._state == self.State.JET_ROI:
                    self._jet_corner_offset = (self._last_mouse_pos[0] - self._jet_x,
                                               self._last_mouse_pos[1] - self._baseline_y)
                elif self._state == self.State.CAMERA_ROI:
                    self._cam_corner_offset = (self._last_mouse_pos[0] - self._cam_x,
                                               self._last_mouse_pos[1] - self._baseline_y)
                self._state = self.State.NONE  # Baseline or ROI has been validated, go back to "None" state

        # Keyboard callback
        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE or self._state != self.State.NONE:
                return

            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                self._window.set_close_flag()
            elif key == UIKeyEvent.KEY_SPACE:
                self._update_cd_img = not self._update_cd_img  # Play/pause events
            elif key == UIKeyEvent.KEY_B:
                self._state = self.State.BASELINE
            elif key == UIKeyEvent.KEY_C:
                self._state = self.State.CAMERA_ROI
                self._is_initializing_roi = True
            elif key == UIKeyEvent.KEY_J:
                self._state = self.State.JET_ROI
                self._is_initializing_roi = True
            elif key == UIKeyEvent.KEY_ENTER:
                if self._jet_x < 0 and self._cam_x < 0:
                    print("No ROIs have been defined. Press C or J to define them.")
                else:
                    print("----------------")
                    if self._transpose_output_rois:
                        print("(ROIs below are expressed in the original, non-transposed, image frame)")
                    if self._jet_x > 0:
                        jet_roi = self._get_roi(self._jet_x, self._baseline_y,
                                                self._jet_corner_offset, self._transpose_output_rois)
                        print(" --detection-roi {:d} {:d} {:d} {:d}".format(*jet_roi))
                    if self._cam_x > 0:
                        cam_roi = self._get_roi(self._cam_x, self._baseline_y,
                                                self._cam_corner_offset, self._transpose_output_rois)
                        print(" --camera-roi {:d} {:d} {:d} {:d}".format(*cam_roi))
                    print("----------------")

        self._window.set_cursor_pos_callback(cursor_cb)
        self._window.set_mouse_callback(mouse_cb)
        self._window.set_keyboard_callback(keyboard_cb)

    def _draw_baseline(self, y, final_state=True):
        """
        Draws an horizontal line on the image, which is supposed so be aligned with the jet direction.

        Args:
            y (int): Baseline ordinate.
            final_state (bool): True if the y-position of the line isn't being modified.
        """
        cv2.line(self._front_img, (0, y), (self._width - 1, y),
                 self._color_baseline if final_state else self._color_tmp)

    def _draw_jet_rois(self, x, y, corner_offset, final_state=True):
        """
        Draws the rectangular Jet ROI and its two surrounding Background Activity ROIs.

        Args:
            x (int): X-position of either the left or the right side of the rectangular Jet ROI.
            y (int): Average Y-position of the rectangular Jet ROI.
            corner_offset ((int, int)): Translation vector that goes from (x,y) to the top opposite corner.
            final_state (bool): True if the Jet ROI isn't being modified.
        """
        # Draw ROI only if it's defined
        if self._jet_x < 0:
            return

        jet_roi = self._get_roi(x, y, corner_offset)
        width, height = jet_roi[2], jet_roi[3]
        left, top, right, bottom = jet_roi[0], jet_roi[1], jet_roi[0] + width, jet_roi[1] + height

        cv2.rectangle(self._front_img,  (left, top - height), (right, bottom - height),
                      self._color_bg_noise_roi if final_state else self._color_tmp, 1)
        cv2.rectangle(self._front_img,  (left, top + height), (right, bottom + height),
                      self._color_bg_noise_roi if final_state else self._color_tmp, 1)

        cv2.rectangle(self._front_img, (left, top), (right, bottom),
                      self._color_roi if final_state else self._color_tmp, 1)

    def _draw_camera_roi(self, x, y, corner_offset, final_state=True):
        """
        Draws the rectangular Camera ROI.

        Args:
            x (int): X-position of either the left or the right side of the rectangular Camera ROI.
            y (int): Average Y-position of the rectangular Camera ROI.
            corner_offset ((int, int)): Translation vector that goes from (x,y) to the top opposite corner.
            final_state (bool): True if the Camera ROI isn't being modified.
        """
        # Draw ROI only if it's defined
        if self._cam_x < 0:
            return

        cam_roi = self._get_roi(x, y, corner_offset)
        width, height = cam_roi[2], cam_roi[3]
        left, top, right, bottom = cam_roi[0], cam_roi[1], cam_roi[0] + width, cam_roi[1] + height

        cv2.rectangle(self._front_img, (left, top), (right, bottom),
                      self._color_roi if final_state else self._color_tmp, 1)

    def _print_help_msg(self, help_msg_list):
        """
        Prints a help message in the top left corner of the image.

        Args:
            help_msg (list of str): Tuple containing a string for each line of the help message.
        """
        y_text_pos = self._help_msg_text_pos[1]
        for s in help_msg_list:
            cv2.putText(self._front_img, s, (self._help_msg_text_pos[0], y_text_pos), self._font_face,
                        self._font_scale, self._color_txt, self._thickness, cv2.LINE_AA)
            y_text_pos += self._help_text_height + self._margin

    def _get_roi(self, x_ref, y_ref, corner_offset, transpose=False):
        """
        Gets a rectangular ROI vertically centered around y_ref.

        Args:
            x (int): X-position of either the left or the right side of the rectangular Camera ROI.
            y (int): Average Y-position of the rectangular Camera ROI.
            corner_offset ((int, int)): Translation vector that goes from (x,y) to the top opposite corner.
            transpose (bool): Transpose the output ROI if set to True.
        """
        # ROI (x, y, width, height) is centered around the baseline
        roi = [x_ref, y_ref - abs(corner_offset[1]), abs(corner_offset[0]), 2 * abs(corner_offset[1])]
        if corner_offset[0] < 0:
            roi[0] += corner_offset[0]

        # Transpose if needed
        if transpose:
            tmp_x = roi[0]
            roi[0] = roi[1]
            roi[1] = self.width_ - roi[2] - tmp_x

            roi[2], roi[3] = roi[3], roi[2]  # Swap width and height

        return tuple(roi)

    def swap_cd_frame_if_required(self, cd_frame):
        """
        Updates the background CD frame by swapping it if we are in "Play" mode, does nothing otherwise ("Pause" mode).

        Args:
            cd_frame (np array): CD frame to be swapped
        """
        if self._update_cd_img:
            self._last_cd_img, cd_frame = cd_frame, self._last_cd_img

    def update(self):
        """
        Generates and updates the display in the window.
        """
        self._front_img = self._last_cd_img.copy()

        if self._state == self.State.NONE:
            self._print_help_msg(["Press 'Space' to play/pause events", "Press 'B' to define the baseline",
                                  "Press 'C' to define the Camera ROI", "Press 'J' to define the Jet ROI",
                                  "Press 'Enter' to print ROIs", "Press 'Q' or 'Escape' to exit"])
            self._draw_baseline(self._baseline_y)
            self._draw_jet_rois(self._jet_x, self._baseline_y, self._jet_corner_offset)
            self._draw_camera_roi(self._cam_x, self._baseline_y, self._cam_corner_offset)
        elif self._state == self.State.BASELINE:
            self._print_help_msg(["Left click when aligned with the jet direction"])
            self._draw_baseline(self._last_mouse_pos[1], False)
            self._draw_jet_rois(self._jet_x, self._last_mouse_pos[1], self._jet_corner_offset, False)
            self._draw_camera_roi(self._cam_x, self._last_mouse_pos[1], self._cam_corner_offset, False)
        elif self._state == self.State.CAMERA_ROI:
            self._print_help_msg(["Click and drag to define the Camera ROI"])
            self._draw_baseline(self._baseline_y)
            self._draw_jet_rois(self._jet_x, self._baseline_y, self._jet_corner_offset)
            if self._is_initializing_roi:
                cv2.line(self._front_img, (self._last_mouse_pos[0], 0), (self._last_mouse_pos[0], self._height - 1),
                         self._color_tmp)
            else:
                self._draw_camera_roi(
                    self._cam_x, self._baseline_y,
                    (self._last_mouse_pos[0] - self._cam_x, self._last_mouse_pos[1] - self._baseline_y),
                    False)
        elif self._state == self.State.JET_ROI:
            self._print_help_msg(["Click and drag to define the Jet ROI and",
                                  "its two surrounding Background Activity ROIs"])
            self._draw_baseline(self._baseline_y)
            self._draw_camera_roi(self._cam_x, self._baseline_y, self._cam_corner_offset)
            if self._is_initializing_roi:
                cv2.line(self._front_img, (self._last_mouse_pos[0], 0), (self._last_mouse_pos[0], self._height - 1),
                         self._color_tmp)
            else:
                self._draw_jet_rois(self._jet_x, self._baseline_y, (self._last_mouse_pos[0] - self._jet_x,
                                                                    self._last_mouse_pos[1] - self._baseline_y), False)

        # Display
        self._window.show_async(self._front_img)

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
