# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Utility classes to generate and display an image from a frequency map and allow the user to interact with the mouse or
the keyboard to select pixels or regions of which he wants to know the dominant frequency.
"""

import cv2
from metavision_sdk_analytics import DominantValueMapAlgorithm, HeatMapFrameGeneratorAlgorithm
from metavision_sdk_ui import BaseWindow, MTWindow, UIAction, UIKeyEvent


class VibrationGUI:
    """
    Class that displays every received frequency map using a color map and prints the dominant frequency. In addition, this GUI
    stage allows:
       - defining ROIs in the frequency map and print their dominant frequency
       - checking the frequency at a specific pixel in the frequency map

    Args:
        width (int): Sensor's width.
        height (int): Sensor's height.
        min_freq (float): Minimum detected frequency (in Hz).
        max_freq (float): Maximum detected frequency (in Hz).
        freq_precision (float): Precision of frequency calculation - Width of frequency bins in histogram (in Hz).
        min_pixel_count (int): Minimum number of pixels to consider a frequency "real", i.e not coming from noise.
        outvideo (str): Path to output AVI file to save slow motion video.
    """

    def __init__(self, width, height, min_freq, max_freq, freq_precision, min_pixel_count, out_video):
        self._frame_generator = DominantFrequencyFrameGenerator(width, height, min_freq, max_freq,
                                                                freq_precision, min_pixel_count)
        self._window = MTWindow(title="Vibration estimation", width=width, height=self._frame_generator.full_height,
                                mode=BaseWindow.RenderMode.BGR, open_directly=True)
        self._window.show_async(self._frame_generator.output_img)

        self._crt_roi_start_pos = (-1, -1)

        # Video Writer
        self.out_video = out_video
        if self.out_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_name = out_video + ".avi"
            self.video_writer = cv2.VideoWriter(self.video_name, fourcc, 20,
                                                (width, self._frame_generator.full_height))

        # Help message
        self._display_long_help = False
        self._long_help_msg = ["Keyboard/mouse actions:",
                               "  \"h\" - show/hide the help menu",
                               "  \"q\" - quit the application",
                               "  \"c\" - clear all the ROIs",
                               "  Click and drag to create ROIs"]

        self._short_help_msg = ["Press 'h' for help"]

        # Event callbacks
        self._last_mouse_pos = (-1, -1)
        self._is_updating_roi = False
        self._rois = []

        def cursor_cb(x, y):
            window_width, window_height = self._window.get_size()
            # The window may have been resized meanwhile. So we map the coordinates to the original window's size.
            mapped_x = x * self._frame_generator.full_width / window_width
            mapped_y = y * self._frame_generator.full_height / window_height
            mouse_pos = (int(mapped_x), int(mapped_y))

            # The frequency map is smaller than the displayed image (a color map bar is added). So we need to check
            # that we won't be out of bounds.
            if (mouse_pos[0] < 0) or (mouse_pos[0] >= width):
                return
            if (mouse_pos[1] < 0) or (mouse_pos[1] >= height):
                return

            # Update the last created ROI
            if (self._is_updating_roi and self._rois):
                xmin = min(self._crt_roi_start_pos[0], mouse_pos[0])
                xmax = max(self._crt_roi_start_pos[0], mouse_pos[0])
                ymin = min(self._crt_roi_start_pos[1], mouse_pos[1])
                ymax = max(self._crt_roi_start_pos[1], mouse_pos[1])
                self._rois[-1] = ((xmin, ymin), (xmax, ymax))

            self._last_mouse_pos = mouse_pos

        def mouse_cb(button, action, mods):
            if action == UIAction.PRESS:
                # Start a new ROI
                self._is_updating_roi = True
                self._crt_roi_start_pos = self._last_mouse_pos
                self._rois.append((self._last_mouse_pos, (self._last_mouse_pos[0]+1,  self._last_mouse_pos[1]+1)))
            elif action == UIAction.RELEASE:
                # Stop updating the ROI
                self._is_updating_roi = False

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                self._window.set_close_flag()
            elif key == UIKeyEvent.KEY_C:
                self._rois = []
            elif key == UIKeyEvent.KEY_H:
                self._display_long_help = not self._display_long_help

        self._window.set_cursor_pos_callback(cursor_cb)
        self._window.set_mouse_callback(mouse_cb)
        self._window.set_keyboard_callback(keyboard_cb)

    def show(self, freq_map):
        """
        Processes the frequency map and displays the resulting image in the window.

        Args:
            freq_map (np.array): Floating point 2D frequency map
        """
        self._frame_generator.generate_bgr_image(freq_map)
        self._frame_generator.print_dominant_frequency(freq_map)
        self._frame_generator.print_cursor_frequency(freq_map, self._last_mouse_pos)
        self._frame_generator.print_rois_frequencies(freq_map, self._rois)
        self._frame_generator.print_help_message(
            self._long_help_msg if self._display_long_help else self._short_help_msg)

        self._window.show_async(self._frame_generator.output_img)
        if self.out_video:
            self.video_writer.write(self._frame_generator.output_img)

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


class DominantFrequencyFrameGenerator:
    """
    Class that converts a frequency map into an image and prints dominant frequencies on top of it:
        - dominant frequency of the whole field of view
        - dominant frequencies of regions of interest in the frequency map
        - dominant frequency pointed by the mouse cursor

    Attributes:
        full_height (int): Image's height (Sensor's height + color bar's height).
        full_width (int): Image's width (Sensor's width).
        output_img (np.array): CV_8UC3 color image representing the frequency map and the dominant frequencies.

    Args:
        width (int): Sensor's width.
        height (int): Sensor's height.
        min_freq (float): Minimum detected frequency (in Hz).
        max_freq (float): Maximum detected frequency (in Hz).
        freq_precision (float): Precision of frequency calculation - Width of frequency bins in histogram (in Hz).
        min_pixel_count (int): Minimum number of pixels to consider a frequency "real", i.e not coming from noise.
    """

    def __init__(self, width, height, min_freq, max_freq, freq_precision, min_pixel_count):
        # Constants
        self._font_face = cv2.FONT_HERSHEY_PLAIN  # Font used for text rendering
        self._font_scale = 1.0  # Font scale used for text rendering
        self._thickness = 1  # Line thickness used for text rendering
        self._margin = 5  # Additional space used for text rendering

        self._width = width
        self._height = height

        # Heatmap frame generator
        self._heat_map_generator = HeatMapFrameGeneratorAlgorithm(
            min_freq, max_freq,  freq_precision, width, height, "Hz")
        self.output_img = self._heat_map_generator.get_output_image()
        self.full_height = self._heat_map_generator.full_height
        self.full_width = self._heat_map_generator.full_width

        # Dominant value map algorithm
        self._dominant_value_map_algo_ = DominantValueMapAlgorithm(
            min_freq, max_freq, freq_precision, min_pixel_count)

        # Initialize some parameters used when rendering texts
        (_, text_height), baseline = cv2.getTextSize("Frequency: XXXX Hz", self._font_face, self._font_scale, self._thickness)
        self._dominant_freq_text_pos = (self._margin, height - baseline - self._margin)
        self._help_msg_text_pos_ = (self._margin, text_height + self._margin)
        self._text_full_height = text_height + baseline

        # Estimate the number of decimal digits to display in a string using freq_precision
        self._value_string_precision = 0
        shifted_float_val = freq_precision
        while abs(shifted_float_val - int(shifted_float_val+0.01)) > 0.00001:
            self._value_string_precision += 1
            shifted_float_val *= 10

        # Print a message in the window at the beginning of the application
        # when no frequency map has been received yet
        self.output_img[:] = 0
        init_message_1 = "NO VIBRATING OBJECT IN THE"
        init_message_2 = "FIELD OF VIEW OF THE CAMERA"

        init_font = cv2.FONT_HERSHEY_SIMPLEX
        init_font_scale = 1
        init_thickness = 2
        y_mid = int(0.5 * self.output_img.shape[0])
        (text_width1, _), baseline1 = cv2.getTextSize(init_message_1, init_font, init_font_scale, init_thickness)
        (text_width2, text_height2), _ = cv2.getTextSize(init_message_2, init_font, init_font_scale, init_thickness)
        x1 = int(0.5 * (self.output_img.shape[1] - text_width1))
        x2 = int(0.5 * (self.output_img.shape[1] - text_width2))

        cv2.putText(self.output_img, init_message_1, (x1, y_mid - baseline1),
                    init_font, init_font_scale, (255, 255, 255), init_thickness)
        cv2.putText(self.output_img, init_message_2, (x2, y_mid + text_height2),
                    init_font, init_font_scale, (255, 255, 255), init_thickness)

    def _freq_to_string(self, freq):
        """
        Converts floating point frequency to string with a given number of decimal digits.

        Args:
            freq (float): Floating point frequency
        """
        return "{:.{prec}f}".format(freq, prec=self._value_string_precision)

    def print_dominant_frequency(self, freq_map):
        """
        Computes and prints the dominant frequency measured in the camera's FOV.

        Args:
            freq_map (np.array): Floating point 2D frequency map
        """
        # Compute dominant frequency
        success, dominant_frequency = self._dominant_value_map_algo_.compute_dominant_value(freq_map)

        msg = "Frequency:"
        if success:
            msg += " " + self._freq_to_string(dominant_frequency) + " Hz"
        else:
            msg += "     N/A"

        cv2.putText(self.output_img, msg, self._dominant_freq_text_pos,
                    self._font_face, self._font_scale, (255, 255, 255), self._thickness)

    def print_cursor_frequency(self, freq_map, mouse_position):
        """
        Prints the frequency pointed by the mouse cursor.

        Args:
            freq_map (np.array): Floating point 2D frequency map
            mouse_position (tuple(int)): Most recent 2D coordinates of the mouse cursor.
        """
        if (mouse_position[0] < 0) or (mouse_position[0] >= self._width):
            return
        if (mouse_position[1] < 0) or (mouse_position[1] >= self._height):
            return

        freq = int(freq_map[mouse_position[1]][mouse_position[0]])
        if freq > 0:
            cv2.putText(self.output_img, self._freq_to_string(freq) + " Hz", mouse_position,
                        self._font_face, self._font_scale, (255, 255, 255), self._thickness)

    def print_rois_frequencies(self, freq_map, rois):
        """
        Prints the ROIs and their associated dominant frequency.

        Args:
            freq_map (np.array): Floating point 2D frequency map
            rois ([tuple(tuple(int))]): Regions of interest for which the dominant frequency is to be estimated.
                                        Each ROI is defined with its top-left and bottom-right positions.
        """
        for roi in rois:
            top_left = roi[0]
            bot_right = roi[1]
            cv2.rectangle(self.output_img, top_left, bot_right, (0, 255, 255))

            roi_freq_map = freq_map[top_left[1]:bot_right[1]+1, top_left[0]:bot_right[0]+1]
            success, dominant_frequency = self._dominant_value_map_algo_.compute_dominant_value(roi_freq_map)

            msg = self._freq_to_string(dominant_frequency) + " Hz" if success else "N/A"
            cv2.putText(self.output_img, msg, (bot_right[0] + self._margin, bot_right[1]),
                        self._font_face, self._font_scale, (255, 255, 255), self._thickness)

    def print_help_message(self, help_msg_list):
        """
        Prints a help message indicating which interactions are possible.

        Args:
            help_msg_list ([string]): Help message, in the form of a list of lines to be printed
        """
        pos_x, pos_y = self._help_msg_text_pos_
        for line in help_msg_list:
            cv2.putText(self.output_img, line, (pos_x, pos_y),
                        self._font_face, self._font_scale, (255, 255, 255), self._thickness)
            pos_y += self._text_full_height

    def generate_bgr_image(self, freq_map):
        """
        Generates and displays a BGR image corresponding to a frequency map.
            - converts the frequency map to an BGR image,
            - prints the dominant frequency in the whole FOV,
            - prints all the created ROIs and their associated dominant frequencies,
            - prints the dominant frequency pointed by the mouse cursor,
            - prints a help message

        Args:
            freq_map (np.array): Floating point 2D frequency map
        """
        # Generate the heat map
        self._heat_map_generator.generate_bgr_heat_map(
            freq_map, self.output_img)
