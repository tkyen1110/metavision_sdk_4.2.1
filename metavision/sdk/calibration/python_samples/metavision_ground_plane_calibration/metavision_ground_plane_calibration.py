# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Script to compute ground plane calibration
"""

import fire
import os
import math
import json
import numpy as np
import cv2
from scipy.optimize import fsolve

from metavision_sdk_core import BaseFrameGenerationAlgorithm, RoiFilterAlgorithm
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.dat_tools import DatWriter
import metavision_sdk_cv

nb_points_to_accumulate = 25
delta_t = 10000

refPt = []
left_button_down = False
events_frame = None


def mouse_handler(event, x, y, flags, param):
    """
    This is used to draw a ROI rectangle in the frame
    """
    global refPt, left_button_down, events_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        left_button_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        if x != refPt[0][0] and y != refPt[0][1]:
            refPt.append((x, y))
        else:
            refPt = []
        left_button_down = False
    elif left_button_down:
        cv2.rectangle(events_frame, refPt[0], (x, y), (255, 255, 0), 3)
        cv2.imshow('events', events_frame[..., ::-1])


class PairOfBlinkingLedExtractor:
    """
    This class is used to detect a pair of blinking LEDs.

    The blinking frequencies (which must be different) are specified.
    Each processed chunk of events must contain exactly only one blob for each frequency, otherwise no clusters is returned
    """

    def __init__(self, height, width, min_freq=120., max_freq=250., min_cluster_size=5, max_time_diff=delta_t,
                 expected_low_freq=150., expected_high_freq=200.):
        self.height = height
        self.width = width

        self.frequency_filter = metavision_sdk_cv.FrequencyAlgorithm(
            width=width, height=height, min_freq=min_freq, max_freq=max_freq)
        self.frequency_clustering_filter = metavision_sdk_cv.FrequencyClusteringAlgorithm(
            width=width, height=height, min_cluster_size=min_cluster_size, max_time_diff=max_time_diff)
        self.freq_buffer = self.frequency_filter.get_empty_output_buffer()
        self.cluster_buffer = self.frequency_clustering_filter.get_empty_output_buffer()

        self.expected_low_freq = expected_low_freq
        self.expected_high_freq = expected_high_freq

    def compute_clusters(self, events):
        """
        Explicitly detects two clusters with specified frequencies
        """
        self.frequency_filter.process_events(events, self.freq_buffer)
        self.frequency_clustering_filter.process_events(self.freq_buffer, self.cluster_buffer)
        if self.cluster_buffer.numpy().size == 2:
            cluster_buffer_np = self.cluster_buffer.numpy().copy()
            if cluster_buffer_np[0]["frequency"] <= cluster_buffer_np[1]["frequency"]:
                cluster_low_freq = cluster_buffer_np[0]
                cluster_high_freq = cluster_buffer_np[1]
            else:
                cluster_low_freq = cluster_buffer_np[1]
                cluster_high_freq = cluster_buffer_np[0]
            if abs(cluster_low_freq["frequency"] - self.expected_low_freq) > 0.1 * self.expected_low_freq:
                print("WARNING: wrong LED frequency detected: {}  (expected {})".format(
                    cluster_low_freq["frequency"], self.expected_low_freq))
                return []
            if abs(cluster_high_freq["frequency"] - self.expected_high_freq) > 0.1 * self.expected_high_freq:
                print("WARNING: wrong LED frequency detected: {}  (expected {})".format(
                    cluster_high_freq["frequency"], self.expected_high_freq))
                return []

            return [cluster_low_freq, cluster_high_freq]
        else:
            return []


def compute_led_positions_in_world(dist_left_lowfreq, dist_left_highfreq, dist_right_lowfreq, dist_right_highfreq,
                                   ref_left_x, ref_left_z, ref_right_x, ref_right_z,
                                   expected_length, tolerance=0.005, verbose=True):
    """
    Given the set of distances, computes the positions of the markers (LEDs) in the world

    Args:
        dist_left_lowfreq (float): distance (in meters) between the low frequency LED and the left reference point
        dist_left_highfreq (float): distance (in meters) between the high frequency LED and the left reference point
        dist_right_lowfreq (float): distance (in meters) between the low frequency LED and the right reference point
        dist_right_highfreq (float): distance (in meters) between the high frequency LED and the right reference point
        ref_left_x (float): x coordinate of the left reference point
        ref_left_z (float): z coordinate of the left reference point
        ref_right_x (float): x coordinate of the right reference point
        ref_right_z (float): z coordinate of the right reference point
        expected_length (float): distance between the low frequency and high frequency LEDs
        tolerance (float): maximum discrepancy between the expected and the computed distances between LEDs
        verbose (boolean): if true, displays more information on screen
    """
    def triangulate_two_points(p):
        xa = p[0]
        za = p[1]
        xb = p[2]
        zb = p[3]
        F = np.empty(4)
        F[0] = (xa - ref_left_x)**2 + (za - ref_left_z)**2 - dist_left_lowfreq**2
        F[1] = (xa - ref_right_x)**2 + (za - ref_right_z)**2 - dist_right_lowfreq**2
        F[2] = (xb - ref_left_x)**2 + (zb - ref_left_z)**2 - dist_left_highfreq**2
        F[3] = (xb - ref_right_x)**2 + (zb - ref_right_z)**2 - dist_right_highfreq**2
        return F
    zGuess = np.array([0., 5., 0., 5.])

    xa, za, xb, zb = fsolve(triangulate_two_points, zGuess)
    d = math.sqrt((xa - xb)**2 + (za - zb)**2)
    if verbose:
        print("computed first cone position : ({}, {})".format(xa, za))
        print("computed second cone position: ({}, {})\n".format(xb, zb))
        print("d: ", d)

    res_str = "{} vs {}    diff: {} (tol: {})".format(d, expected_length, abs(d - expected_length), tolerance)
    if abs(d - expected_length) > tolerance:
        print("Error! Measures are inconsistent: {}".format(res_str))
        print("Measures are not added")
        return None
    else:
        if verbose:
            print("OK: all measures are consistent: {}".format(res_str))
        return (xa, za), (xb, zb)


def calib_interactive(output_directory, intrinsics_directory, input_recording="",
                      ref_left_x=-0.5, ref_left_z=0., ref_right_x=0.5, ref_right_z=0.,
                      y_offset_led=-0.42,
                      led_pix_position_stability_tolerance=1.,
                      leds_relative_position_tolerance=0.005):
    """
    Calibrate using a recording or live camera using interactive window

        Top view:
            - origin (0, 0, 0) is on the ground, in the middle of the front bumper
            - X axis is horizontal, pointing to the right
            - Y axis is vertical, pointing downwards
            - Z axis is horizontal, pointing in forward direction

        On this drawing, z_left and z_right both equal 0


                               z
                                ^
                                |
                                |
                                |
                                |
                                |
                                |
                                |
        ----------o-------------*--------------o----------->  x
                left         (0,0,0)         right
              reference                    reference
                point                        point
           (x_left,0,z_left)           (x_right,0,z_right)



    Args:
        output_directory (str): output directory. If it does not exist, it is created by this application
        intrinsics_directory (str): directory which contains cam.txt and dist.txt (or intrinsics.json)
                                    camera calibration (in OpenCV format)
        input_recording (str): sequence to process. If None, use the first available camera
        ref_left_x (float): x coordinate of the left reference point
        ref_left_z (float): z coordinate of the left reference point
        ref_right_x (float): x coordinate of the right reference point
        ref_right_z (float): z coordinate of the right reference point
        y_offset_led (float): y coordinate of all leds
        led_pix_position_stability_tolerance (float): LEDs pixel positions are average over several consecutive timesteps. Discrepancy between min/max and average value should not be larger than this threshold (camera is static and LEDs are also static)
        leds_relative_position_tolerance (float): the distance between the pair of LEDs is either fixed or measured. This threshold ensures that this computed distance between the two LEDs matches the real world distance.
    """
    global events_frame, refPt
    print("*************************************************************************")
    print("*  Click and drag mouse to select an ROI                                *")
    print("*  Press R to cancel ROI                                                *")
    print("*  Press A to add a measurement (when red and green dots are detected)  *")
    print("*  Press C to compute the calibration                                   *")
    print("*  Press Q to quit the program                                          *")
    print("*************************************************************************")
    print("\n")
    assert not os.path.exists(output_directory)
    os.makedirs(output_directory)
    mv_it = EventsIterator(input_path=input_recording, delta_t=delta_t)
    ev_height, ev_width = mv_it.get_size()
    events_frame = np.zeros((ev_height, ev_width, 3), dtype=np.uint8)
    window_events = cv2.namedWindow("events")
    cv2.setMouseCallback('events', mouse_handler, 0)

    filename_output_events = os.path.join(output_directory, "ground_plane_calibration_sequence.dat")

    if input_recording == "":
        dat_writer = DatWriter(filename=filename_output_events, height=ev_height, width=ev_width)

    blinking_leds_extractor = PairOfBlinkingLedExtractor(height=ev_height, width=ev_width)

    roi_filter_algorithm = RoiFilterAlgorithm(x0=0, y0=0, x1=ev_width - 1, y1=ev_height - 1)
    roi_events_buffer = roi_filter_algorithm.get_empty_output_buffer()
    last_clusters = []

    filename_measurements = os.path.join(output_directory, "measurements.json")

    all_measures = []
    for ev in mv_it:
        if ev.size == 0:
            continue

        if input_recording == "":
            dat_writer.write(ev)
        current_ts = ev[0]["t"]

        if len(refPt) == 2:
            cv2.rectangle(events_frame, refPt[0], refPt[1], (255, 255, 0), 3)
            roi_filter_algorithm.x0 = min(refPt[0][0], refPt[1][0])
            roi_filter_algorithm.y0 = min(refPt[0][1], refPt[1][1])
            roi_filter_algorithm.x1 = max(refPt[0][0], refPt[1][0])
            roi_filter_algorithm.y1 = max(refPt[0][1], refPt[1][1])

        roi_filter_algorithm.process_events(ev, roi_events_buffer)
        ev = roi_events_buffer.numpy()
        BaseFrameGenerationAlgorithm.generate_frame(ev, events_frame)

        clusters = blinking_leds_extractor.compute_clusters(ev)

        for cluster in clusters:
            x0 = int(cluster["x"]) - 10
            y0 = int(cluster["y"]) - 10
            cv2.rectangle(events_frame, (x0, y0), (x0 + 20, y0 + 20), color=(0, 0, 255))
            cv2.putText(events_frame, "{} Hz".format(int(cluster["frequency"])), (x0, y0 - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 255), 1)
        if len(clusters) == 2:
            cv2.circle(events_frame, (int(clusters[0]["x"]), int(clusters[0]["y"])), 3, (0, 255, 0), 3)
            cv2.circle(events_frame, (int(clusters[1]["x"]), int(clusters[1]["y"])), 3, (255, 0, 0), 3)
            last_clusters.append(clusters)
            if len(last_clusters) > nb_points_to_accumulate:
                last_clusters.pop(0)
        else:
            last_clusters = []

        cv2.imshow('events', events_frame[..., ::-1])
        key = cv2.waitKey(1)
        if key == ord('q'):
            # quit the program
            break
        elif key == ord('r'):
            # reset ROI
            refPt = []
            roi_filter_algorithm.x0, roi_filter_algorithm.y0 = 0, 0
            roi_filter_algorithm.x1, roi_filter_algorithm.y1 = ev_width - 1, ev_height - 1
        elif key == ord('a'):
            # add current points
            if len(last_clusters) != nb_points_to_accumulate:
                print("Warning: points detection is too unstable ({}), please try again".format(len(last_clusters)))
            else:
                clusters_low_freq, clusters_high_freq = [], []
                for cluster_low_freq, cluster_high_freq in last_clusters:
                    clusters_low_freq.append(cluster_low_freq)
                    clusters_high_freq.append(cluster_high_freq)
                clusters_low_freq_np = np.array(clusters_low_freq)
                clusters_high_freq_np = np.array(clusters_high_freq)

                if not check_cluster_measurements_stability(
                        clusters_low_freq_np, led_pix_position_stability_tolerance):
                    print("Warning: low freq cluster is too unstable")
                    continue
                if not check_cluster_measurements_stability(
                        clusters_high_freq_np, led_pix_position_stability_tolerance):
                    print("Warning: high freq cluster is too unstable")
                    continue
                print("Adding new points")
                low_freq_pix_x = np.average(clusters_low_freq_np["x"])
                low_freq_pix_y = np.average(clusters_low_freq_np["y"])
                high_freq_pix_x = np.average(clusters_high_freq_np["x"])
                high_freq_pix_y = np.average(clusters_high_freq_np["y"])
                print("low_freq pix: ({}, {})".format(low_freq_pix_x, low_freq_pix_y))
                print("high_freq pix: ({}, {})".format(high_freq_pix_x, high_freq_pix_y))

                if not check_cluster_is_different_than_existing(
                        low_freq_pix_x, low_freq_pix_y, high_freq_pix_x, high_freq_pix_y, all_measures):
                    print("Current measure is too close from existing one (data already acquired, please move the LEDs)")
                    continue

                dist_left_lowfreq = float(input("Enter distance between left reference point and low freq led: "))
                dist_left_highfreq = float(input("Enter distance between left reference point and high freq led: "))
                dist_right_lowfreq = float(input("Enter distance between right reference point and low freq led: "))
                dist_right_highfreq = float(input("Enter distance between right reference point and high freq led: "))
                dist_between_leds = input("Enter distance between high and low leds (default is 1.0): ")
                if dist_between_leds == '':
                    dist_between_leds = '1'
                dist_between_leds = float(dist_between_leds)

                print(
                    dist_left_lowfreq,
                    dist_left_highfreq,
                    dist_right_lowfreq,
                    dist_right_highfreq,
                    dist_between_leds)
                led_positions = compute_led_positions_in_world(
                    dist_left_lowfreq=dist_left_lowfreq,
                    dist_left_highfreq=dist_left_highfreq,
                    dist_right_lowfreq=dist_right_lowfreq,
                    dist_right_highfreq=dist_right_highfreq,
                    ref_left_x=ref_left_x,
                    ref_left_z=ref_left_z,
                    ref_right_x=ref_right_x,
                    ref_right_z=ref_right_z,
                    expected_length=dist_between_leds,
                    tolerance=leds_relative_position_tolerance)
                if led_positions is None:
                    continue
                (x_lowfreq, z_lowfreq), (x_highfreq, z_highfreq) = led_positions
                print("led_positions: ", led_positions)

                if input_recording == "":
                    filename_events = filename_output_events
                else:
                    filename_events = input_recording

                current_measurement = {
                    "recording": filename_events,
                    "start_ts_us": str(current_ts - (nb_points_to_accumulate - 1) * delta_t),
                    "end_ts_us": str(current_ts + delta_t),
                    "ref_left_x": str(ref_left_x),
                    "ref_left_z": str(ref_left_z),
                    "ref_right_x": str(ref_right_x),
                    "ref_right_z": str(ref_right_z),
                    "y_offset_led": str(y_offset_led),
                    "low_freq_pix_x": str(low_freq_pix_x),
                    "low_freq_pix_y": str(low_freq_pix_y),
                    "high_freq_pix_x": str(high_freq_pix_x),
                    "high_freq_pix_y": str(high_freq_pix_y),
                    "dist_left_lowfreq": str(dist_left_lowfreq),
                    "dist_left_highfreq": str(dist_left_highfreq),
                    "dist_right_lowfreq": str(dist_right_lowfreq),
                    "dist_right_highfreq": str(dist_right_highfreq),
                    "expected_length": str(dist_between_leds),
                    "roi_x0": str(roi_filter_algorithm.x0),
                    "roi_y0": str(roi_filter_algorithm.y0),
                    "roi_x1": str(roi_filter_algorithm.x1),
                    "roi_y1": str(roi_filter_algorithm.y1),
                    "leds_relative_position_tolerance": str(leds_relative_position_tolerance)
                }
                all_measures.append(current_measurement)
        elif key == ord('c') or key == ord(' '):
            # compute calibration
            if len(all_measures) < 3:
                print(
                    "Not enough measures collected. Please collect at least 3 pairs of points (current number of pairs: {}".format(
                        len(all_measures)))
                continue
            print("Computing calibration")
            fic = open(filename_measurements, "w")
            json.dump(all_measures, fic, indent=2)
            fic.close()
            calib_json(filename_json=filename_measurements,
                       intrinsics_directory=intrinsics_directory,
                       output_calib_filename=os.path.join(output_directory, "Tr_world_to_cam.txt"))
            print("OK")
            break

    cv2.destroyAllWindows()
    if input_recording == "":
        dat_writer.close()
    if not os.path.exists(filename_measurements) and len(all_measures) > 0:
        filename_measurements_wip = os.path.join(output_directory, "measurements_wip.json")
        print("Calibration NOT complete. Current measurements have been saved in file: {}".format(filename_measurements_wip))
        fic = open(filename_measurements_wip, "w")
        json.dump(all_measures, fic, indent=2)
        fic.close()


def check_cluster_measurements_stability(clusters, led_pix_position_stability_tolerance):
    """
    Checks that the measurements are stable in time (camera and LEDs are static)

    Args:
        clusters (np.array): last nb_points_to_accumulate (default 25) positions corresponding to the same blinding LED
        led_pix_position_stability_tolerance (float): maximum tolerance value between average and min or max
                                                      (no outlier is allowed)
    """
    avg_x = np.average(clusters["x"])
    avg_y = np.average(clusters["y"])
    avg_freq = np.average(clusters["frequency"])
    if avg_x - np.min(clusters["x"]) > led_pix_position_stability_tolerance or np.max(clusters["x"] -
                                                                                      avg_x) > led_pix_position_stability_tolerance:
        print("fail: ", avg_x - np.min(clusters["x"]), np.max(clusters["x"] - avg_x))
        return False
    if avg_y - np.min(clusters["y"]) > led_pix_position_stability_tolerance or np.max(clusters["y"] -
                                                                                      avg_y) > led_pix_position_stability_tolerance:
        print("fail: ", avg_y - np.min(clusters["y"]), np.max(clusters["y"] - avg_y))
        return False
    if avg_freq - np.min(clusters["frequency"]) > 5. or np.max(clusters["frequency"] - avg_freq) > 5.:
        return False
    return True


def check_cluster_is_different_than_existing(low_freq_pix_x, low_freq_pix_y,
                                             high_freq_pix_x, high_freq_pix_y,
                                             all_measurements):
    """
    Checks that the current pose is sufficiently different than the previous ones

    Args:
        low_freq_pix_x (float): x coordinate of the pixel corresponding to the low frequency LED
        low_freq_pix_y (float): y coordinate of the pixel corresponding to the low frequency LED
        high_freq_pix_x (float): x coordinate of the pixel corresponding to the high frequency LED
        high_freq_pix_y (float): y coordinate of the pixel corresponding to the high frequency LED
        all_measurements (list): list of previously recorded measurements
    """
    for measurement in all_measurements:
        if (abs(low_freq_pix_x - float(measurement["low_freq_pix_x"])) <
                10) and (abs(low_freq_pix_y - float(measurement["low_freq_pix_y"])) < 10):
            return False
        if (abs(high_freq_pix_x - float(measurement["high_freq_pix_x"])) <
                10) and (abs(high_freq_pix_y - float(measurement["high_freq_pix_y"])) < 10):
            return False
    return True


def calib_json(filename_json, intrinsics_directory, output_calib_filename):
    """
    Calibrate using a json file of measures (no user interaction)

    Args:
        filename_json (str): Filename which contains information necessary to compute the calibration
        intrinsics_directory (str): directory which contains the cam.txt and dist.txt (or intrinsics.json)
                                    calibration (OpenCV format)
        output_calib_filename (str): filename for output 4x4 matrix Tr_world_to_cam
    """
    assert os.path.isfile(filename_json)
    assert not os.path.exists(output_calib_filename)
    assert os.path.isdir(intrinsics_directory)

    cam, dist = load_intrinsics(intrinsics_directory)

    measurements = json.load(open(filename_json, "r"))
    assert len(measurements) >= 3
    objectPoints = np.zeros((2 * len(measurements), 3), dtype=np.float64)
    imagePoints = np.zeros((2 * len(measurements), 2), dtype=np.float64)
    for i, measurement in enumerate(measurements):
        dist_left_lowfreq = float(measurement["dist_left_lowfreq"])
        dist_left_highfreq = float(measurement["dist_left_highfreq"])
        dist_right_lowfreq = float(measurement["dist_right_lowfreq"])
        dist_right_highfreq = float(measurement["dist_right_highfreq"])
        ref_left_x = float(measurement["ref_left_x"])
        ref_left_z = float(measurement["ref_left_z"])
        ref_right_x = float(measurement["ref_right_x"])
        ref_right_z = float(measurement["ref_right_z"])
        expected_length = float(measurement["expected_length"])
        leds_relative_position_tolerance = float(measurement["leds_relative_position_tolerance"])
        y_offset_led = float(measurement["y_offset_led"])
        led_positions = compute_led_positions_in_world(
            dist_left_lowfreq=dist_left_lowfreq,
            dist_left_highfreq=dist_left_highfreq,
            dist_right_lowfreq=dist_right_lowfreq,
            dist_right_highfreq=dist_right_highfreq,
            ref_left_x=ref_left_x,
            ref_left_z=ref_left_z,
            ref_right_x=ref_right_x,
            ref_right_z=ref_right_z,
            expected_length=expected_length,
            tolerance=leds_relative_position_tolerance,
            verbose=False)
        if led_positions is None:
            print("Error: could not perform the calibration given the set of measurements in the file: ", filename_json)
            return

        (x_lowfreq, z_lowfreq), (x_highfreq, z_highfreq) = led_positions

        objectPoints[2 * i][0] = x_lowfreq
        objectPoints[2 * i][1] = y_offset_led
        objectPoints[2 * i][2] = z_lowfreq
        objectPoints[2 * i + 1][0] = x_highfreq
        objectPoints[2 * i + 1][1] = y_offset_led
        objectPoints[2 * i + 1][2] = z_highfreq
        imagePoints[2 * i][0] = float(measurement["low_freq_pix_x"])
        imagePoints[2 * i][1] = float(measurement["low_freq_pix_y"])
        imagePoints[2 * i + 1][0] = float(measurement["high_freq_pix_x"])
        imagePoints[2 * i + 1][1] = float(measurement["high_freq_pix_y"])

    cam_identity = np.eye(3).astype(cam.dtype)
    dist_zero = np.zeros_like(dist)
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)

    if dist.size == 5:
        use_fisheye = False
    elif dist.size == 4:
        use_fisheye = True
    else:
        raise RuntimeError("Unknown distortion matrix (size must be either 4 or 5")

    if use_fisheye:
        imagePoints_undistorted = cv2.fisheye.undistortPoints(np.expand_dims(imagePoints, axis=0), cam, dist).squeeze()
    else:
        imagePoints_undistorted = cv2.undistortPoints(imagePoints, cam, dist)

    ret, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints_undistorted, cam_identity, dist_zero,
                                   rvec, tvec, flags=cv2.SOLVEPNP_IPPE)

    if not ret:
        print("Solving pose has failed. Please collect more points and measurements and try again")
        return
    Tr_world_to_cam = np.eye(4).astype(np.float64)
    Tr_world_to_cam[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    Tr_world_to_cam[0:3, 3] = tvec
    print(Tr_world_to_cam)
    dn = os.path.dirname(output_calib_filename)
    if dn != "" and not os.path.isdir(dn):
        os.makedirs(dn)
    np.savetxt(output_calib_filename, Tr_world_to_cam)

    Tr_cam_to_world = np.linalg.inv(Tr_world_to_cam)
    print("")
    print("Position of the camera with regard to world coordinate frame:")
    print("\tX: lateral (positive means to the right)    : ", Tr_cam_to_world[0, 3])
    print("\tY: height (negative means above the ground) : ", Tr_cam_to_world[1, 3])
    print("\tZ: longitudinal (negative means behind)     : ", Tr_cam_to_world[2, 3])


def init_target_non_rectangular_grid(pattern_height=2, pattern_width=3, radius=5., B=2.,
                                     grid_offset_x=0., grid_offset_y=-0.42, grid_offset_z=0.):
    """
    This function is a helper to get the 3d coordinates of the LEDs pattern in a non-rectangular grid.
    LEDs are placed in concentric arcs of circles around the origin.

    Args:
        pattern_height (int): number of concentric circles. Radii are multiple of radius
        pattern_width (int): number of LEDs on a given arc of a circle
        radius (float): Radii are multiples of this value
        B (float): distance between LEDs on the closest arc of a circle
        grid_offset_x (float): global offset to apply to the grid along the x axis
        grid_offset_y (float): global offset to apply to the grid along the y axis (default value
                               is 0.42 since this is the height of cones on top of which LEDs are mounted)
        grid_offset_z (float): global offset to apply to the grid along the z axis
    """
    assert pattern_width == 3
    g = np.zeros((pattern_width * pattern_height, 3), np.float32)
    theta = 2 * math.asin(B / (2 * radius))
    for r in range(pattern_height):
        D = (pattern_height - r) * radius
        g[r * pattern_width] = (-D * math.cos((math.pi / 2.) - theta), 0., D * math.sin((math.pi / 2.) - theta))
        g[r * pattern_width + 1] = (0, 0., D)
        g[r * pattern_width + 2] = (D * math.cos((math.pi / 2.) - theta), 0., D * math.sin((math.pi / 2.) - theta))
    g += (grid_offset_x, grid_offset_y, grid_offset_z)
    shape_of_target = g.flatten().tolist()
    print("Pattern 3D points (non-rectangular grid): \n", g)
    return shape_of_target


def get_world_to_cam_given_measures(angle_pointing_downwards_degrees, height):
    """
    This function is a helper to get the 4x4 world to cam matrix, given a constrained use-case
    when the camera is at a known position vertically above the origin, with a known pitch angle pointing downwards.
    In this case, there is no need to perform the ground plane calibration using the cones. The 4x4 matrix can be
    deduced from the known height and angle.

    Args:
        angle_pointing_downwards_degrees (float): pitch angle in degrees (downwards is positive)
        height (float): height of the camera above the ground in meters (positive value)
    """
    T = np.eye(4)
    assert angle_pointing_downwards_degrees >= 0 and angle_pointing_downwards_degrees < 90
    assert height >= 0, "Height above the ground should be a positive value"
    angle_radians = math.radians(angle_pointing_downwards_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    T[1, 1] = cos_angle
    T[1, 2] = -sin_angle
    T[2, 1] = sin_angle
    T[2, 2] = cos_angle
    T[1, 3] = height * cos_angle
    T[2, 3] = height * sin_angle
    return T


def load_intrinsics(intrinsics_directory):
    """
    Loads 3x3 camera matrix (K) and the distortion coefficients (dist)

    If camera model is pinhole, number of distortion coefficients is equal to 5
    If camera model is fisheye, number of distortion coefficients is equal to 4

    If cam.txt and dist.txt are available, they are used. Otherwise intrinsics.json is loaded
    """
    assert os.path.isdir(intrinsics_directory)
    cam_file = os.path.join(intrinsics_directory, "cam.txt")
    dist_file = os.path.join(intrinsics_directory, "dist.txt")
    intrisics_json_file = os.path.join(intrinsics_directory, "intrinsics.json")
    if os.path.isfile(cam_file) or os.path.isfile(dist_file):
        assert os.path.isfile(cam_file)
        assert os.path.isfile(dist_file)
        cam = np.loadtxt(cam_file)
        dist = np.loadtxt(dist_file)
        assert cam.shape == (3, 3)
        assert dist.size in [4, 5]
    else:
        assert os.path.isfile(intrisics_json_file)
        fs = cv2.FileStorage(intrisics_json_file, cv2.FILE_STORAGE_READ)
        assert fs.isOpened()
        cam_node = fs.getNode("camera_matrix").getNode("data")
        assert cam_node.isSeq()
        cam_data = []
        assert cam_node.size() == 9
        for i in range(cam_node.size()):
            cam_data.append(cam_node.at(i).real())
        cam = np.array(cam_data).reshape(3, 3)

        dist_node = fs.getNode("distortion_coefficients").getNode("data")
        assert dist_node.isSeq()
        dist_data = []
        assert dist_node.size() in [4, 5]
        for i in range(dist_node.size()):
            dist_data.append(dist_node.at(i).real())
        dist = np.array(dist_data)

    return cam, dist


if __name__ == "__main__":
    import fire
    fire.Fire({
        'calib_interactive': calib_interactive,
        'calib_json': calib_json
    })
