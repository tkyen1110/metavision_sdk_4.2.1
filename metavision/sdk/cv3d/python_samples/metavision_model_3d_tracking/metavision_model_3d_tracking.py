# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Script to detect and track an object, the 3D model of which is provided by the user.
"""

import cv2
import numpy as np
import os
import json

from metavision_core.event_io import RawReader
from metavision_sdk_core import MostRecentTimestampBuffer, PeriodicFrameGenerationAlgorithm
from metavision_sdk_cv import load_camera_geometry
from metavision_sdk_cv3d import EigenMatrix4f, Model3dDetectionAlgorithm, Model3dTrackingAlgorithm
import metavision_sdk_cv3d as mv_cv3d  # Static methods: load_model_3d_from_json, select_visible_edges, draw_edges
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model 3D Tracking sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base options
    base_options = parser.add_argument_group('Base options')
    base_options.add_argument('-i', '--input-base-path', dest='base_path', default="",
                              help="Base path used to load a recording, a 3D model, a camera calibration and an "
                              "initialization pose. Apart from the recording, every file can also be set individually.")
    base_options.add_argument('-r', '--input-raw-file', dest='raw_path', default="",
                              help="Path to input RAW file. Ignored if a base path is set.")
    base_options.add_argument('-m', '--input-model-file', dest='model_path', default="",
                              help="Path to a JSON file containing the description of a 3D model.")
    base_options.add_argument('-p', '--input-pose-file', dest='model_pose_path', default="",
                              help="Path to a JSON file containing a camera pose used to detect the 3D model.")
    base_options.add_argument('-c', '--input-calibration-file', dest='calibration_path', default="",
                              help="Path to a JSON file containing the camera's calibration.")
    base_options.add_argument('--process-from', dest='process_from', type=int, default=0,
                              help='Start time to process events (in us).')
    base_options.add_argument('--process-to', dest='process_to', type=int, default=None,
                              help='End time to process events (in us).')

    # Detection options
    detection_options = parser.add_argument_group('Detection options')
    detection_options.add_argument(
        '--num-detections', dest='n_detections', type=int, default=10,
        help='Number of successive valid detection to consider the model as detected.')
    detection_options.add_argument('--detection-period', dest='detection_period_us', type=int,
                                   default=10000, help='Amount of time after which a detection is attempted.')

    # Tracking options
    tracking_options = parser.add_argument_group('Tracking options')
    tracking_options.add_argument('--n-events', dest='n_events', type=int, default=5000,
                                  help='Number of events after which a tracking step is attempted.')
    tracking_options.add_argument('--n-us', dest='n_us', type=int, default=10000,
                                  help='Amount of time after which a tracking step is attempted.')

    # Outcome Options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument('-a', '--accumulation-time', dest='display_acc_time_us', type=int, default=5000,
                                 help='Accumulation time in us used to generate frames for display.')
    outcome_options.add_argument('-f', '--fps', dest='display_fps', type=float, default=30.,
                                 help="Display's fps.")
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to output AVI file to save slow motion video. A frame is generated after each process of the algorithm. The video "
        "will be written only for processed events.")

    args = parser.parse_args()

    if args.process_to and args.process_from > args.process_to:
        print(f"The processing time interval is not valid. [{args.process_from,}, {args.process_to}]")
        exit(1)

    if not args.base_path:
        if not args.raw_path:
            print("You should provide either a base path or a path to a RAW file.")
            exit(2)
        if not os.path.exists(args.raw_path):
            print("Invalid RAW path: " + args.raw_path)
            exit(3)
        args.base_path = os.path.splitext(args.raw_path)[0]  # Remove the .raw extension
    else:
        extension = os.path.splitext(args.base_path)[1]
        if extension != "":
            print("Invalid base path. Remove the extension " + extension)
            if extension == ".raw":
                print("Or use -r instead of -i to run the app on a RAW file.")
            exit(4)

        root_path = os.path.dirname(args.base_path)  # Directory containing the base path
        if not os.path.exists(root_path):
            print("Invalid base path: " + args.base_path)
            exit(5)
        args.raw_path = ""  # Ignored if a base path is set

    # Model
    if not args.model_path:
        args.model_path = args.base_path + ".json"
    if not os.path.exists(args.model_path):
        print("Invalid model path: " + args.model_path)
        exit(6)

    # Model pose
    if not args.model_pose_path:
        args.model_pose_path = args.base_path + "_init_pose.json"
    if not os.path.exists(args.model_pose_path):
        print("Invalid model's pose path: " + args.model_pose_path)
        exit(7)

    # Calibration
    if not args.calibration_path:
        args.calibration_path = os.path.join(os.path.dirname(args.base_path), "calibration.json")
    if not os.path.exists(args.calibration_path):
        print("Invalid camera calibration path: " + args.calibration_path)
        exit(8)

    print("Base path       :" + args.base_path)
    print("Raw path        :" + args.raw_path)
    print("Model path      :" + args.model_path)
    print("Model pose path :" + args.model_pose_path)
    print("Calibration path:" + args.calibration_path)

    return args


def load_init_pose(model_pose_path):
    with open(model_pose_path) as json_file:
        data = json.load(json_file)["camera_pose"]
        inv_pose = data["T_w_c"]
        pose = np.linalg.inv(inv_pose)

        T_c_w = EigenMatrix4f()
        T_c_w.numpy()[...] = pose[...]

        return T_c_w


def main():
    """ Main """
    args = parse_args()

    # [LOAD_3D_MODEL_BEGIN]
    # Load 3D Model
    model_3d = mv_cv3d.load_model_3d_from_json(args.model_path)
    if not model_3d:
        print("Impossible to load the 3D model from " + args.model_path)
        exit(9)
    # [LOAD_3D_MODEL_END]

    T_c_w_init = load_init_pose(args.model_pose_path)  # Load initial pose
    T_c_w = T_c_w_init.copy()

    # Raw Reader on Camera or RAW file
    reader = RawReader(args.raw_path)

    height, width = reader.get_size()  # Camera Geometry
    camera_geometry = load_camera_geometry(args.calibration_path)
    if not camera_geometry:
        print("Impossible to load the camera calibration from " + args.calibration_path)
        exit(10)

    # Time Surface
    time_surface = MostRecentTimestampBuffer(height, width, 2)

    # [INSTANTIATE_ALGOS_BEGIN]
    # Detection and Tracking algorithms
    detection_algo = Model3dDetectionAlgorithm(camera_geometry, model_3d, time_surface)
    tracking_algo = Model3dTrackingAlgorithm(camera_geometry, model_3d, time_surface)
    # [INSTANTIATE_ALGOS_END]

    visible_edges = set()
    detected_edges = set()
    is_tracking = False
    n_detection = 0

    def set_detection_params():
        nonlocal T_c_w
        T_c_w = T_c_w_init.copy()
        detection_algo.set_init_pose(T_c_w_init)

    def set_tracking_params(ts):
        nonlocal n_detection
        tracking_algo.set_previous_camera_pose(ts, T_c_w)
        n_detection = 0

    set_detection_params()

    # Window - Graphical User Interface (Display tracking results and process keyboard events)
    with MTWindow(title="Model 3D Tracking", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        if args.out_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_name = args.out_video + ".avi"
            video_writer = cv2.VideoWriter(video_name, fourcc, 20, (width, height))

        def keyboard_cb(key, scancode, action, mods):
            nonlocal is_tracking
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            elif key == UIKeyEvent.KEY_SPACE:
                is_tracking = False

        window.set_keyboard_callback(keyboard_cb)
        print("Press 'q' to leave the program.\n"
              "Press 'Space' to reset the tracking.\n")

        # [FRAME_GENERATOR_CALLBACK_BEGIN]
        # Periodic Event Frame Generator
        def periodic_frame_gen_cb(ts, cv_frame):
            nonlocal is_tracking
            nonlocal visible_edges
            nonlocal detected_edges

            cv2.putText(cv_frame, str(ts), (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                        (0, 255, 0) if is_tracking else (0, 0, 255))
            # Be careful, here the events and the 3D model are not rendered in a tightly synchronized way,
            # meaning that some shifts might occur. However, most of the time they should not be noticeable
            if is_tracking:
                visible_edges = mv_cv3d.select_visible_edges(T_c_w, model_3d)
                mv_cv3d.draw_edges(camera_geometry, T_c_w, model_3d, visible_edges, cv_frame, (0, 255, 0))
                cv2.putText(cv_frame, "tracking", (0, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
            else:
                mv_cv3d.draw_edges(camera_geometry, T_c_w, model_3d, visible_edges, cv_frame, (0, 0, 255))
                mv_cv3d.draw_edges(camera_geometry, T_c_w, model_3d, detected_edges, cv_frame, (0, 255, 0))
                cv2.putText(cv_frame, "detecting", (0, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            window.show_async(cv_frame)
            if args.out_video:
                video_writer.write(cv_frame)
        # [FRAME_GENERATOR_CALLBACK_END]

        events_frame_gen_algo = PeriodicFrameGenerationAlgorithm(width, height, args.display_acc_time_us,
                                                                 args.display_fps)
        events_frame_gen_algo.set_output_callback(periodic_frame_gen_cb)

        # [PROCESSING_LOOP_BEGIN]
        # Process events
        reader.seek_time(args.process_from)
        while (not reader.is_done()) and (args.process_to is None or reader.current_time < args.process_to):
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            prev_is_tracking = is_tracking

            if is_tracking:
                evs = reader.load_n_events(args.n_events)
                is_tracking = tracking_algo.process_events(evs, T_c_w)
            else:
                evs = reader.load_delta_t(args.detection_period_us)
                success, visible_edges, detected_edges = detection_algo.process_events(evs, T_c_w)
                if success:
                    # We wait for several detections before considering the model as detected
                    # to avoid false positive detections
                    n_detection += 1
                    is_tracking = (n_detection > args.n_detections)

            # The frame generation algorithm processing can trigger a call to show which can trigger
            # a reset of the tracking if the space bar has been pressed.
            events_frame_gen_algo.process_events(evs)

            if prev_is_tracking != is_tracking:
                if is_tracking:
                    set_tracking_params(evs["t"][-1])
                else:
                    set_detection_params()
        # [PROCESSING_LOOP_END]

            if window.should_close():
                break

        if args.out_video:
            video_writer.release()
            print("Video has been saved in " + video_name)


if __name__ == "__main__":
    main()
