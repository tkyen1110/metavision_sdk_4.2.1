# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

import argparse
import os
import sys
import numpy as np
import cv2
from skvideo.io import FFmpegWriter
from metavision_core.event_io import EventsIterator
from metavision_ml.detection_tracking import ObjectDetector
from metavision_ml.detection_tracking import draw_detections_and_tracklets
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_cv import TrailFilterAlgorithm, SpatioTemporalContrastAlgorithm
from metavision_sdk_ml import DataAssociation
from metavision_sdk_core import EventBbox


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Detection and Tracking Inference pipeline",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_options = parser.add_argument_group("Input options")
    input_options.add_argument("--record_file", default="",
                               help="File name to read events from. Leave empty to use a live camera")
    input_options.add_argument("--pipeline_delta_t", type=int, default=10000,
                               help="data accumulation time for EventsIterator (in µs)")
    input_options.add_argument("--start_ts", type=int, default=None, help="start timestamp (in µs)")
    input_options.add_argument("--end_ts", type=int, default=None, help="end timestamp (in µs)")

    output_options = parser.add_argument_group("Output options")
    output_options.add_argument("--display", action='store_true', help="Enable output display")
    output_options.add_argument("--output_detections_filename", default=None,
                                help="File name to write detected bbox (EventBbox npy)")
    output_options.add_argument("--output_tracks_filename", default=None,
                                help="File name to write tracked bbox (EventTrackedBox npy)")
    output_options.add_argument("--output_video_filename", default=None, help="File name to write a video (in mp4)")

    object_detector_options = parser.add_argument_group("Object Detector options")
    object_detector_options.add_argument(
        "--object_detector_dir", required=True,
        help="Directory of the object detector, including two files named: 1) 'model.ptjit' 2)'info_ssd_jit.json'; "
        "INFO: 'model.ptjit' is a Pytorch model compiled and exported with TorchScript; 'info_ssd_jit.json' is a file which contains "
        "hyperparameters used to train the model")
    object_detector_options.add_argument(
        "--detector_confidence_threshold", type=float, default=None,
        help="Use this confidence threshold value instead of the one stored  in the model's json. A valid value should be in: ]0., 1[")
    object_detector_options.add_argument(
        "--detector_NMS_IOU_threshold", type=float, default=None,
        help="Use this IOU (Intersection Over Union) threshold instead of the one stored in the model's json. A valid value should be in: ]0., 1[")
    object_detector_options.add_argument(
        "--device", default="gpu",
        help="Machine Learning Device; Choice of 'cpu', 'gpu', or ('gpu:0', 'gpu:1', etc. if several gpu are available)")
    object_detector_options.add_argument(
        "--network_input_width", type=int, default=None,
        help="Neural Network input width (by default same as event frame width)")
    object_detector_options.add_argument(
        "--network_input_height", type=int, default=None,
        help="Neural Network input height (by default same as event frame height")

    noise_filtering_options = parser.add_argument_group("Noise Filtering options")
    noise_filtering_options.add_argument(
        "--noise_filtering_type", default="trail", choices=["trail", "stc"],
        help="Type of noise filtering: STC or Trail")
    noise_filtering_options.add_argument(
        "--noise_filtering_threshold", type=int, default=10000,
        help="Length of the time window for STC or Trail filtering (in µs)")

    data_association_options = parser.add_argument_group("Data Association options")
    data_association_options.add_argument(
        "--detection_merge_weight", type=float, default=0.7,
        help="Weight used to compute weighted average of detection and already tracked position (Pos = Detection * Weight + (1 - Weight) * Pos")
    data_association_options.add_argument("--deletion_time", type=int, default=100000,
                                          help="Time without activity after which the track is detected")
    data_association_options.add_argument(
        "--max_iou_inter_track", type=float, default=0.5,
        help="Maximum Intersection Over Union (IOU) inter tracklets before deleting one")
    data_association_options.add_argument(
        "--iou_to_match_a_detection", type=float, default=0.2,
        help="Minimum Intersection Over Union (IOU) to match a detection and existing track")
    data_association_options.add_argument(
        "--max_iou-for_one_det_to_many_tracks", type=float, default=0.5,
        help="Threshold at which the tracking is not done if several tracks match with a higher IOU")
    data_association_options.add_argument(
        "--use_descriptor", action='store_true',
        help="Boolean to enable the use of a Histogram Of Gradient (HOG) descriptor")
    data_association_options.add_argument(
        "--number_of_consecutive_detections_to_create_a_new_track", type=int, default=1,
        help="Number of consecutive detections to create a new track")
    data_association_options.add_argument(
        "--timesurface_delta_t", type=int, default=200000,
        help="Time after which the events are removed from the time surface")
    data_association_options.add_argument(
        "--do_not_update_tracklets_between_detections", action='store_false',
        dest='update_tracklets_between_detections',
        help="Boolean to disable update of tracklets between detections")
    parser.add_argument('--max-duration', type=int, default=None,
                        help='maximum duration of the inference file in us.')
    
    args = parser.parse_args(argv)

    # check args
    assert args.pipeline_delta_t > 0
    if args.record_file == "":
        # Live camera
        print("No recording is specified. Using live camera...")
        if args.start_ts is not None:
            raise ValueError("Invalid argument --start_ts when using a live camera")
        if args.end_ts is not None:
            raise ValueError("Invalid argument --end_ts when using a live camera")
        args.start_ts = 0
        args.end_ts = sys.maxsize
    else:
        # Use a recording file
        assert os.path.isfile(args.record_file)
        if args.start_ts is None:
            args.start_ts = 0
        if args.start_ts is not None:
            assert args.start_ts >= 0
        if args.end_ts is None:
            args.end_ts = sys.maxsize
        if args.end_ts is not None:
            assert args.end_ts > args.start_ts

    if args.output_detections_filename is not None:
        assert args.output_detections_filename.lower().endswith(".npy"), "Detection results filename should be .npy"
    if args.output_tracks_filename:
        assert args.output_tracks_filename.lower().endswith(".npy"), "Tracks results filename should be .npy"
    if args.output_video_filename is not None:
        assert args.output_video_filename.lower().endswith(".mp4"), "Video results filename should be .mp4"

    assert args.noise_filtering_type in ["trail", "stc"]
    assert args.noise_filtering_threshold >= 0

    assert os.path.isdir(args.object_detector_dir)

    assert args.device.startswith("gpu") or args.device.startswith("cpu")

    if args.network_input_width is not None:
        assert args.network_input_width > 0
    if args.network_input_height is not None:
        assert args.network_input_height > 0

    assert args.detection_merge_weight > 0

    assert args.deletion_time > 0
    assert args.max_iou_inter_track > 0
    assert args.iou_to_match_a_detection > 0
    assert args.max_iou_for_one_det_to_many_tracks > 0
    assert args.number_of_consecutive_detections_to_create_a_new_track > 0
    assert args.timesurface_delta_t > 0

    return args


def run(args):
    # Init Events Iterator
    mv_it = EventsIterator(args.record_file, start_ts=args.start_ts,
                           delta_t=args.pipeline_delta_t, relative_timestamps=False,
                           max_duration=args.max_duration)
    # Set ERC to 20Mev/s
    if hasattr(mv_it.reader, "device") and mv_it.reader.device:
        erc_module = mv_it.reader.device.get_i_erc_module()
        if erc_module:
            erc_module.set_cd_event_rate(20000000)
            erc_module.enable(True)

    ev_height, ev_width = mv_it.get_size()

    # Init Object Detector
    network_input_width = ev_width if args.network_input_width is None else args.network_input_width
    network_input_height = ev_height if args.network_input_height is None else args.network_input_height
    object_detector = ObjectDetector(args.object_detector_dir,
                                     events_input_width=ev_width,
                                     events_input_height=ev_height,
                                     runtime=args.device,
                                     network_input_width=network_input_width,
                                     network_input_height=network_input_height)
    if args.detector_confidence_threshold:
        object_detector.set_detection_threshold(args.detector_confidence_threshold)
    if args.detector_NMS_IOU_threshold:
        object_detector.set_iou_threshold(args.detector_NMS_IOU_threshold)

    cdproc = object_detector.get_cd_processor()
    frame_buffer = cdproc.init_output_tensor()

    label_map_dic = dict(zip(range(object_detector.get_num_classes()), object_detector.get_label_map()))

    # Init Noise Filter
    if args.noise_filtering_type == "trail":
        noise_filter = TrailFilterAlgorithm(width=ev_width, height=ev_height, threshold=args.noise_filtering_threshold)
    elif args.noise_filtering_type == "stc":
        noise_filter = SpatioTemporalContrastAlgorithm(
            width=ev_width, height=ev_height, threshold=args.noise_filtering_threshold)
    else:
        raise RuntimeError("Unknown noise filtering type: {}".format(args.noise_filtering_type))
    ev_filtered_buffer = noise_filter.get_empty_output_buffer()

    # Init Data Association
    data_assoc = DataAssociation(
        detection_merge_weight=args.detection_merge_weight, deletion_time=args.deletion_time,
        max_iou_inter_track=args.max_iou_inter_track, iou_to_match_a_detection=args.iou_to_match_a_detection,
        max_iou_for_one_det_to_many_tracks=args.max_iou_for_one_det_to_many_tracks, use_descriptor=args.use_descriptor,
        number_of_consecutive_detections_to_create_a_new_track=args.
        number_of_consecutive_detections_to_create_a_new_track, width=ev_width, height=ev_height,
        time_surface_delta_t=args.timesurface_delta_t,
        update_tracklets_between_detections=args.update_tracklets_between_detections)
    data_assoc_buffer = data_assoc.get_empty_output_buffer()

    # Init Display
    if args.output_video_filename is not None:
        video_process = FFmpegWriter(args.output_video_filename)
    else:
        video_process = None

    if args.display:
        cv2.namedWindow("Detection and Tracking", cv2.WINDOW_NORMAL)
    else:
        print("Display window disabled by default. Use  --display   to enable display during processing")

    if video_process or args.display:
        frame = np.zeros((ev_height, ev_width * 2, 3), dtype=np.uint8)

    # Results
    list_all_detections = []
    list_all_tracks = []
    list_recent_tracks_for_display = []

    # Process the sequence
    current_frame_start_ts = args.start_ts
    for ev in mv_it:
        ts = mv_it.get_current_time()
        if ts > args.end_ts:
            break
        cdproc.process_events(current_frame_start_ts, ev, frame_buffer)
        detections = np.empty(0, dtype=EventBbox)
        if ts % object_detector.get_accumulation_time() == 0:
            # call neural network to detect objects
            detections = object_detector.process(ts, frame_buffer)
            # reset neural network input frame
            frame_buffer.fill(0)
            current_frame_start_ts = ts
            if args.output_detections_filename:
                list_all_detections.append(detections.copy())

        noise_filter.process_events(ev, ev_filtered_buffer)
        data_assoc.process_events(ts, ev_filtered_buffer, detections, data_assoc_buffer)
        if args.output_tracks_filename:
            list_all_tracks.append(data_assoc_buffer.numpy().copy())

        if video_process or args.display:
            # build image frame
            BaseFrameGenerationAlgorithm.generate_frame(ev, frame[:, :ev_width])
            frame[:, ev_width:] = frame[:, :ev_width]
            list_recent_tracks_for_display.append(data_assoc_buffer.numpy().copy())
            if len(list_recent_tracks_for_display) > 50:
                list_recent_tracks_for_display.pop(0)
            draw_detections_and_tracklets(ts=ts, frame=frame, width=ev_width, height=ev_height,
                                          detections=detections, tracklets=data_assoc_buffer.numpy(),
                                          label_map=label_map_dic,
                                          list_previous_tracks=list_recent_tracks_for_display)

        if args.display:
            # display image on screen
            cv2.imshow('Detection and Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if video_process:
            # write video
            video_process.writeFrame(frame.astype(np.uint8)[..., ::-1])

    # Save results
    if args.output_detections_filename:
        np.save(args.output_detections_filename, np.concatenate(list_all_detections))
    if args.output_tracks_filename:
        np.save(args.output_tracks_filename, np.concatenate(list_all_tracks))

    # Clean display
    if video_process:
        video_process.close()
    if args.display:
        cv2.destroyAllWindows()


def main():
    args = parse_args(sys.argv[1:])
    print(args)
    run(args)


if __name__ == "__main__":
    main()
