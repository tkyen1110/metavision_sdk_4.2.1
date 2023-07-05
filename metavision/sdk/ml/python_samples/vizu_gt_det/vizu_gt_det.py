# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

import sys
import argparse
import numpy as np
import os
import cv2

from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_core import EventBbox
from metavision_ml.detection_tracking.display_frame import draw_box_events


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Display GT and DET")
    parser.add_argument("--record_file", required=True, help="Input recording sequence")
    parser.add_argument("--gt", required=True, help="Filename GT boxes (.npy)")
    parser.add_argument("--det", required=True, help="Filename detections boxes (.npy)")
    parser.add_argument("--display_delta_t", type=int, default=10000, help="Display accumulation time (in µs)")

    args = parser.parse_args()

    # check args
    assert os.path.isfile(args.record_file)
    assert os.path.isfile(args.gt)
    assert os.path.splitext(args.gt)[1] == ".npy"
    assert os.path.isfile(args.det)
    assert os.path.splitext(args.det)[1] == ".npy"

    return args


def reformat_np_boxes(boxes):
    """ReFormat boxes to make sure they are consistent with EventBbox format
    This allows to be backward-compatible with old annotation.
        't' = 'ts'
        'class_confidence' = 'confidence'
    """
    if 't' not in boxes.dtype.names or 'class_confidence' not in boxes.dtype.names:
        new = np.zeros((len(boxes),), dtype=EventBbox)
        for name in boxes.dtype.names:
            if name == 'ts':
                new['t'] = boxes[name]
            elif name == 'confidence':
                new['class_confidence'] = boxes[name]
            else:
                new[name] = boxes[name]
        return new
    else:
        return boxes


def run(args):
    mv_it = EventsIterator(args.record_file, start_ts=0, delta_t=args.display_delta_t, relative_timestamps=False)
    ev_height, ev_width = mv_it.get_size()
    print("Frame size: width x height: {} x {}".format(ev_width, ev_height))

    gt_boxes = reformat_np_boxes(np.load(args.gt))
    assert gt_boxes.dtype == EventBbox
    label_map_gt = ["gt_{}".format(i) for i in range(np.max(gt_boxes["class_id"]) + 1)]

    det_boxes = np.load(args.det)
    assert det_boxes.dtype == EventBbox
    label_map_det = ["det_{}".format(i) for i in range(np.max(det_boxes["class_id"]) + 1)]

    frame_gt = np.zeros((ev_height, ev_width, 3), dtype=np.uint8)
    frame_det = frame_gt.copy()
    IMG_NAME = "Boxes GT (left) & DET (right)"
    cv2.namedWindow(IMG_NAME, cv2.WINDOW_NORMAL)

    prev_ts = 0
    for ev in mv_it:
        cur_ts = mv_it.get_current_time()
        BaseFrameGenerationAlgorithm.generate_frame(ev, frame_gt)
        frame_det[...] = frame_gt

        keep_gt_boxes = (gt_boxes["t"] >= prev_ts) * (gt_boxes["t"] < cur_ts)
        cur_gt_boxes = gt_boxes[keep_gt_boxes]
        draw_box_events(frame_gt, cur_gt_boxes, label_map_gt)

        keep_det_boxes = (det_boxes["t"] >= prev_ts) * (det_boxes["t"] < cur_ts)
        cur_det_boxes = det_boxes[keep_det_boxes]
        draw_box_events(frame_det, cur_det_boxes, label_map_det)

        frame_tot = cv2.hconcat([frame_gt, frame_det])
        cv2.imshow(IMG_NAME, frame_tot[..., ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_ts = cur_ts

    cv2.destroyAllWindows()


def main():
    args = parse_args(sys.argv[1:])
    print(args)
    run(args)


if __name__ == "__main__":
    main()
