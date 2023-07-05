# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

import argparse
import sys
import os
import json
import glob
import numpy as np
from metavision_sdk_core import EventBbox
from metavision_sdk_ml import EventTrackedBox
from metavision_ml.metrics.coco_eval import evaluate_detection


def get_gt_classes(filename_gt_labels):
    assert os.path.isfile(filename_gt_labels)
    dic_labels_gt_full = json.load(open(filename_gt_labels, "r"))
    assert len(dic_labels_gt_full) >= 1
    return dic_labels_gt_full


def get_det_classes_without_background(filename_info_ssd_json):
    assert os.path.isfile(filename_info_ssd_json)
    info_ssd_dic = json.load(open(filename_info_ssd_json, "r"))
    label_map = info_ssd_dic["label_map"]
    assert isinstance(label_map, list)
    assert len(label_map) > 1
    assert label_map[0] == "background"
    return label_map[1:]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluate coco kpi on detection/tracking results",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--GT-dir", required=True, help="Directory which contains the groundtruth")
    parser.add_argument("--GT-labels-dict", required=True, help="Filename which contains a dictionary for GT labels")
    parser.add_argument("--DET-dir", required=True, help="Directory which contains the detections")
    parser.add_argument("--DET-model-dir", required=True,
                        help="Model directory (must contain a file info_ssd_jit.json)")
    parser.add_argument("--evaluate-classes", nargs='+', type=str, default=None,
                        help="List of classes to evaluate (by default, all model's output classes)")
    parser.add_argument("--detections-delta-t", type=int, required=True,
                        help="Time between two consecutive detections (in µs)")
    parser.add_argument("--labels-delta-t", type=int, required=True, help="Time between two consecutive GT (in µs)")
    parser.add_argument(
        "--detection-type", choices=["csv_detections", "csv_tracks", "npy_detections", "npy_tracks"],
        required=True, help="Type of detection")
    parser.add_argument("--min-box-diag", type=int, default=60, help="Minimum diagonal size")
    parser.add_argument("--skip-ts", type=int, default=1e5, help="Skip beginning of sequence in µs")
    parser.add_argument("--height", type=int, default=720, help="Height of the input frame")
    parser.add_argument("--width", type=int, default=1280, help="Width of the input frame")

    args = parser.parse_args(argv)

    # check args
    assert os.path.isfile(args.GT_labels_dict)
    dic_labels_gt_full = get_gt_classes(args.GT_labels_dict)
    assert os.path.isdir(args.DET_model_dir)
    filename_info_ssd_json = os.path.join(args.DET_model_dir, "info_ssd_jit.json")
    assert os.path.isfile(filename_info_ssd_json)
    list_labels_det_full = get_det_classes_without_background(filename_info_ssd_json)
    inv_dic_labels_det_full = {}
    for idx_class, str_class in enumerate(list_labels_det_full):
        inv_dic_labels_det_full[str_class] = idx_class + 1
    list_dt_idx_to_keep = []
    if args.evaluate_classes is None:
        args.evaluate_classes = list_labels_det_full
        list_dt_idx_to_keep = range(1, len(list_labels_det_full) + 1)
    else:
        for current_class in args.evaluate_classes:
            assert current_class in list_labels_det_full, "Unknown detection class: {}".format(current_class)
            list_dt_idx_to_keep.append(inv_dic_labels_det_full[current_class])
    args._list_dt_idx_to_keep = list_dt_idx_to_keep
    assert len(args.evaluate_classes) >= 1
    map_gt_idx_to_det_idx = {}
    inv_dic_labels_gt_full = {v: k for k, v in dic_labels_gt_full.items()}
    list_gt_idx_to_keep = []
    for current_class in args.evaluate_classes:
        assert current_class in inv_dic_labels_gt_full, "Unknown labels class: {}".format(current_class)
        map_gt_idx_to_det_idx[int(inv_dic_labels_gt_full[current_class])] = inv_dic_labels_det_full[current_class]
        list_gt_idx_to_keep.append(int(inv_dic_labels_gt_full[current_class]))
    args._map_gt_idx_to_det_idx = map_gt_idx_to_det_idx
    args._list_gt_idx_to_keep = list_gt_idx_to_keep

    return args


def main():
    args = parse_args(sys.argv[1:])
    print(args)
    stats = evaluate_folders(args)
    print(stats)


def reformat_np_boxes(boxes):
    """ReFormat boxes according to new rule
    This allows to be backward-compatible with imerit annotation.
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


def reformat_csv_boxes(boxes, box_format):
    assert box_format in ["csv_detections", "csv_tracks"]
    rows, cols = boxes.shape
    if box_format == "csv_detections":
        assert cols == len(EventBbox)
        new = np.empty(rows, dtype=EventBbox)
        new["t"] = boxes[:, 0]
        new["class_id"] = boxes[:, 1]
        new["track_id"] = boxes[:, 2]
        new["x"] = boxes[:, 3]
        new["y"] = boxes[:, 4]
        new["w"] = boxes[:, 5]
        new["h"] = boxes[:, 6]
        new["class_confidence"] = boxes[:, 7]
    else:
        assert box_format == "csv_tracks"
        assert cols == len(EventTrackedBox)
        new = np.empty(rows, dtype=EventBbox)
        new["t"] = boxes[:, 0]
        new["class_id"] = boxes[:, 1]
        new["track_id"] = boxes[:, 2]
        new["x"] = boxes[:, 3]
        new["y"] = boxes[:, 4]
        new["w"] = boxes[:, 5]
        new["h"] = boxes[:, 6]
        new["class_confidence"] = boxes[:, 8]
    return new


def convert_EventTrackedBox_to_EventBbox(boxes):
    assert boxes.dtype == EventTrackedBox
    new = np.empty(len(boxes), dtype=EventBbox)
    new["t"] = boxes["t"]
    new["x"] = boxes["x"]
    new["y"] = boxes["y"]
    new["w"] = boxes["w"]
    new["h"] = boxes["h"]
    new["class_id"] = boxes["class_id"]
    new["track_id"] = boxes["track_id"]
    new["class_confidence"] = boxes["tracking_confidence"]
    return new


def filter_boxes(boxes, skip_ts=int(5e5), min_box_diag=60, min_box_side=20):
    """Filters boxes according to the rule described in the following paper:
    https://nips.cc/virtual/2020/public/poster_c213877427b46fa96cff6c39e837ccee.html

    To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
    To note: we assume the initial time of the video is always 0
    Args:
        boxes (np.ndarray): dtype is EventBbox
    Returns:
        boxes: filtered boxes
    """
    ts = boxes['t']
    width = boxes['w']
    height = boxes['h']
    diag_square = width**2+height**2
    mask = (ts > skip_ts)*(diag_square >= min_box_diag**2)*(width >= min_box_side)*(height >= min_box_side)
    return boxes[mask]


def evaluate_folders(args):
    gt_file_paths = sorted(glob.glob(args.GT_dir + '/**/*.npy', recursive=True))
    if args.detection_type in ["csv_detections", "csv_tracks"]:
        extension_det = "csv"
    else:
        assert args.detection_type in ["npy_detections", "npy_tracks"]
        extension_det = "npy"

    dt_file_paths = sorted(glob.glob(args.DET_dir + '/**/*.{}'.format(extension_det), recursive=True))

    assert len(dt_file_paths) == len(gt_file_paths)
    print(f"There are {len(gt_file_paths)} GT bboxes and {len(dt_file_paths)} PRED bboxes")

    gt_boxes_list = [np.load(p) for p in gt_file_paths]
    gt_boxes_list = [reformat_np_boxes(p) for p in gt_boxes_list]
    for idx_gt in range(len(gt_boxes_list)):
        assert gt_boxes_list[idx_gt].dtype == EventBbox

        # keep only desired classes
        mask_keep = np.zeros(len(gt_boxes_list[idx_gt]), dtype=bool)
        for k in args._list_gt_idx_to_keep:
            mask_keep = np.logical_or(mask_keep, gt_boxes_list[idx_gt]["class_id"] == k)
        gt_boxes_list[idx_gt] = gt_boxes_list[idx_gt][mask_keep]

        # use det indices for classes to keep
        for idx_box in range(len(gt_boxes_list[idx_gt])):
            class_id = gt_boxes_list[idx_gt][idx_box]['class_id']
            gt_boxes_list[idx_gt][idx_box]['class_id'] = args._map_gt_idx_to_det_idx[class_id]

    if args.detection_type == "csv_detections":
        dt_boxes_list = [np.loadtxt(p) for p in dt_file_paths]
        dt_boxes_list = [reformat_csv_boxes(p, box_format=args.detection_type) for p in dt_boxes_list]
    elif args.detection_type == "csv_tracks":
        dt_boxes_list = [np.loadtxt(p, delimiter=",") for p in dt_file_paths]
        dt_boxes_list = [reformat_csv_boxes(p, box_format=args.detection_type) for p in dt_boxes_list]
    elif args.detection_type == "npy_tracks":
        dt_boxes_list = [np.load(p) for p in dt_file_paths]
        dt_boxes_list = [convert_EventTrackedBox_to_EventBbox(p) for p in dt_boxes_list]
    else:
        assert args.detection_type == "npy_detections"
        dt_boxes_list = [np.load(p) for p in dt_file_paths]
    for idx_dt in range(len(dt_boxes_list)):
        assert dt_boxes_list[idx_dt].dtype == EventBbox
        # keep only desired classes
        mask_keep = np.zeros(len(dt_boxes_list[idx_dt]), dtype=bool)
        for k in args._list_dt_idx_to_keep:
            mask_keep = np.logical_or(mask_keep, dt_boxes_list[idx_dt]["class_id"] == k)
        dt_boxes_list[idx_dt] = dt_boxes_list[idx_dt][mask_keep]

    def filter_boxes_fn(x): return filter_boxes(boxes=x, skip_ts=args.skip_ts,
                                                min_box_diag=args.min_box_diag, min_box_side=args.min_box_diag/3)

    gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
    dt_boxes_list = map(filter_boxes_fn, dt_boxes_list)

    dummy_labelmap = ["class_{}".format(i) for i in range(max(args._list_dt_idx_to_keep) + 1)]
    stats = evaluate_detection(
        gt_boxes_list=gt_boxes_list, dt_boxes_list=dt_boxes_list, classes=dummy_labelmap, height=args.height,
        width=args.width, time_tol=(args.labels_delta_t // 2) + 1, eval_rate=args.detections_delta_t)
    return stats


if __name__ == "__main__":
    main()
