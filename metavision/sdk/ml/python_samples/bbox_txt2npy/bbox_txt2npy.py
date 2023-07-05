# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Tool to convert BBOX text format (BB_CREATE, ...) into EventBbox numpy arrays
"""

import numpy as np
import argparse
import os
import sys

from metavision_sdk_core import EventBbox


def bboxstr2array(lines):
    """Converts the lines of a bbox text file into an EventBbox numpy array

    Args:
        lines: list of lines (content of the _bbox.txt file)
    """
    assert isinstance(lines, list)
    dic_current_boxes = {}
    list_boxes = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        line_list = line.split()
        assert len(line_list) >= 2, "Invalid file:\n{}".format(line)
        ts = int(line_list[0])
        object_id = int(line_list[1])
        command = line_list[2]
        assert command in ["BB_CREATE", "BB_MOVE", "BB_RESIZE",
                           "BB_MOVE_AND_RESIZE", "BB_DELETE"], "Invalid file:\n{}".format(line)
        if command == "BB_CREATE":
            assert len(line_list) == 9, "Invalid line for BB_CREATE:\n{}".format(line)
            class_id = int(line_list[3])
            x, y, width, height, confidence = [float(i) for i in line_list[4:]]
            box = np.zeros(1, dtype=EventBbox)
            box["t"] = ts
            box["x"] = x
            box["y"] = y
            box["w"] = width
            box["h"] = height
            box["class_id"] = class_id
            box["track_id"] = object_id
            box["class_confidence"] = confidence
            list_boxes.append(box.copy())
            assert object_id not in dic_current_boxes
            dic_current_boxes[object_id] = box.copy()
        elif command == "BB_MOVE":
            assert object_id in dic_current_boxes
            x, y, confidence = [float(i) for i in line_list[3:]]
            dic_current_boxes[object_id]["x"] = x
            dic_current_boxes[object_id]["y"] = y
            dic_current_boxes[object_id]["class_confidence"] = confidence
            list_boxes.append(dic_current_boxes[object_id].copy())
        elif command == "BB_RESIZE":
            assert object_id in dic_current_boxes
            width, height, confidence = [float(i) for i in line_list[3:]]
            dic_current_boxes[object_id]["t"] = ts
            dic_current_boxes[object_id]["w"] = width
            dic_current_boxes[object_id]["h"] = height
            dic_current_boxes[object_id]["class_confidence"] = confidence
            list_boxes.append(dic_current_boxes[object_id].copy())
        elif command == "BB_MOVE_AND_RESIZE":
            assert object_id in dic_current_boxes
            x, y, width, height, confidence = [float(i) for i in line_list[3:]]
            dic_current_boxes[object_id]["t"] = ts
            dic_current_boxes[object_id]["x"] = x
            dic_current_boxes[object_id]["y"] = y
            dic_current_boxes[object_id]["w"] = width
            dic_current_boxes[object_id]["h"] = height
            dic_current_boxes[object_id]["class_confidence"] = confidence
            list_boxes.append(dic_current_boxes[object_id].copy())
        elif command == "BB_DELETE":
            assert object_id in dic_current_boxes
            del dic_current_boxes[object_id]
        else:
            raise RuntimeError("Wrong key in bbox file: {}".format(command))
    if len(dic_current_boxes) != 0:
        print("Warning: Some boxes were created but not deleted !  Remaining keys: {}".format(dic_current_boxes.keys()))
    boxes_array = np.concatenate(list_boxes)
    return boxes_array


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Convert bbox text format to npy")
    parser.add_argument("-i", dest="input_filename", required=True, help="Input bbox TXT filename")
    parser.add_argument("-o", dest="output_filename", required=True, help="Output npy filename")
    args = parser.parse_args(argv)
    assert os.path.isfile(args.input_filename)
    assert not os.path.exists(args.output_filename), "Output filename already exists"
    assert args.output_filename.lower().endswith(".npy"), "Output filename should by a numpy array (.npy)"
    return args


def process_args(args):
    lines = open(args.input_filename, "r").readlines()
    bboxes_array = bboxstr2array(lines)
    np.save(args.output_filename, bboxes_array)


def main():
    args = parse_args(sys.argv[1:])
    process_args(args)


if __name__ == "__main__":
    main()
