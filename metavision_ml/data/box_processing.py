# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Collections of functions to add bounding box loading capabilities to the SequentialDataLoader
"""
from __future__ import division

import numpy as np
import json

from metavision_sdk_core import EventBbox
from metavision_core.event_io import EventNpyReader
from metavision_ml.data.scheduler import FileMetadata


def load_boxes(metadata, batch_start_time, duration, tensor, **kwargs):
    """Function to fetch boxes and preprocess them. Should be passed to a SequentialDataLoader.

    Since this function has additional arguments compared to load_labels_stub, one has to specialize it:

    Examples:
        >>> from functools import partial
        >>> n_classes = 21
        >>> class_lookup = np.arange(n_classes)  # each class is mapped to itself
        >>> load_boxes_function = partial(load_boxes, class_lookup=class_lookup)

    Args:
        metadata (FileMetadata): Record details.
        batch_start_time (int): (us) Where to seek in the file to load corresponding bounding boxes
        duration (int): (us) How long to load events from bounding box file
        tensor (np.ndarray): Current preprocessed input, can be used for data dependent preprocessing,
            for instance remove boxes without any features in them.
        **kwargs, containing:
            class_lookup (np.array): Look up array for class indices.
            labelling_delta_t (int): Indicates the period of labelling in order to only consider time bins
                with actual labels when computing the loss.
            min_box_diag (int): Diagonal value under which boxes are not considerated. Defaults to 60 pixels.

    Returns:
        boxes (List[np.ndarray]): List of structured array of dtype EventBbox corresponding to each time
            bins.
        frames_contain_gt (np.ndarray): This boolean mask array of *length* num_tbins indicates
            whether the frame contains a label. It is used to differentiate between time bins that actually
            contain an empty label (for instance no bounding boxes) from time bins that weren't labeled due
            to cost constraints. The latter time bins shouldn't contribute to supervised losses used during
            training.

    """
    assert 'class_lookup' in kwargs, "you need to provide a class_lookup array corresponding to your labels!"
    class_lookup = kwargs['class_lookup']

    # size of the image
    height_orig, width_orig = metadata.get_original_size()
    # size of the feature tensor
    num_tbins, _, height_dst, width_dst = tensor.shape

    box_events = load_box_events(metadata, batch_start_time, duration)

    """Frames are considered as annotated if there is any bbox inside
    If no bounding box is present in a frame but you still want the
    training to consider it to be annotated, you must annotate it with
    a special "empty" label, included in the label_map_dictionary.json.
    This "empty" label is there to mark empty frames that are annotated,
    the box annotated is removed after.

    Example:
    -------
        label_map = {'car': 0, 'pedestrian': 1, 'empty': 100}
        wanted_keys = {'car', 'pedestrian'}

    the output will filter the "ignored" class
    However, frame_is_labeled will be set to True even
    when the frame contains a single "ignored" box.
    """
    tmp = split_boxes(box_events, batch_start_time=batch_start_time,
                      delta_t=duration // num_tbins,
                      num_tbins=num_tbins)
    frame_is_labeled = np.array([len(item) > 0 for item in tmp])

    # we filter out object by small diagonal
    min_box_diag_network = kwargs.get('min_box_diag_network', 0)

    # clip some boxes that might be out of field of view.
    shift = int(np.log2(width_orig / width_dst))

    box_events = clip_boxes(box_events, width_orig, height_orig)

    # filter classes
    idx_to_filter = np.array([])
    area_box_filter = 0.1
    filter_beginning = True
    ignore_filtered = False
    total_tbins_delta_t = duration
    if area_box_filter > 0:
        # we filter event from the beginning of the video
        if filter_beginning:
            time_to_filter = int(5e5)
            last_time_to_filter = time_to_filter - batch_start_time  # last_time_to_filter + first_time < time_to_filter
        else:
            last_time_to_filter = None
        idx_to_filter = filter_empty_tensor(tensor.numpy(), box_events, shift=shift,
                                            area_box_filter=area_box_filter,
                                            time_per_bin=total_tbins_delta_t // num_tbins,
                                            last_time_to_filter=last_time_to_filter, batch_start_time=batch_start_time)
    else:
        idx_to_filter = np.array([])

    box_events = filter_boxes(box_events, class_lookup, idx_to_filter, ignore_filtered)

    # and finally put them each in their time bin
    boxes = split_boxes(box_events, batch_start_time=batch_start_time,
                        delta_t=duration // num_tbins,
                        num_tbins=num_tbins)

    min_box_diag_rescaled = min_box_diag_network
    for i, box in enumerate(boxes):
        box = rescale_boxes(box, width_orig, height_orig, width_dst, height_dst)
        idx_to_keep = (box["w"]**2 + box["h"]**2) >= min_box_diag_rescaled**2
        idx_to_keep *= box["w"] >= min_box_diag_rescaled / 3
        idx_to_keep *= box["h"] >= min_box_diag_rescaled / 3
        idx_to_keep *= box["w"] <= 0.8 * width_dst
        boxes[i] = box[idx_to_keep]

    return boxes, frame_is_labeled


def load_box_events(metadata, batch_start_time, duration):
    """Fetches box events from FileMetadata object, batch_start_time & duration.

    Args:
        metadata (object): Record details.
        batch_start_time (int): (us) Where to seek in the file to load corresponding bounding boxes
        duration (int): (us) How long to load events from bounding box file

    Returns:
        box_events (structured np.ndarray): Nx1 of dtype EventBbox
    """
    ending = metadata.get_ending()
    box_path = '_bbox.npy'.join(metadata.path.rsplit(ending, 1))

    box_events = np.load(box_path)

    """
    Warning! DO NOT REMOVE THIS CAST YET
    This small cast is very important for backward
    compatibility with old datasets where EventBbox
    was slightly different.
    TODO: remove this when all datasets are exactly of
    type EventBbox (otherwise this will cause
    weird bugs when merging dtypes during validation)
    """
    out = np.zeros((len(box_events),), dtype=EventBbox)
    for k in box_events.dtype.names:
        if k == "confidence":
            out["class_confidence"] = box_events["confidence"]
        elif k == "ts":
            out["t"] = box_events["ts"]
        else:
            out[k] = box_events[k]

    out = np.sort(out, order=["t"])
    out = out[(out['t'] >= batch_start_time) & (out['t'] < batch_start_time+duration)]
    return out


def nms_by_class(box_events, scores, iou_thresh=0.5):
    """NMS on box_events done independently by class

    Args:
        box_events (np.ndarray): nx1 with dtype EventBbox , the sorting order of those box is used as a
            a criterion for the nms.
        scores (np.ndarray): nx1 dtype of plain dtype, needs to be argsortable.
        iou_thresh (float): if two boxes overlap with more than `iou_thresh` (intersection over union threshold)
            with each other, only the one with the highest criterion value is kept.

    Returns:
        keeps (np.ndarray): Indices of the box to keep in the input array.
    """
    classes = np.unique(box_events['class_id'])
    idx = np.arange(len(box_events))
    keeps = []
    for class_id in classes:
        mask = box_events['class_id'] == class_id
        keep = nms(box_events[mask], scores[mask])
        keeps += idx[mask][keep].tolist()
    keeps = sorted(keeps)
    return keeps


def nms(box_events, scores, iou_thresh=0.5):
    """NMS on box_events

    Args:
        box_events (np.ndarray): nx1 with dtype EventBbox, the sorting order of those box is used as a
            a criterion for the nms.
        scores (np.ndarray): nx1 dtype of plain dtype, needs to be argsortable.
        iou_thresh (float): if two boxes overlap with more than `iou_thresh` (intersection over union threshold)
            with each other, only the one with the highest criterion value is kept.

    Returns:
        keep (np.ndarray): Indices of the box to keep in the input array.
    """
    x1 = box_events['x']
    y1 = box_events['y']
    x2 = box_events['x'] + box_events['w']
    y2 = box_events['y'] + box_events['h']

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return sorted(keep)


def clip_boxes(box_events, width_orig, height_orig):
    """Clips boxes so that they belong to the viewport width and height.
    Discards those that ends up being empty.

    Args:
        box_events (structured np.ndarray): Nx1 of dtype EventBbox
        width_orig (int): Original width of sensor for annotation
        height_orig (int): Original height of sensor for annotation

    Returns:
        box_events (structured np.ndarray): Nx1 of dtype EventBbox
    """
    x, y = "x", "y"

    xmax = box_events[x] + box_events['w']
    ymax = box_events[y] + box_events['h']
    box_events[x] = np.clip(box_events[x], 0, width_orig)
    box_events[y] = np.clip(box_events[y], 0, height_orig)
    box_events["w"] = np.clip(xmax, 0, width_orig) - box_events[x]
    box_events["h"] = np.clip(ymax, 0, height_orig) - box_events[y]

    return box_events[(box_events["w"] > 2) * (box_events['h'] > 2)]


def rescale_boxes(box_events, width_orig, height_orig, width_dst, height_dst):
    """Rescales boxes to new height and width.

    Args:
        box_events (structured np.ndarray): Array of length n of dtype EventBbox.
        width_orig (int): Original width of sensor for annotation.
        height_orig (int): Original height of sensor for annotation.
        width_dst (int): Destination width.
        height_dst (int): Destination height.

    Returns:
        box_events (structured np.ndarray): Array of length n of dtype EventBbox.
    """
    w_ratio = float(width_dst) / width_orig
    h_ratio = float(height_dst) / height_orig
    box_events['w'] *= w_ratio
    box_events["h"] *= h_ratio
    box_events['x'] *= w_ratio
    box_events["y"] *= h_ratio
    return box_events


def filter_boxes(box_events, class_lookup, idx_to_filter, ignore_filtered):
    """Filters or ignores boxes or in box_events according to idx_to_filter.

    Ignored boxes are still present but are marked with a -1 class_id. At the loss computation stage this
    information can be used so that they don't contribute to the loss.This ius used when you don't want
    the proposals matched with those *ignored boxes* to be considered as False positives in the loss.
    For instance if you train on cars only in a dataset containing trucks they could be ignored.

    Args:
        box_events (np.ndarray): Box events.
        class_lookup (int list): Lookup table for converting class indices to contiguous int values.
        idx_to_filter (np.ndarray): Boxes indices to filter out or ignore (see below).
        ignore_filtered (bool): If true, ignores the boxes filtered in the loss those boxes are marked with
            a -1 *class_id* in order to discard them in a loss.

    Returns:
        (np.ndarray): Box_events with class_id translated using the class_lookup.
    """
    classes = class_lookup[box_events['class_id']]  # classes not used will be -1
    assert (classes != 0).all(), "classes cannot be zero: " + str(classes)
    to_keep = (classes > 0)
    if len(idx_to_filter) > 0:
        if ignore_filtered:
            classes[idx_to_filter] = -1  # considered as ignored in the loss
        else:  # removing boxes (considered as background)
            to_keep[idx_to_filter] = False

    box_events = box_events[to_keep]  # filter classes that are -1 (considered as background)
    box_events['class_id'] = classes[to_keep]
    return box_events


def split_boxes(box_events, batch_start_time, delta_t=None, num_tbins=None):
    """Split box_events to a list of box events clustered by delta_t
    Removes a bounding box from the input list box_events if:
    there are less than min_box_area_thr*bbox_area events in the box
    and timestamp of bbox < last_time_to_filter"

    Box times are in range(0, num_tbins*tbin)

    Args:
        box_events (structured np.ndarray): Box events inputs of type EventBbox
        delta_t (optional int): Duration of time bin in us. Used for chronological NMS.
        num_tbins (optional int): Number of time bins.

    Returns:
        box_events (np.ndarray list): List of box_events of type EventBbox separated in time bins.
    """
    time_field = box_events.dtype.names[0]

    masks = [np.where((box_events[time_field] > t) * (box_events[time_field] <= delta_t + t))[0]
             for t in range(batch_start_time, batch_start_time + delta_t * num_tbins, delta_t)]

    masks = [mask[nms(box_events[mask], box_events[mask]['t'], iou_thresh=0.5)] for mask in masks]

    boxes = [box_events[mask] for mask in masks]
    return boxes


def create_class_lookup(labelmap_path, wanted_keys=[]):
    """
    Takes as argument a json path storing a dictionary with class_id as key and class_name as value for the ground
    truth. Takes also as argument a list of wanted keys (class_names that we want to select).

    Args:
        labelmap_path (string): Path to the label map ex of inside the json : '{"0": "pedestrian",
                                                                          "1": "two wheeler",
                                                                          "2": "car",
                                                                          "3": "truck"
                                                                         }'
        wanted_keys (list): List of classes to extract example: ['car', 'pedestrian']

    Returns:
        class_lookup numpy array [1, -1, 2, -1]

    In the example we get 0 for background, 1 for pedestrians and 2 for cars.
    At the end, if you do new_label = class_lookup[gt_label] you can transform ground truth ids array in an array with
    ids that fit your network. Reminder : Ground truth does not have id for background. For our network we get id 0 for
    background and consecutive ids for other classes.
    """
    with open(labelmap_path, "r") as read_file:
        # label_dic is the original dataset dictionary (id -> class name)
        label_dic = json.load(read_file)
        label_dic = {int(str(key)): str(value) for key, value in label_dic.items()}

        # we take maximum class id + 1 because of class id 0
        size = max(label_dic.keys()) + 1

        # check that all wanted classes are inside the dataset
        classes = label_dic.values()
        if wanted_keys:
            assert any(item != 'empty' for item in wanted_keys)
            for key in wanted_keys:
                assert key in classes, "key '{}' not found in the dataset".format(key)
        else:
            # we filter out 'empty' because this is used to annotate empty frames
            wanted_keys = [label for label in classes if label != 'empty']

    wanted_map = {label: idx for idx, label in enumerate(wanted_keys)}
    class_lookup = np.full(size, -1)
    for src_idx in range(size):
        if src_idx not in label_dic:
            continue
        src_label = label_dic[src_idx]
        if src_label not in wanted_keys:
            continue
        class_lookup[src_idx] = wanted_map[src_label] + 1
    return class_lookup


def bboxes_to_box_vectors(bbox):
    """Converts back EventBbox bounding boxes
    to plain numpy array.

    Args:
        bbox: np.ndarray Nx1 dtype EventBbox (x1,y1,w,h,score,conf,track_id)

    WARNING: Here class id must be in 0-C (-1: ignore, 0: background, [1,C]: classes)

    Returns:
        out: torch.array Nx6 dtype (x1,y1,x2,y2,label,track_id)
    """
    out = np.zeros((len(bbox), 6), dtype=np.float32)
    bbox = {key: np.float32(bbox[key].copy()) for key in bbox.dtype.names}
    out[:, 0] = bbox['x']
    out[:, 1] = bbox['y']
    out[:, 2] = bbox['x'] + bbox['w']
    out[:, 3] = bbox['y'] + bbox['h']
    out[:, 4] = bbox['class_id']
    out[:, 5] = bbox['track_id']
    return out


def box_vectors_to_bboxes(boxes, labels, scores=None, track_ids=None, ts=0):
    """Concatenates box vectors into a structured array of EventBbox.

    Args:
        boxes (np.ndarray): Bboxes coordinates (x1,y2,x2,y2).
        labels (np.ndarray): Class index for each box.
        scores (np.ndarray): Score for each box.
        track_ids (np.ndarray): Individual track id for each box.
        ts (int): Timestamp in us.

    Returns:
        box_events (np.ndarray): Box with EventBbox.
    """
    box_events = np.zeros((len(boxes),), dtype=EventBbox)
    if scores is None:
        scores = np.zeros((len(boxes),), dtype=np.float32)
    if track_ids is None:
        track_ids = np.arange(len(boxes), dtype=np.uint32)

    box_events['t'] = ts
    box_events['x'] = boxes[:, 0]
    box_events['y'] = boxes[:, 1]
    box_events['w'] = boxes[:, 2] - boxes[:, 0]
    box_events['h'] = boxes[:, 3] - boxes[:, 1]
    box_events['class_confidence'] = scores
    box_events['class_id'] = labels
    box_events['track_id'] = track_ids
    return box_events


def could_frame_contain_valid_gt(batch_start_time, duration, labelling_delta_t, num_tbins):
    """This function returns a np.array of num_tbins boolean,
    indicating whether a frame was labeled or not.

    This is useful if our recordings are labeled at a fix frame rate but we want to train at a higher
    framerate (i.e. small delta_t.)
    The number of frames in a batch (num_tbins) is the duration of this batch divided by delta_t

    Note: If you train at faster frequency than your annotations it is also possible to interpolate
    your bounding box files offline to avoid this.

    For example, given the following setup
       - num_tbins = 5  (number of frames in a batch)
       - delta_t = 50   (time of each frame)
       - labelling_delta_t = 120  (delta_t at which labels are provided)
       - duration = batch_size * delta_t = 250

    -> this function will be called several times,
    with batch_start_time = 0, then 250, then 500, etc.
    Each time this function is called, it returns an array of 5 booleans to indicate
    which frames could contain a label::

                  GT            GT              GT            GT            GT            GT
       |            120           240|            360           480|          600           720  |
       |             |             | |             |             | |           |             |   |
       |             v             v |             v             v |           v             v   |
       |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
       0    50    100   150   200   250   300   350   400   450   500   550   600   650   700   750
       |                             |                             |                             |
       |< F > < F > < T > < F > < T >|< F > < F > < T > < F > < T >|< F > < T > < F > < F > < T >|
       |                             |                             |                             |
       |<-------- first call ------->|<------- second call ------->|<-------- third call ------->|
       |                             |                             |                             |



    Same setup as before, but now with labelling_delta_t = 100 instead of 120::

                GT          GT          GT          GT          GT           GT         GT
       |          100         200    |    300         400         500          600        700    |
       |           |           |     |     |           |           |           |           |     |
       |           v           v     |     v           v           v           v           v     |
       |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
       0    50    100   150   200   250   300   350   400   450   500   550   600   650   700   750
       |                             |                             |                             |
       |< F > < T > < F > < T > < F >|< T > < F > < T > < F > < T >|< F > < T > < F > < T > < F >|
       |                             |                             |                             |
       |<-------- first call ------->|<------- second call ------->|<-------- third call ------->|
       |                             |                             |                             |


    Note: if labelling_delta_t <= delta_t, all frames could contain a valid GT

    Note: If the FileMetadata is a pure distractor file (with no label at all), it will have a 1 us labelling_delta_t,
    and therefore all the frames will be considered labeled.

    Args:
        batch_start_time: Time from when to start loading (in us).
        duration: Duration to load (in us).
        labelling_delta_t: Period (in us) of your labelling annotation system.
        num_tbins: Number of frames to load.
    Returns:
        frame_could_contain_gt: (boolean nd array): This boolean mask array of *length* num_tbins indicates
            whether the frame contains a label. It is used to differentiate between time_bins that actually
            contain an empty label (for instance no bounding boxes) from time bins that weren't labeled due
            to cost constraints. The latter timebins shouldn't contribute to supervised losses used during
            training.
    """
    assert labelling_delta_t > 0
    assert duration % num_tbins == 0
    delta_t = duration / num_tbins
    batch_end_time = batch_start_time + duration + 1
    first_expected_label_ts = (batch_start_time // labelling_delta_t + 1) * labelling_delta_t

    expected_label_ts = np.arange(first_expected_label_ts, batch_end_time, labelling_delta_t)

    end_ts = np.arange(batch_start_time + delta_t, batch_end_time, delta_t)

    contain_valid_gt = np.zeros(num_tbins, dtype=np.bool)
    indexes_gt = np.unique(np.searchsorted(end_ts, expected_label_ts, side="left"))

    contain_valid_gt[indexes_gt] = True

    return contain_valid_gt


def filter_empty_tensor(
        array: np.array,
        box_events: np.array,
        area_box_filter: float = 0.1,
        shift: int = 0,
        time_per_bin: int = 10000,
        batch_start_time: int = 0,
        last_time_to_filter: int = None) -> np.array:
    """
    Preprocessing bounding boxes: discard bbox with empty event data inside of it

    Args:
        array: (T,C,H,W) event frame
        box_events:numpy array of bbox
        area_box_filter: minimum percentage area of bbox which contain events
        shift: downsampling coefficient
        time_per_bin: time interval per time bin along T axis
        batch_start_time: starting time stamp of the array batch
        last_time_to_filter: stop filtering bbox after this time stamp

    Returns:

    """
    idx_to_filter = []
    MAX_TIME_BEFORE = 100000  # max time before the current bbox
    # transfer the boxes coordinates in the array index system
    ts_a = box_events['t'].astype(np.float32) - batch_start_time
    xmin_a = box_events['x'].astype(np.int16) >> shift
    xmax_a = (box_events['x'] + box_events['w']).astype(np.int16) >> shift
    ymin_a = box_events['y'].astype(np.int16) >> shift
    ymax_a = (box_events['y'] + box_events['h']).astype(np.int16) >> shift
    for i, (ts, xmin, xmax, ymin, ymax) in enumerate(zip(ts_a, xmin_a, xmax_a,
                                                         ymin_a, ymax_a)):
        if last_time_to_filter is not None and ts > last_time_to_filter:
            continue
        tmax = min(int(ts / time_per_bin) + 1, array.shape[0])
        min_t = max(0, tmax - int(MAX_TIME_BEFORE / time_per_bin))
        ymin = max(0, ymin)
        xmin = max(0, xmin)
        if ymax <= ymin or xmax <= xmin:  # skipping zero-area boxes
            idx_to_filter.append(i)
            continue
        # we only filter out bbox when there are few events inside the bbox within the selected time interval
        # by default we filter out bbox with less than 10% of the area filled with events
        box_slice = np.abs(array[min_t:tmax, :, ymin:ymax, xmin:xmax])
        if (box_slice > 0).max(0).max(0).sum() < area_box_filter * box_slice.shape[-1] * box_slice.shape[-2]:
            idx_to_filter.append(i)
    idx_to_filter = np.array(idx_to_filter)
    return idx_to_filter
