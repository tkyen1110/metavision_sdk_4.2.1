# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Generates frames from boxes and tracks
"""
import cv2

import metavision_sdk_core
import metavision_sdk_ml

import numpy as np

COLORS = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)[:, 0]


def draw_box_events(frame, box_events, label_map, force_color=None, draw_score=True, thickness=1,
                    color_from="class_id", confidence_field="class_confidence",
                    dic_history={}):
    """Draws boxes on a RGB image.

    Args:
        frame (np.ndarray): H, W, 3 image of dtype uint8
        box_events (np.ndarray): Box_events in EventBbox format.
        label_map (list): List of class names indexed by class_id.
        force_color (int list): If not None, overrides the color from `color_field` and is put as the color
        of all boxes.
        draw_score (bool): Whether to add confidence to the boxes.
        thickness (int): Rectangle line thickness.
        color_from (string): Field of the EventBbox used to choose the box color, defaults to `class_id`.
        confidence_field (string): Field of the EventBbox used to read confidence score.
        dic_history (dict): Dictionary where keys are track_id and values are np.arrays of boxes
    Returns:
        img: Drawn image.
    """

    def choose_color(box_events, color_field, force_color=None):
        """
        Choose a colors for a array of boxes depending of an integer field.

        Args:
            box_events (structured np.ndarray): Box events inputs of type EventBbox
            color_field (string): Name of the field used to choose the color
                (could be `track_id` or `class_id` for instance)
            force_color (int list): If not None, overrides the color from `color_field` and is put as the color
                of all boxes.

        Returns:
            np.array of size len(box_events) x 3
        """
        if force_color is not None:
            assert len(force_color) == 3
            return np.array([force_color for _ in box_events], dtype=np.uint8)
        else:
            assert np.issubdtype(box_events[color_field].dtype,
                                 np.integer), 'color_field {:s} should be integer'.format(color_field)
            assert color_field in box_events.dtype.names, 'color_field should be a field of box_events dtype'
            return COLORS[box_events[color_field] * 60 % len(COLORS)]

    height, width = frame.shape[:2]
    if len(box_events) == 0:
        return frame

    assert confidence_field in box_events.dtype.names, 'wrong confidence field in dtype: {}'.format(
        confidence_field)

    topleft_x = np.clip(box_events["x"], 0, width - 1).astype('int')
    topleft_y = np.clip(box_events["y"], 0, height - 1).astype('int')
    botright_x = np.clip(box_events["x"] + box_events["w"], 0, width - 1).astype('int')
    botright_y = np.clip(box_events["y"] + box_events["h"], 0, height - 1).astype('int')

    colors = choose_color(box_events, color_from, force_color=force_color)

    for i, (tlx, tly, brx, bry) in enumerate(zip(topleft_x, topleft_y, botright_x, botright_y)):
        color = colors[i].tolist()
        cv2.rectangle(frame, (tlx, tly), (brx, bry), color, thickness)
        text = label_map[box_events[i]["class_id"]]
        if draw_score:
            text += " {:.2f}".format(box_events[i][confidence_field])
        cv2.putText(frame, text, (int(tlx + 0.05 * (brx - tlx)), int(tly + 0.94 * (-tly + bry))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
        if color_from == "track_id":
            track_id = box_events[color_from][i]
            tracks_np = dic_history[track_id]
            center_x, center_y = int((tlx + brx) / 2), int((tly + bry) / 2)
            for box_idx in range(tracks_np.size - 1, -1, -1):
                prev_box = tracks_np[box_idx]
                center_x_prev = int(prev_box["x"] + prev_box["w"] / 2)
                center_y_prev = int(prev_box["y"] + prev_box["h"] / 2)
                cv2.line(frame, (center_x, center_y), (center_x_prev, center_y_prev), color, thickness=4)
                center_x, center_y = center_x_prev, center_y_prev

    return frame


def draw_detections_and_tracklets(ts, frame, width, height, detections, tracklets,
                                  label_map={0: "background", 1: "pedestrian", 2: "two wheeler", 3: "car"},
                                  list_previous_tracks=[]):
    """Draws a visualization for both detected boxes and tracked boxes side by side.

    Detections are on the left pane, with box colors indicating class membership. Tracks are drawn on the right pane
    with colors denoting track ids.

    Args:
        ts (int): Current timestamp in us. (Used for display only).
        frame (np.ndarray): Array of dtype np.uint8 and of shape `height` x `width * 2` x 3
        width (int): Width of the imager in pixels.
        height (int): Height of the imager in pixels.
        detections (nd.array): Array of EventBbox to be drawn.
        tracklets (ndarray): Array of EventTrackedBox to be drawn.
        label_map (dict): Dictionary mapping class ids to the name of the corresponding class
        list_previous_tracks (list): list of np.arrays of tracks (one np.array for each previous timestep)
    """
    assert frame.shape[:2] == (height, width * 2)
    assert detections.dtype == metavision_sdk_core.EventBbox
    assert tracklets.dtype == metavision_sdk_ml.EventTrackedBox

    # add boxes in image frame (detections)
    draw_box_events(frame[:, :width], detections, label_map, thickness=3,
                    color_from="class_id", confidence_field="class_confidence")
    # add boxes in image frame (tracks)
    if list_previous_tracks == []:
        list_previous_tracks = [tracklets]
    previous_tracks_np = np.concatenate(list_previous_tracks)
    dic_trackid_history = {}
    for i in range(tracklets.size):
        track_id = tracklets[i]["track_id"]
        assert track_id not in dic_trackid_history
        dic_trackid_history[track_id] = previous_tracks_np[previous_tracks_np["track_id"] == track_id]

    draw_box_events(frame[:, width:], tracklets, label_map, thickness=3,
                    color_from='track_id', confidence_field="tracking_confidence", dic_history=dic_trackid_history)
    # add time information on top left of the image
    cv2.putText(frame, "{:02d}:{:02d}.{:d}".format(ts // 1000000 // 60,
                                                   ts // 1000000 % 60,
                                                   ts % 1000000),
                (int(0.05 * (width)), 40),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 240, 12))
