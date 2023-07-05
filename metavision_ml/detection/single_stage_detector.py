# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Detector Interface
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import metavision_ml.detection.feature_extractors as feat
import metavision_ml.detection.rpn as rpn_module

from metavision_ml.detection.losses import DetectionLoss
from metavision_ml.detection.anchors import Anchors

from itertools import chain


class Detector(nn.Module):
    """
    This in an interface for neural network learning to predict boxes.
    The trainer expects "compute_loss" and "get_boxes" to be coded.
    """

    def __init__(self):
        super(Detector, self).__init__()

    def get_boxes(self, x, score_thresh=0.4):
        raise NotImplementedError()

    def compute_loss(self, x, targets):
        raise NotImplementedError()


class SingleStageDetector(Detector):
    """ This in an interface for "single stage" methods (e.g: RetinaNet, SSD, etc.)

    Args:
        feature_extractor (string): name of the feature extractor architecture
        in_channels (int): number of channels for the input layer
        num_classes (int): number of output classes for the classifier head
        feature_base(int): factor to grow the feature extractor width
        feature_channels_out(int): number of output channels for the feature extractor
        anchor_list (couple list): list of couple (aspect ratio, scale) to be used for each extracted feature
        max_boxes_per_input (int): max number of boxes to be considered before thresholding or NMS.
    """

    def __init__(self, feature_extractor,
                 in_channels,
                 num_classes,
                 feature_base,
                 feature_channels_out,
                 anchor_list,
                 nlayers=0, max_boxes_per_input=500):
        super(SingleStageDetector, self).__init__()
        feature_extractor = getattr(feat, feature_extractor)

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.feature_extractor = feature_extractor(in_channels, base=feature_base, cout=feature_channels_out)
        self.box_coder = Anchors(num_levels=self.feature_extractor.levels,
                                 anchor_list=anchor_list,
                                 variances=[0.1, 0.2])

        self.num_anchors = self.box_coder.num_anchors
        self.max_boxes_per_input = max_boxes_per_input

        self.rpn = rpn_module.BoxHead(self.feature_extractor.cout, self.box_coder.num_anchors,
                                      self.num_classes + 1, nlayers)

        self.criterion = DetectionLoss("softmax_focal_loss")

    def forward(self, x):
        xs = self.feature_extractor(x)
        return self.rpn(xs)

    def select_valid_frames(self, xs, targets, frame_is_labeled):
        frame_is_labeled = frame_is_labeled.bool()
        mask = frame_is_labeled.view(-1)
        xs = [item[mask] for item in xs]
        targets = [
            [targets[r][c] for c in range(len(frame_is_labeled[0])) if frame_is_labeled[r][c]]
            for r in range(len(frame_is_labeled))]
        return xs, targets

    def compute_loss(self, x, targets, frame_is_labeled):
        xs = self.feature_extractor(x)

        if frame_is_labeled.sum().item() == 0:
            return None

        xs, targets = self.select_valid_frames(xs, targets, frame_is_labeled)

        loc_preds, cls_preds = self.rpn(xs)

        targets = list(chain.from_iterable(targets))
        targets = self.box_coder.encode(xs, x, targets)

        assert targets['cls'].shape[1] == cls_preds.shape[1]

        loss_dict = self.criterion(loc_preds, targets['loc'], cls_preds, targets["cls"])
        return loss_dict

    def get_boxes(self, x, score_thresh=0.4, nms_thresh=0.6):
        features = self.feature_extractor(x)
        loc_preds, cls_preds = self.rpn(features)
        scores = self.rpn.get_scores(cls_preds)
        return self.box_coder.decode(features, x, loc_preds, scores, batch_size=x.shape[1], score_thresh=score_thresh,
                                     nms_thresh=nms_thresh, max_boxes_per_input=self.max_boxes_per_input)

    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        for name, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset(mask)

    def reset_all(self):
        for name, module in self._modules.items():
            if hasattr(module, "reset_all"):
                module.reset_all()

    def freeze_feature_extractor(self, is_frozen):
        for param in self.feature_extractor.parameters():
            param.requires_grad = not is_frozen

    def map_cls_weights(self, module_src, old_classes, new_classes):
        """
        Maps old classes to new classes
        if some overlap exists between old classes and new ones.

        Args:
            module_src: old model
            old_classes: old list of classes
            new_classes: new list of classes
        """
        self.rpn.init_cls_head()
        old_classes = ['background'] + old_classes
        new_classes = ['background'] + new_classes
        old_map = {label: i for i, label in enumerate(old_classes)}
        src_weight, src_bias = module_src.rpn.get_classification_params()
        dst_weight, dst_bias = self.rpn.get_classification_params()
        for dst_idx, dst_label in enumerate(new_classes):
            if dst_label in old_map:
                print(dst_label, 'was in old set: transfer weights...')
                src_idx = old_map[dst_label]
                dst_weight.data[:, dst_idx] = src_weight[:, src_idx]
                dst_bias.data[:, dst_idx] = src_bias[:, src_idx]
