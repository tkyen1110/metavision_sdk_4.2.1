# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This contains the classes to encode/ decode the gt into 'anchor-boxes'.

All targets are encoded in parallel for better throughput.

The module can resize its grid internally (so batches can change sizes)
"""
import sys
import time
import torch
import torch.nn as nn
import numpy as np

from typing import List

from metavision_ml.detection import box
from torchvision.ops.boxes import batched_nms


MNIST_ANCHORS = [(1, 1), (1, 1.5)]
COCO_ANCHORS = [(ratio, scale) for ratio in [0.5, 1.0, 2.0] for scale in [1.0, 2**1. / 3, 2**2. / 3]]
PSEE_ANCHORS = [(1. / 3, 1), (0.5, 1), (1, 1), (1, 1.5), (2, 1), (3, 1)]


class AnchorLayer(nn.Module):
    """
    For one level of the pyramid: Manages One Grid (x,y,w,h)

    The anchors grid is (height, width, num_anchors_per_position, 4)

    The grid is cached, but changes if featuremap size changes

    Args:
        box_size (int): base size for anchor box
        anchor_list (List): a list of ratio, scale tuples configuration of anchors
    """

    def __init__(self, box_size=32, anchor_list=PSEE_ANCHORS):
        super(AnchorLayer, self).__init__()
        self.num_anchors = len(anchor_list)
        self.stride = 0
        self.register_buffer("box_sizes", AnchorLayer.generate_anchors(box_size, anchor_list), persistent=True)
        self.register_buffer("anchors", torch.zeros(
            (0, 0, self.num_anchors, 4), dtype=torch.float32), persistent=False)

    @staticmethod
    def generate_anchors(box_size, ratio_scale_list):
        """
        Generates the anchors sizes

        Args:
            box_size (int): base size for anchor boxes
            ratio_scale_list (List): a list of ratio, scale tuples configuration of anchors

        Returns:
            anchors (torch.Tensor): Nx2 (width, height)
        """
        anchors = np.zeros((len(ratio_scale_list), 2), dtype=np.float32)
        for i, (ratio, scale) in enumerate(ratio_scale_list):
            anchors[i, 0] = scale * box_size * np.sqrt(ratio)
            anchors[i, 1] = scale * box_size / np.sqrt(ratio)

        ratios = anchors[:, 0] / anchors[:, 1]
        scales = np.sqrt((anchors[:, 0] * anchors[:, 1]) / (box_size * box_size))
        return torch.from_numpy(anchors).float()

    def make_grid(self, height: int, width: int, stride: int):
        """
        Makes a coordinate grid for anchor boxes

        Args:
            height (int): feature-map height
            width (int): feature-map width
        Returns:
            grid (torch.Tensor): (H,W,NanchorPerPixel,2) size (width & height per anchor per pixel)
        """

        grid_h, grid_w = torch.meshgrid([torch.linspace(0.5 * stride, (height - 1 + 0.5) * stride, height),
                                         torch.linspace(0.5 * stride, (width - 1 + 0.5) * stride, width)
                                         ])
        grid = torch.cat([grid_w[..., None], grid_h[..., None]], dim=-1)

        grid = grid[:, :, None, :].expand(height, width, self.num_anchors, 2)
        return grid

    def forward(self, shape: torch.Tensor, stride: int):
        """Generates anchors

        Args:
            shape (torch.Tensor): shape of feature map
            stride (int): stride compared to network input
        """
        device = self.anchors.device
        height, width = int(shape[-2].item()), int(shape[-1].item())
        has_changed = False
        if self.anchors.shape[0] != height or self.anchors.shape[1] != width or stride != self.stride:
            has_changed = True
            self.stride = stride
            grid = self.make_grid(height, width, stride).to(device)
            wh = torch.zeros((self.num_anchors * 2, height, width), dtype=torch.float32, device=device) + \
                self.box_sizes.view(self.num_anchors * 2, 1, 1)
            wh = wh.permute([1, 2, 0]).view(height, width, self.num_anchors, 2)
            self.anchors = torch.cat([grid, wh], dim=-1)

        return self.anchors.view(-1, 4)


class Anchors(nn.Module):
    """
    Pyramid of Anchoring Grids.
    Handle encoding/decoding algorithms.
    Encoding uses padding in order to parallelize iou & assignment computation.
    Decoding uses "batched_nms" of torchvision to parallelize across images and classes.
    The option "max_decode" means we only decode the best score, otherwise we decode per class


    Args:
        num_levels: number of pyramid levels
        base_size: minimum box size
        sizes: box sizes per level
        anchor_list: list of anchors sizes per pyramid level
        fg/bg_iou_threshold: threshold to accept/ reject a matching anchor
        allow_low_quality_matches: assign all gt even if no anchor really matches
        variances: box variance following ssd formula
    """

    def __init__(self,  num_levels=4,
                 base_size=32,
                 anchor_list='PSEE_ANCHORS',
                 fg_iou_threshold=0.5,
                 bg_iou_threshold=0.3,
                 allow_low_quality_matches=True,
                 variances=[0.1, 0.2],
                 max_decode=False):
        super(Anchors, self).__init__()
        self.num_levels = num_levels
        self.base_size = base_size
        self.sizes = [self.base_size * 2 ** x for x in range(self.num_levels)]
        self.anchor_list = getattr(sys.modules[__name__], anchor_list)
        self.fg_iou_threshold = fg_iou_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.num_anchors = len(self.anchor_list)
        self.allow_low_quality_matches = allow_low_quality_matches
        self.max_decode = max_decode
        self.variances = variances
        self.anchor_generators = nn.ModuleList()
        for i, box_size in enumerate(self.sizes):
            self.anchor_generators.append(AnchorLayer(box_size, self.anchor_list))

        self.register_buffer("anchors", torch.zeros((0, 4), dtype=torch.float32), persistent=False)
        self.register_buffer("anchors_xyxy", torch.zeros((0, 4), dtype=torch.float32), persistent=False)
        self.register_buffer("last_shapes", torch.zeros((self.num_levels, 2), dtype=torch.int), persistent=False)

    def encode(self, features, x, targets):
        """
        Encodes input and features into target vectors
        expressed in anchor coordinate system.

        Args:
            x (torch.Tensor): input with original size
            targets (List): list of list of targets
        Returns:
            loc (torch.Tensor): encoded anchors regression targets
            cls (torch.Tensor): encoded anchors classification targets
        """
        anchors = self(features, x)
        return self.encode_anchors(self.anchors, self.anchors_xyxy, targets)

    def encode_anchors(self, anchors, anchors_xyxy, targets):
        """
        Encodes targets into target vectors
        expressed in anchor coordinate system.

        Args:
            anchors (torch.Tensor): anchors in cx,cy,w,h format
            anchors (torch.Tensor): anchors in x1,y1,x2,y2 format
        Returns:
            loc (torch.Tensor): encoded anchors regression targets
            cls (torch.Tensor): encoded anchors classification targets
        """
        gt_boxes, gt_labels, sizes = self._pad_boxes(targets, anchors.device)

        # this is computing best gt per anchor for every images in the batch
        best_target_per_prior_indices, mask_bg, mask_ign = self._iou_assignement(
            gt_boxes, gt_labels, anchors_xyxy, sizes)

        index = best_target_per_prior_indices[..., None].expand(len(gt_boxes), len(anchors), 4)
        boxes = torch.gather(gt_boxes, 1, index)
        loc_targets = box.bbox_to_deltas(boxes, anchors[None], self.variances)

        labels = torch.gather(gt_labels, 1, best_target_per_prior_indices)
        labels[mask_ign] = -1
        labels[mask_bg] = 0
        cls_targets = labels.view(-1, len(anchors))

        return {"loc": loc_targets, "cls": cls_targets}

    def decode(
            self, features, x, loc_preds, scores, batch_size, score_thresh=0.5, nms_thresh=0.6,
            max_boxes_per_input=500):
        """Decodes prediction vectors into boxes

        Args:
            features (list): list of feature maps
            x (torch.Tensor): network's input
            loc_preds (torch.Tensor): regression prediction vector (N,Nanchors,4)
            scores (torch.Tensor): score prediction vector (N,Nanchors,C) with C classes (background is
            excluded)
            score_thresh (float): apply this threshold before nms
            nms_thresh (float): grouping threshold on IOU similarity between boxes
            max_boxes_per_input (int): maximum number of boxes per image. Too small might reduce recall, too high might
                entail extra computational cost in decoding, loss or evaluation.
        Returns:
            decoded boxes (List): list of list of decoded boxes
        """
        anchors = self(features, x)

        # loc_preds [N, C] (do not include background column)
        box_preds = box.deltas_to_bbox(loc_preds, anchors, self.variances)

        # Decoding
        decoded_dict = self.batched_decode(box_preds, scores, score_thresh, nms_thresh, topk=max_boxes_per_input)

        num_tbins = len(scores) // batch_size
        return self._expand_decoded_boxes(decoded_dict, batch_size, num_tbins)

    def has_changed(self, xs: List[torch.Tensor], x: torch.Tensor):
        """Detects if feature maps sizes has change.

        Args:
            xs (List): list of feature maps
            x (torch.Tensor): network's input
        """
        shapes = torch.tensor([item.size()[-2:] for item in xs]).int().to(self.last_shapes)
        any_diff = ~torch.all(torch.eq(shapes, self.last_shapes))
        return any_diff.item()

    def forward(self, xs: List[torch.Tensor], x: torch.Tensor):
        """Generates Anchors

        Args:
            xs (List): list of feature maps
            x (torch.Tensor): network's input

        Returns:
            anchors (torch.Tensor): (N,Nanchors,4) anchor boxes in (cx,cy,w,h) format
        """
        # infer shapes and call _forward
        shapes = torch.tensor([item.size()[-2:] for item in xs]).int()
        imsize = x.size()[-2:]
        strides = torch.tensor([int(imsize[-1] // shape[-1]) for shape in shapes])

        device = self.anchors.device
        shapes = shapes.to(device)
        strides = strides.to(device)

        return self._forward(shapes, strides)

    def _forward(self, shapes: torch.Tensor, strides: torch.Tensor):
        """Generates anchors if shapes have changed
        Will do nothing if shapes have not.

        Args:
            shapes (torch.Tensor): current shapes of feature maps
            strides (List): list of strides compared to input size.

        Returns:
            anchors (torch.Tensor): N,Nanchors,4 in (cx,cy,w,h) format
        """
        if not torch.eq(shapes, self.last_shapes).any().item():
            default_boxes = []
            for i, anchor_layer in enumerate(self.anchor_generators):
                shape = shapes[i]
                stride = strides[i]
                anchors = anchor_layer(shape, stride)
                default_boxes.append(anchors)
            self.anchors = torch.cat(default_boxes, dim=0)
            self.anchors_xyxy = box.xywh2xyxy(self.anchors)
        self.last_shapes = shapes
        return self.anchors

    def _pad_boxes(self, targets, device):
        """put the input targets in form of tensor padded with -2

        Args:
            targets (list): list of size num_tbins of list
                of size batch_size of Tensors (number of boxes x (5/6)
            device (torch.device): device where to put the resulting tensor (cpu/cuda) etc
        """
        gt_padded, num_boxes = box.pack_boxes_list(targets)

        gt_padded = gt_padded.to(device)
        gt_boxes = gt_padded[..., :4]
        gt_labels = gt_padded[..., 4].long()
        return gt_boxes, gt_labels, num_boxes

    def _iou_assignement(self, gt_boxes, gt_labels, anchors_xyxy, sizes):
        """matches GT boxes with anchors using IOU criteria

        Args:
            gt_boxes (Tensor): shape (batch len x 4) all gt boxes
            gt_labels (Tensor): shape (batch len) all class labels (-2 if box is padding)
            anchors_xyxy (Tensor): all anchor boxes in xyxy format
            sizes (int list): number of valid (non padding) boxes for each bin

        Returns:
            batch_best_target_per_prior_indices (torch.Tensor): gt assignement per anchor
            mask_bg (torch.Tensor): background boolean matrix
            mask_ign torch.Tensor): ignore boolean matrix
        """
        ious = box.batch_box_iou(anchors_xyxy, gt_boxes)  # [N, A, M]

        # make sure to not select the dummies
        mask = (gt_labels == -2).float().unsqueeze(1)
        ious = (-1 * mask) + ious * (1 - mask)

        # this is computing best gt per anchor for every images in the batch
        batch_best_target_per_prior, batch_best_target_per_prior_indices = ious.max(-1)  # [N, A]

        if self.allow_low_quality_matches:
            self.set_low_quality_matches(ious, batch_best_target_per_prior_indices,
                                         batch_best_target_per_prior, sizes)

        mask_bg = batch_best_target_per_prior < self.bg_iou_threshold
        mask_ign = (batch_best_target_per_prior > self.bg_iou_threshold) * \
            (batch_best_target_per_prior < self.fg_iou_threshold)

        return batch_best_target_per_prior_indices, mask_bg, mask_ign

    def set_low_quality_matches(self, ious, batch_best_target_per_prior_indices, batch_best_target_per_prior, sizes):
        """Makes sure that every GT is assigned to at least 1 anchor.

        Args:
            ious (torch.Tensor): (N,Nanchors,MaxGT) IOU cost matrix
            batch_best_target_per_prior_indices (torch.Tensor): (N,Nanchors)
            sizes (int list): number of valid (non padding) boxes for each bin
        """
        batch_best_prior_per_target, batch_best_prior_per_target_index = ious.max(-2)  # [N, M]

        # be here to not select dummies by looking where ious is not -1
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            (ious != -1) * (ious == batch_best_prior_per_target.unsqueeze(1)), as_tuple=False)
        batch_index = gt_pred_pairs_of_highest_quality[:, 0]
        pred_index = gt_pred_pairs_of_highest_quality[:, 1]
        gt_index = gt_pred_pairs_of_highest_quality[:, 2]
        batch_best_target_per_prior_indices[batch_index, pred_index] = gt_index
        # 2.0 is used to make sure every target has a prior assigned
        batch_best_target_per_prior[batch_index, pred_index] = 2.0

    def _expand_decoded_boxes(self, decoded_dict, batch_size, num_tbins):
        """expands the batch along the temporal dimension"""

        targets = [[{"boxes": None, "labels": None, "scores": None} for _ in range(batch_size)]
                   for _ in range(num_tbins)]

        bidx, sidx = decoded_dict.pop('batch_index').sort()
        bidx_vals, sizes = torch.unique(bidx.long(), return_counts=True)
        sidx_list = sidx.split(sizes.cpu().numpy().tolist())

        for bidx_val, group in zip(bidx_vals.cpu().numpy(), sidx_list):
            t, i = divmod(bidx_val, batch_size)

            targets[t][i] = {key: decoded_dict[key][group] for key in decoded_dict}

        return targets

    def batched_decode(self, boxes, scores, score_thresh, nms_thresh,
                       topk=int(1e2)):
        """Decodes all prediction vectors across batch, classes and time.

        Args:
            boxes (torch.Tensor): (N,Nanchors,4)
            scores (torch.Tensor): (N,Nanchors,C) with C classes excluding
            background.
            score_thresh (float): threshold to score
            nms_thresh (float): threshold to IOU.
            topk (int): maximum number of anchors considered for NMS.
        """
        batch_size = boxes.shape[0]
        num_anchors = boxes.shape[1]
        num_classes = scores.shape[-1]

        scores, idxs = self._box_indices(scores, batch_size, num_anchors, num_classes)
        if not self.max_decode:
            boxes = boxes.unsqueeze(2).expand(-1, num_anchors, num_classes, 4).contiguous()

        # filtering boxes by score
        scores_out, score_indices = self._score_filtering_per_image(scores, score_thresh, topk)

        boxes = boxes.view(-1, 4)[score_indices].contiguous()
        idxs = idxs.view(-1)[score_indices].contiguous()
        # apply NMS to boxes with sufficient score
        keep = batched_nms(boxes, scores_out, idxs, nms_thresh)

        idxs = idxs[keep]
        labels = idxs % num_classes + 1
        batch_index = idxs // num_classes
        res_dict = {"boxes": boxes[keep], "scores": scores_out[keep], "labels": labels, "batch_index": batch_index}
        return res_dict

    def _box_indices(self, scores, batch_size, num_anchors, num_classes):
        """Gives the boxes indices whether we give the best scoring boxes by classes or considered altogether

        Args:
            scores (torch.Tensor): (N,Nanchors,C) with C classes
            batch_size (int): original batch size
            num_anchors (int): total number of anchors per image.
            num_classes (int): number of classes
        Returns:
            scores (torch.Tensor): best scores per anchor
            idxs (torch.Tensor): indices of best scores
        """
        if self.max_decode:
            scores, idxs = scores.max(dim=-1)
            rows = torch.arange(batch_size, dtype=torch.long, device=scores.device)[:, None] * num_classes
            idxs += rows
        else:
            rows = torch.arange(batch_size, dtype=torch.long, device=scores.device)[:, None]
            cols = torch.arange(num_classes, dtype=torch.long, device=scores.device)[None, :]
            idxs = rows * num_classes + cols
            idxs = idxs.unsqueeze(1).expand(batch_size, num_anchors, num_classes).contiguous()
        return scores, idxs

    def _score_filtering_per_image(self, scores, score_threshold, topk_per_image):
        """Returns the sorted scores and their indices

        Top-K is performed per image

        Args:
            scores (Tensor): batch_size x num_anchors x num_classes
            score_theshold (float): minimal score to be retained
            topk_per_image (int): maximum items to be retained for each frame.
        Returns:
            scores (torch.Tensor): best scores per anchor
            idxs (torch.Tensor): indices of best scores
        """
        batch_size, num_anchors, num_classes = scores.shape
        topk_per_image = min(topk_per_image, num_anchors * num_classes)
        scores = scores.reshape(batch_size, num_anchors * num_classes)
        scores_filtered, scores_filtered_indices = torch.topk(scores, topk_per_image, dim=1)

        scores_filtered = scores_filtered.view(-1)

        offset = torch.arange(0, scores.numel(), num_anchors * num_classes)[:, None]
        scores_filtered_indices += offset.to(scores_filtered_indices)
        scores_filtered_indices = scores_filtered_indices.view(-1)

        index_filtered = scores_filtered >= score_threshold
        return scores_filtered[index_filtered], scores_filtered_indices[index_filtered]

    def _score_filtering(self, scores, score_threshold, topk):
        """Returns the sorted scores and their indices

        Args:
            scores (Tensor): batch_size x num_anchors x num_classes
            score_theshold (float): minimal score to be retained
            topk (int): maximum items to be retained per batch

        Returns:
            scores (torch.Tensor): best scores per anchor
            idxs (torch.Tensor): indices of best scores
        """
        scores = scores.view(-1)
        score_indices = torch.arange(scores.shape[0]).to(scores.device)
        mask = (scores >= score_threshold).to(scores.device)
        score_indices = score_indices[mask].contiguous()
        if mask.sum() > topk:
            # we only take the topk first boxes
            scores, idx = torch.sort(scores[score_indices], descending=True)
            scores = scores[:topk]
            score_indices = score_indices[idx[:topk]]  # indices of the topk boxes above score threshold
        else:
            scores = scores[score_indices]
        return scores, score_indices
