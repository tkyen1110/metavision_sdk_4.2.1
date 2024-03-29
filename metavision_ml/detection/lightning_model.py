# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Pytorch Lightning Module for training the detector
"""
import argparse
import os
from itertools import islice
from collections import defaultdict

import cv2
import numpy as np
from skvideo.io import FFmpegWriter

import torch
import pytorch_lightning as pl

from metavision_ml.detection.data_factory import get_dataloader
from metavision_ml.metrics.coco_eval import CocoEvaluator
from metavision_ml.data import box_processing as box_api
from metavision_ml.detection.single_stage_detector import SingleStageDetector
from metavision_ml.detection_tracking.display_frame import draw_box_events
from metavision_sdk_core import EventBbox


def bboxes_to_box_vectors(bbox):
    """
    uniformizes bbox dtype to x1,y1,x2,y2,class_id,track_id
    """
    if isinstance(bbox, list):
        return [bboxes_to_box_vectors(item) for item in bbox]
    elif isinstance(bbox, np.ndarray) and bbox.dtype != np.float32:
        return box_api.bboxes_to_box_vectors(bbox)
    else:
        return bbox


class LightningDetectionModel(pl.LightningModule):
    """
    Pytorch Lightning model for neural network to predict boxes.

    The detector built by build_ssd should be a Detector with "compute_loss" and "get_boxes" implemented.

    Args:
        hparams (argparse.Namespace): argparse from train.py application
    """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(hparams)

        model_keys = ['feature_extractor', 'in_channels', 'feature_base',
                      'feature_channels_out', 'anchor_list', 'max_boxes_per_input']
        params = vars(hparams)
        model_params = {k: params[k] for k in model_keys}
        model_params['num_classes'] = len(params['classes'])
        self.detector = SingleStageDetector(**model_params)
        self.label_map = ['background'] + hparams.classes

    def load_pretrained(self, checkpoint_path):
        """ Loads a pretrained detector (of this class)
            and transfer the weights to this module for fine tuning.

            In addition it may remap the old classification weights if
            some overlap exists between old and new list of classes.

            Args:
                checkpoint_path (str): path to checkpoint of pretrained detector.
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
        module_src = LightningDetectionModel(hparams)
        module_src.load_state_dict(checkpoint['state_dict'])

        # Copy Buffers
        dict_dst = dict(self.detector.named_buffers())
        for name, buf in module_src.detector.named_buffers():
            if name in dict_dst:
                if dict_dst[name].data.shape == buf.data.shape:
                    dict_dst[name].data.copy_(buf.data)

        # Copy Parameters
        dict_dst = dict(self.named_parameters())
        for name, param in module_src.named_parameters():
            if name in dict_dst:
                if dict_dst[name].data.shape == param.data.shape:
                    dict_dst[name].data.copy_(param.data)

        # Re-Organize logits if you can
        if hasattr(self.detector, "map_cls_weights"):
            old_classes = hparams.classes
            new_classes = self.hparams.classes
            self.detector.map_cls_weights(module_src.detector, old_classes, new_classes)

    def forward(self, x):
        return self.detector.forward(x)

    def training_step(self, batch, batch_nb):
        batch["labels"] = bboxes_to_box_vectors(batch["labels"])

        self.detector.reset(batch["mask_keep_memory"])

        loss_dict = self.detector.compute_loss(batch["inputs"], batch["labels"], batch['frame_is_labeled'])

        if loss_dict is None:
            return

        loss = sum([value for key, value in loss_dict.items()])

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.trainer.fit_loop.running_loss.append(loss)

        self.log('train_metrics/loss', loss.item())
        for k, v in loss_dict.items():
            self.log('train_metrics/' + k, v.item())

        return loss

    def validation_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

    def test_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        if self.hparams.lr_scheduler_step_gamma is None:
            print("No Learning Rate Scheduler")
            return optimizer
        print("Using Learning Rate Scheduler: ", self.hparams.lr_scheduler_step_gamma)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.lr_scheduler_step_gamma)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataloader = get_dataloader(self.hparams, 'train')
        if hasattr(dataloader, "dataset"):
            dataloader.dataset.shuffle()
        return dataloader

    def val_dataloader(self):
        dataloader = get_dataloader(self.hparams, 'val')
        return dataloader

    def test_dataloader(self):
        dataloader = get_dataloader(self.hparams, 'test')
        return dataloader

    def demo_dataloader(self):
        dataloader = get_dataloader(self.hparams, 'val')
        return dataloader

    def training_epoch_end(self, outputs):
        if self.current_epoch and self.current_epoch % self.hparams.demo_every == 0:
            self.demo_video(self.current_epoch, show_video=False)
        return

    def validation_epoch_end(self, outputs):
        return self.inference_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self.inference_epoch_end(outputs, 'test')

    def inference_epoch_end(self, outputs, mode='val'):
        """
        Runs Metrics

        Args:
            outputs: accumulated outputs
            mode: 'val' or 'test'
        """
        print('==> Start evaluation')
        # merge all dictionaries
        dt_detections = defaultdict(list)
        gt_detections = defaultdict(list)

        for item in outputs:
            for k, v in item['gt'].items():
                gt_detections[k].extend(v)
            for k, v in item['dt'].items():
                dt_detections[k].extend(v)

        evaluator = CocoEvaluator(classes=self.label_map, height=self.hparams.height, width=self.hparams.width,
                                  verbose=self.hparams.verbose)
        for key in gt_detections:
            evaluator.partial_eval([np.concatenate(gt_detections[key])], [np.concatenate(dt_detections[key])])
        coco_kpi = evaluator.accumulate()

        for k, v in coco_kpi.items():
            print(k, ': ', v)
            self.log(f'coco_metrics/{k}', v)

        self.log(mode + '_acc', coco_kpi['mean_ap'])

    def inference_step(self, batch, batch_nb):
        """
        One step of validation
        """
        with torch.no_grad():
            self.detector.reset(batch["mask_keep_memory"])
            preds = self.detector.get_boxes(batch["inputs"], score_thresh=0.05)

        dt_dic, gt_dic = self.accumulate_predictions(
            preds,
            batch["labels"],
            batch["video_infos"],
            batch["frame_is_labeled"])
        return {'dt': dt_dic, 'gt': gt_dic}

    def accumulate_predictions(self, preds, targets, video_infos, frame_is_labeled):
        """
        Accumulates prediction to run coco-metrics on the full videos
        """
        dt_detections = {}
        gt_detections = {}
        for t in range(len(targets)):
            for i in range(len(targets[t])):
                gt_boxes = targets[t][i]
                pred = preds[t][i]

                video_info, tbin_start, _ = video_infos[i]

                # skipping when padding or the frame is not labeled
                if video_info.padding or frame_is_labeled[t, i] == False:
                    continue

                name = video_info.path
                if name not in dt_detections:
                    dt_detections[name] = [np.zeros((0), dtype=box_api.EventBbox)]
                if name not in gt_detections:
                    gt_detections[name] = [np.zeros((0), dtype=box_api.EventBbox)]
                assert video_info.start_ts == 0
                ts = tbin_start + t * video_info.delta_t

                if ts < self.hparams.skip_us:
                    continue

                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.cpu().numpy()
                if gt_boxes.dtype == np.float32:
                    gt_boxes = box_api.box_vectors_to_bboxes(gt_boxes[:, :4], gt_boxes[:, 4], ts=ts)

                # Targets are timed
                # Targets timestamped before 0.5s are skipped
                # Labels are in range(1, C) (0 is background) (not in 0, C-1, where 0 would be first class)
                if pred['boxes'] is not None and len(pred['boxes']) > 0:
                    boxes = pred['boxes'].cpu().data.numpy()
                    labels = pred['labels'].cpu().data.numpy()
                    scores = pred['scores'].cpu().data.numpy()
                    dt_boxes = box_api.box_vectors_to_bboxes(boxes, labels, scores, ts=ts)
                    dt_detections[name].append(dt_boxes)
                else:
                    dt_detections[name].append(np.zeros((0), dtype=EventBbox))

                if len(gt_boxes):
                    gt_boxes["t"] = ts
                    gt_detections[name].append(gt_boxes)
                else:
                    gt_detections[name].append(np.zeros((0), dtype=EventBbox))

        return dt_detections, gt_detections

    def demo_video(self, epoch=0, num_batches=100, show_video=False):
        """
        This runs our detector on several videos of the testing dataset
        """
        self.test_data = self.demo_dataloader() if not hasattr(self, "test_data") else self.test_data
        hparams = self.hparams

        height, width = hparams.height, hparams.width
        batch_size = hparams.batch_size
        nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
        ncols = int(np.ceil(hparams.batch_size / nrows))

        grid = np.zeros((nrows * hparams.height, ncols * hparams.width, 3), dtype=np.uint8)
        video_name = os.path.join(hparams.root_dir, 'videos', f'video#{epoch:d}.mp4')

        dir = os.path.dirname(video_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)

        video_writer = FFmpegWriter(video_name, outputdict={'-crf': '20', '-preset': 'veryslow'})

        self.detector.eval()

        if show_video:
            window_name = "test epoch {:d}".format(epoch)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        for batch_nb, batch in enumerate(islice(self.test_data, num_batches)):
            batch["inputs"] = batch["inputs"].to(self.device)
            batch["mask_keep_memory"] = batch["mask_keep_memory"].to(self.device)

            images = batch["inputs"].cpu().clone().data.numpy()

            with torch.no_grad():
                self.detector.reset(batch["mask_keep_memory"])
                predictions = self.detector.get_boxes(batch["inputs"], score_thresh=0.5)

            for t in range(len(images)):
                for i in range(len(images[0])):
                    frame = self.test_data.get_vis_func()(images[t][i])
                    pred = predictions[t][i]
                    target = batch["labels"][t][i]

                    if isinstance(target, torch.Tensor):
                        target = target.cpu().numpy()
                    if target.dtype.isbuiltin or target.dtype in [
                            np.dtype('float16'),
                            np.dtype('float32'),
                            np.dtype('float64'),
                            np.dtype('int16'),
                            np.dtype('int32'),
                            np.dtype('int64'),
                            np.dtype('uint16'),
                            np.dtype('uint32'),
                            np.dtype('uint64')]:
                        target = box_api.box_vectors_to_bboxes(target[:, :4], target[:, 4])

                    if pred['boxes'] is not None:
                        boxes = pred['boxes'].cpu().data.numpy()
                        labels = pred['labels'].cpu().data.numpy()
                        scores = pred['scores'].cpu().data.numpy()
                        bboxes = box_api.box_vectors_to_bboxes(boxes, labels, scores)
                        frame = draw_box_events(frame, bboxes, self.label_map, draw_score=True, thickness=2)

                    frame = draw_box_events(
                        frame, target, self.label_map, force_color=[
                            255, 255, 255], draw_score=False, thickness=1)

                    y = i // ncols
                    x = i % ncols
                    y1, y2 = y * height, (y + 1) * height
                    x1, x2 = x * width, (x + 1) * width
                    grid[y1:y2, x1:x2] = frame

                if show_video:
                    cv2.imshow(window_name, grid)
                    cv2.waitKey(5)

                video_writer.writeFrame(grid)

        video_writer.close()

        if show_video:
            cv2.destroyWindow(window_name)
