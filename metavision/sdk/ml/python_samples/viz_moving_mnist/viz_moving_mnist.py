# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Toy Problem visualization
"""
import sys
import time
import numpy as np
import cv2
from metavision_ml.detection_tracking.display_frame import draw_box_events
from metavision_ml.data import box_processing as box_api
from metavision_ml.data.moving_mnist import MovingMNISTDataset


def show_mnist(dataset_dir, tbins=10, height=128, width=128, num_workers=1, batch_size=8, max_frames_per_video=20):
    """show moving mnist dataset

    Args:
        dataset_dir (str): directory where the data is stored (will be downloaded if necessary)
        tbins (int): number of time-bins per batch
        height (int): height of image
        width (int): width of image
        num_workers (int): number of working processes
        batch_size (int): number of sequences per batch
        max_frames_per_video (int): number of frames from the same sequence (must be greater than tbins)
    """
    assert max_frames_per_video >= tbins, "max_frames_per_video ({}) must be greater than tbins ({})".format(
        max_frames_per_video,
        tbins)
    dataloader = MovingMNISTDataset(
        tbins,
        num_workers,
        batch_size,
        height,
        width,
        max_frames_per_video,
        10000,
        True,
        dataset_dir=dataset_dir)
    label_map = ["background"] + [str(i) for i in range(10)]
    show_batchsize = batch_size
    start = 0
    nrows = 2 ** ((show_batchsize.bit_length() - 1) // 2)
    ncols = show_batchsize // nrows
    grid = np.zeros((nrows, ncols, height, width, 3), dtype=np.uint8)
    for i, data in enumerate(dataloader):
        batch, targets = data['inputs'], data['labels']
        height, width = batch.shape[-2], batch.shape[-1]
        runtime = time.time() - start
        for t in range(10):
            grid[...] = 0
            for n in range(batch_size):
                img = (batch[t, n].permute(1, 2, 0).cpu().numpy() * 255).copy()
                boxes = targets[t][n]
                boxes = box_api.box_vectors_to_bboxes(boxes[:, :4], boxes[:, 4])
                img = draw_box_events(img, boxes, label_map, draw_score=False)
                grid[n // ncols, n % ncols] = img
            im = grid.swapaxes(1, 2).reshape(nrows * height, ncols * width, 3)
            cv2.imshow('dataset', im)
            key = cv2.waitKey(20)
            if key == 27:
                break
        sys.stdout.write('\rtime: %f' % (runtime))
        sys.stdout.flush()
        start = time.time()


if __name__ == '__main__':
    import fire
    fire.Fire(show_mnist)
