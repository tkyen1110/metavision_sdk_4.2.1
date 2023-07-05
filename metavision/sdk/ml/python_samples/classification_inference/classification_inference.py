# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Run inference on classification module
"""

import argparse
import os
import numpy as np
import cv2
import torch
import json
from skvideo.io import FFmpegWriter
import h5py
from metavision_ml.data import CDProcessorIterator, HDF5Iterator
from metavision_ml.utils.h5_writer import HDF5Writer
from datetime import datetime
import glob
from collections import deque


def viz_histo_filtered(im, val_max=0.5):
    """visualize strongly filtered histo image with 3 channels

      Args:
          im (np.ndarray): Array of shape (2,H,W)
          val_max (float): cutoff threshold for visualization

      Returns:
          output_array: array of shape (H,W,3)
      """
    im = im.astype(np.float32)
    im = im[1] - im[0]
    im = np.clip(im, -val_max, val_max)
    im = ((im + val_max) / (2 * val_max) * 255).astype(np.uint8)
    im = im[..., None].repeat(3, 2)
    return im


def inference_parser():
    parser = argparse.ArgumentParser(description='Perform inference with the classification module',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('torchscript_dir', type=str, help='path to the torchscript model and the json file '
                                                          'with model description.')
    parser.add_argument('-p', '--path', type=str, default="",
                        help='RAW, HDF5 or DAT filename, leave blank to use a camera. '
                             'Warning if you use a HDF5 file the parameters used for pre-computation must '
                             'match those of the model.')
    parser.add_argument('--delta-t', type=int, default=50000,
                        help='duration of timeslice (in us) in which events are accumulated to compute features.')
    parser.add_argument('--start-ts', type=int, default=0,
                        help='timestamp (in microseconds) from which the computation begins. ')
    parser.add_argument('--max-duration', type=int, default=None,
                        help='maximum duration of the inference file in us.')
    parser.add_argument('--height-width', dest='hw', nargs=2, default=None, type=int,
                        help="if set, downscale the feature tensor to the requested resolution using interpolation"
                             " Possible values are only power of two of the original resolution.")
    parser.add_argument('-t', '--threshold', dest='cls_threshold', default=0.7, type=float,
                        help="classification threshold")
    parser.add_argument("--cpu", action="store_true", help='run on CPU')
    parser.add_argument("-s", "--save", dest='save_h5', default='',
                        help='Path of the directory to save the result in a hdf5 format')
    parser.add_argument("-w", "--write-video", default='',
                        help='Path of the directory to save the visualization in a .mp4 video.')
    parser.add_argument("--no-display", dest="display", action="store_false",
                        help='if set, deactivate the display Window')
    parser.add_argument('--max-incr-per-pixel', type=int, default=2,
                        help='Maximum number of increments (events) per pixel. This value needs to be consistent '
                             'with that of the training')
    # args for RNN-type model
    parser.add_argument("--max-low-activity-tensor", default=0.15, type=float,
                        help="Maximum tensor value for a frame to be considered as low activity")
    parser.add_argument("--max-low-activity-nb-frames", default=5, type=int,
                        help="Maximum number of low activity frames before the model internal state is reset")
    parser.add_argument("--display-reset-memory", action="store_true",
                        help="Displays when network is reset (low activity)")
    # args for FF-type model
    parser.add_argument('--use-FF-model', action="store_true", help='use FF type of model')
    parser.add_argument('--max_rolling-window', type=int, default=4,
                        help='Maximum number of frames for calculating the rolling average of prediction')
   # args = parser.parse_args()
    return parser


@torch.no_grad()
def _proc(
        preprocessor,
        cls_model,
        model_json,
        args,
        video_process=None,
        h5writer=None,
):
    """Sub function performing preprocessing and visualization. """
    COLOR = (0, 255, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    nb_consecutive_low_activity_frames = 0
    if args.display:
        WINDOW_NAME = "Gesture Recognition"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if args.use_FF_model:
        Q = deque(maxlen=args.max_rolling_window)

    for tensor in preprocessor:
        do_reset = False
        if not args.use_FF_model:
            tensor = tensor[None]
        out = torch.squeeze(cls_model(tensor))
        yhat = torch.nn.functional.softmax(out, dim=-1).cpu().numpy()

        if tensor.max() < args.max_low_activity_tensor:
            nb_consecutive_low_activity_frames += 1
        else:
            nb_consecutive_low_activity_frames = 0

        if nb_consecutive_low_activity_frames >= args.max_low_activity_nb_frames and args.display_reset_memory:
            do_reset = True
            cls_model.reset_all()
            nb_consecutive_low_activity_frames = 0

        if args.use_FF_model:
            tensor = tensor[None]
        tensor = tensor.detach()[0, 0].cpu().numpy()
        img = viz_histo_filtered(tensor)
        if args.use_FF_model:
            Q.append(yhat)
            q_mean = np.array(Q).mean(axis=0)
            yhat_indice = np.argmax(q_mean, axis=-1)
        else:
            yhat_indice = np.argmax(yhat, axis=-1)
        yhat_cls = model_json["label_map"][yhat_indice]

        # filter out background and predictions with low confidence value
        if yhat[yhat_indice] >= args.cls_threshold and yhat_indice != 0:
            cv2.putText(img, yhat_cls, (10, img.shape[0] - 60), FONT, 0.5, COLOR)
            cv2.putText(img, "Score: {:.2f}".format(yhat[yhat_indice]), (10, img.shape[0] - 20), FONT, 0.5, COLOR)
        if do_reset and args.display_reset_memory:
            cv2.putText(img, "RESET MEMORY", (10, 20), FONT, 0.4, COLOR)
        
        if args.display:
            cv2.imshow(WINDOW_NAME, img[..., ::-1])
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break

        if video_process is not None:
            video_process.writeFrame(img)

        if h5writer is not None:
            h5writer.write(yhat)


def main():
    args = inference_parser().parse_args()
    run(args)

def run(args):
    # Load the network
    model_file = glob.glob(os.path.join(args.torchscript_dir, "*.ptjit"))
    assert len(model_file) == 1, "more than one torchjit models is ambiguous"
    model = torch.jit.load(model_file[0])
    json_file = glob.glob(os.path.join(args.torchscript_dir, "*.json"))
    assert len(json_file) == 1, "more than one json files is ambiguous"
    json_file = json_file[0]
    with open(json_file, "r") as jfile:
        model_json = json.load(jfile)

    # Get delta t
    if model_json['delta_t'] != args.delta_t:
        print(
            f"\nModel was trained for a delta_t of {model_json['delta_t']}us but is used with {args.delta_t}!\n")
        print("Its performance could be negatively affected.\n")

    # Prepare H5 output path
    if args.save_h5:
        h5_dirname = os.path.dirname(args.save_h5)
        if not os.path.exists(h5_dirname):
            os.makedirs(h5_dirname)

    # Get height, width
    if args.hw:
        height, width = args.hw
    else:
        height = model_json["height"]
        width = model_json["width"]

    # Initialize the device
    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    model.to(device)

    preprocess_kwargs = model_json["preprocess_kwargs"]
    if not "quantized" in model_json["preprocess"]:
        # Get max_incr_per_pixel
        max_incr_per_pixel = model_json["max_incr_per_pixel"] if "max_incr_per_pixel" in model_json \
            else args.max_incr_per_pixel
        preprocess_kwargs.update({'max_incr_per_pixel': max_incr_per_pixel})
    
    # Process the events
    if args.path.endswith('h5'):
        preprocessor = HDF5Iterator(args.path, device=device, height=height, width=width)
        preprocessor.checks(model_json["preprocess"], delta_t=args.delta_t)
    else:
        preprocessor = CDProcessorIterator(
            args.path, model_json["preprocess"],
            delta_t=args.delta_t, max_duration=args.max_duration, device=device, height=height, width=width,
            start_ts=args.start_ts, preprocess_kwargs=preprocess_kwargs)

    # Initialize video outputs
    filename = os.path.splitext(os.path.basename(args.path))[0] if args.path != "" \
        else datetime.now().strftime("chifoumi_inference_%Y%m%d_%H%M%S")
    if args.write_video:
        video_path = os.path.join(args.write_video, filename + '.mp4')
        process = FFmpegWriter(video_path)
    else:
        process = None

    # Initialize H5 output
    if args.save_h5:
        h5_file = os.path.join(args.save_h5, filename + '_cls.h5')
        shape = [len(model_json["label_map"])]
        h5w = HDF5Writer(
            h5_file, "cls", shape, dtype=np.float16,
            attrs={"events_to_tensor": np.string_(model_json["preprocess"]),
                   'checkpoint_path': os.path.basename(args.torchscript_dir),
                   'input_file_name': os.path.basename(args.path),
                   "delta_t": np.uint32(args.delta_t),
                   'model_input_height': height,
                   "model_input_width": width})
        h5w.dataset_size_increment = 100
    else:
        h5w = None

    model.eval()

    _proc(
        preprocessor,
        model,
        model_json,
        args=args,
        video_process=process,
        h5writer=h5w
    )

    # close everything
    if args.write_video:
        process.close()
    if args.save_h5:
        h5w.close()
        # Update hdf5
        cls_h5 = h5py.File(h5_file, "r+")
        T, _ = cls_h5["cls"].shape
        cls_start_ts_np = np.arange(0, T * args.delta_t, args.delta_t)
        cls_end_ts_np = np.arange(args.delta_t, T * args.delta_t + 1, args.delta_t)
        cls_h5.create_dataset("cls_start_ts", data=cls_start_ts_np, compression="gzip")
        cls_h5.create_dataset("cls_end_ts", data=cls_end_ts_np, compression="gzip")
        cls_h5.close()

    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
