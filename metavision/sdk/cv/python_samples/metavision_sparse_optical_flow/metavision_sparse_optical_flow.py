# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Code sample showing how to use Metavision SDK to display results of sparse optical flow.
"""

import numpy as np
import os
import h5py
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import SparseOpticalFlowAlgorithm, SparseOpticalFlowConfigPreset, SparseFlowFrameGeneratorAlgorithm, SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
from skvideo.io import FFmpegWriter


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Sparse Optical Flow sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group(
        "Input", "Arguments related to input sequence.")
    input_group.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    input_group.add_argument(
        '-r', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
    input_group.add_argument(
        '-a', '--accumulation-time-us', type=int, default=10000, dest="accumulation_time_us",
        help="Accumulation time (in us). Used to generate a frame.")

    noise_filtering_group = parser.add_argument_group(
        "Noise Filtering", "Arguments related to STC noise filtering.")
    noise_filtering_group.add_argument(
        "--disable-stc", dest="disable_stc", action="store_true",
        help="Disable STC noise filtering. All other options related to noise filtering are discarded.")
    noise_filtering_group.add_argument("--stc-filter-thr", dest="stc_filter_thr", type=int,  default=40000,
                                       help="Length of the time window for filtering (in us).")
    noise_filtering_group.add_argument(
        "--disable-stc-cut-trail", dest="stc_cut_trail", default=True, action="store_false",
        help="When stc cut trail is enabled, after an event goes through, it removes all events until change of polarity.")

    output_flow_group = parser.add_argument_group(
        "Output flow", "Arguments related to output optical flow.")
    output_flow_group.add_argument(
        "--output-sparse-npy-filename", dest="output_sparse_npy_filename",
        help="If provided, the predictions will be saved as numpy structured array of EventOpticalFlow. In this "
        "format, the flow vx and vy are expressed in pixels per second.")
    output_flow_group.add_argument(
        "--output-dense-h5-filename", dest="output_dense_h5_filename",
        help="If provided, the predictions will be saved as a sequence of dense flow in HDF5 data. The flows are "
        "averaged pixelwise over timeslices of --accumulation-time-us. The dense flow is expressed in terms of "
        "pixels per timeslice (of duration accumulation-time-us), not in pixels per second.")
    output_flow_group.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting video.")
    output_flow_group.add_argument(
        '--fps', dest='fps', type=int, default=25,
        help="replay fps of output video")

    args = parser.parse_args()

    if args.output_sparse_npy_filename:
        assert not os.path.exists(args.output_sparse_npy_filename)
    if args.output_dense_h5_filename:
        assert not os.path.exists(args.output_dense_h5_filename)

    return args


def main():
    """ Main """
    args = parse_args()

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(
        input_path=args.input_path, delta_t=args.accumulation_time_us)

    # Set ERC to 20Mev/s
    if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device:
        erc_module = mv_iterator.reader.device.get_i_erc_module()
        if erc_module:
            erc_module.set_cd_event_rate(20000000)
            erc_module.enable(True)

    if args.replay_factor > 0 and not is_live_camera(args.input_path):
        mv_iterator = LiveReplayEventsIterator(
            mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Event Frame Generator
    event_frame_gen = OnDemandFrameGenerationAlgorithm(
        width, height, args.accumulation_time_us)

    # Sparse Optical Flow Algorithm

    flow_algo = SparseOpticalFlowAlgorithm(
        width, height, SparseOpticalFlowConfigPreset.SlowObjects)
    flow_buffer = SparseOpticalFlowAlgorithm.get_empty_output_buffer()

    # Flow Frame Generator
    flow_frame_gen = SparseFlowFrameGeneratorAlgorithm()

    # STC filter
    stc_filter = SpatioTemporalContrastAlgorithm(
        width, height, args.stc_filter_thr, args.stc_cut_trail)
    events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()

    all_flow_events = []
    all_dense_flows = []
    all_dense_flows_start_ts = []
    all_dense_flows_end_ts = []

    # Window - Graphical User Interface
    with Window(title="Metavision Sparse Optical Flow", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        if args.out_video:
            video_name = args.out_video + ".avi"
            writer = FFmpegWriter(video_name, inputdict={'-r': str(args.fps)}, outputdict={
                '-vcodec': 'libx264',
                '-r': str(args.fps)
            })

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        output_img = np.zeros((height, width, 3), np.uint8)
        processing_ts = mv_iterator.start_ts
        # Process events
        for evs in mv_iterator:
            processing_ts += mv_iterator.delta_t

            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            if args.disable_stc:
                events_buf = evs
            else:
                # Filter Events using STC
                stc_filter.process_events(evs, events_buf)

            # Generate Frame of Events
            event_frame_gen.process_events(events_buf)
            event_frame_gen.generate(processing_ts, output_img)

            # Estimate the flow events
            flow_algo.process_events(events_buf, flow_buffer)
            if args.output_sparse_npy_filename:
                all_flow_events.append(flow_buffer.numpy().copy())
            if args.output_dense_h5_filename:
                all_dense_flows_start_ts.append(
                    processing_ts - args.accumulation_time_us)
                all_dense_flows_end_ts.append(processing_ts)
                flow_np = flow_buffer.numpy()
                if flow_np.size == 0:
                    all_dense_flows.append(
                        np.zeros((2, height, width), dtype=np.float32))
                else:
                    xs, ys, vx, vy = flow_np["x"], flow_np["y"], flow_np["vx"], flow_np["vy"]
                    coords = np.stack((ys, xs))
                    abs_coords = np.ravel_multi_index(coords, (height, width))
                    counts = np.bincount(abs_coords, weights=np.ones(flow_np.size),
                                         minlength=height*width).reshape(height, width)
                    flow_x = np.bincount(
                        abs_coords, weights=vx, minlength=height*width).reshape(height, width)
                    flow_y = np.bincount(
                        abs_coords, weights=vy, minlength=height*width).reshape(height, width)
                    mask_multiple_events = counts > 1
                    flow_x[mask_multiple_events] /= counts[mask_multiple_events]
                    flow_y[mask_multiple_events] /= counts[mask_multiple_events]

                    # flow expressed in pixels per delta_t
                    flow_x *= args.accumulation_time_us * 1e-6
                    flow_y *= args.accumulation_time_us * 1e-6
                    flow = np.stack((flow_x, flow_y)).astype(np.float32)
                    all_dense_flows.append(flow)

            # Draw the flow events on top of the events
            flow_frame_gen.add_flow_for_frame_update(flow_buffer)
            flow_frame_gen.clear_ids()
            flow_frame_gen.update_frame_with_flow(output_img)

            # Update the display
            window.show(output_img)

            if args.out_video:
                writer.writeFrame(output_img.astype(np.uint8)[..., ::-1])

            if window.should_close():
                break

    if args.out_video:
        writer.close()

    if args.output_sparse_npy_filename:
        print("Writing output file: ", args.output_sparse_npy_filename)
        all_flow_events = np.concatenate(all_flow_events)
        np.save(args.output_sparse_npy_filename, all_flow_events)
    if args.output_dense_h5_filename:
        print("Writing output file: ", args.output_dense_h5_filename)
        flow_start_ts = np.array(all_dense_flows_start_ts)
        flow_end_ts = np.array(all_dense_flows_end_ts)
        flows = np.stack(all_dense_flows)
        N = flow_start_ts.size
        assert flow_end_ts.size == N
        assert flows.shape == (N, 2, height, width)
        dirname = os.path.dirname(args.output_dense_h5_filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        flow_h5 = h5py.File(args.output_dense_h5_filename, "w")
        flow_h5.create_dataset(
            "flow_start_ts", data=flow_start_ts, compression="gzip")
        flow_h5.create_dataset(
            "flow_end_ts", data=flow_end_ts, compression="gzip")
        flow_h5.create_dataset("flow", data=flows.astype(
            np.float32), compression="gzip")
        flow_h5["flow"].attrs["input_file_name"] = os.path.basename(
            args.input_path)
        flow_h5["flow"].attrs["checkpoint_path"] = "metavision_sparse_optical_flow"
        flow_h5["flow"].attrs["event_input_height"] = height
        flow_h5["flow"].attrs["event_input_width"] = width
        flow_h5["flow"].attrs["delta_t"] = args.accumulation_time_us
        flow_h5.close()


if __name__ == "__main__":
    main()
