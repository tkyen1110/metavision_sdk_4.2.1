# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Main script for export flow model to Torch.Jit.
"""

import os
import torch

from metavision_ml.flow.lightning_model import FlowModel
from metavision_ml.preprocessing import PREPROCESSING_DICT

import json
import numpy as np


PARAMS_TO_EXPORT = ["delta_t", "array_dim", "loss_weights", "feature_extractor_name", "network_kwargs", "cin",
                    "height", "width", "num_tbins", "preprocess", "max_incr_per_pixel", "split_polarity"]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def assert_lists_of_tensor_all_close(list1, list2):
    assert len(list1) == len(list2)
    for tensor1, tensor2 in zip(list1, list2):
        np.testing.assert_allclose(to_numpy(tensor1), to_numpy(tensor2), rtol=1e-4, atol=1e-8)


def export_flow(lightning_model, out_directory, precision=32):
    """
    Exports Jitted flow model
    & json parameter files
    Args:
        lightning_model : Pytorch lightning Flow model
        out_directory: output directory
        precision (int): set to 16 to export in half precision (float16)  
    """ 
    assert precision in (16,32), "only 16 and 32 precision (float) are supported"


    flow_model = lightning_model.network.cpu()
    flow_model.eval()
    params = lightning_model.hparams

    jit_model = torch.jit.script(flow_model)
    if precision==16:
        jit_model.half()
    jit_model.save(os.path.join(out_directory, "model_flow.ptjit"))

    dic_json = {}
    assert "preprocess" in params
    preproc_name = params["preprocess"]
    for key in PARAMS_TO_EXPORT:
        if key not in params:
            assert key in PREPROCESSING_DICT[preproc_name]["kwargs"]
            dic_json[key] = PREPROCESSING_DICT[preproc_name]["kwargs"][key]
        elif type(params[key]).__module__ == np.__name__:
            assert params[key].size == 1
            dic_json[key] = params[key].item()
        else:
            dic_json[key] = params[key]

    filename_json = os.path.join(out_directory, "info_flow_jit.json")
    json.dump(dic_json, open(filename_json, "w"), indent=4, default=lambda o: o.__dict__, sort_keys=True)

    # sanity check
    x = torch.rand((3, 4, params['cin'], params['height'], params['width']))

    if precision == 16:
        if not torch.cuda.is_available():
            print("Warning: Model exported with half precision, but no GPU available! Tests will be skipped")
            return
        else:
            device = torch.device("cuda")
            x = x.to(device).half()
            flow_model = flow_model.to(device).half()
            jit_model = jit_model.to(device)
    # for test we need to disable jit optimizations 
    # otherwise there will be some difference in the results (especially for half precision)
    # see https://github.com/pytorch/pytorch/issues/74534
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_nvfuser_enabled(False)

    with torch.no_grad():
        ckpt_out_0 = flow_model(x)
        ckpt_out_1 = flow_model(x)
        flow_model.reset_all()
        ckpt_out_2 = flow_model(x)
        assert_lists_of_tensor_all_close(ckpt_out_0, ckpt_out_2)

        jit_out_0 = jit_model(x)
        jit_out_1 = jit_model(x)
        jit_model.reset_all()
        jit_out_2 = jit_model(x)
        assert_lists_of_tensor_all_close(ckpt_out_0, jit_out_0)
        assert_lists_of_tensor_all_close(ckpt_out_1, jit_out_1)
        assert_lists_of_tensor_all_close(ckpt_out_2, jit_out_2)

    print("torchjit result has been tested, OK")


def main(
        checkpoint_path,
        out_directory,
        precision=32):
    """
    Performs the export of a model

    Args:
        checkpoint_path (str): path to checkpoint file saved during training
        out_directory (str): output directory where the exported model will be saved
        precision (int): set to 16 to export in half precision (float16)  
    """
    # 1. create directory
    if not os.path.exists(out_directory):
        print('Creating destination folder: {}'.format(out_directory))
        os.makedirs(out_directory)

    # 2. load flow model
    model = FlowModel.load_from_checkpoint(checkpoint_path, strict=False)

    # 3. export
    export_flow(
        model,
        out_directory,
        precision)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
