# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Main script for export classification model to Torch.Jit.
"""

import os
import argparse
import torch

from metavision_ml.classification import get_model_class, is_rnn
import json
import numpy as np

PARAMS_TO_EXPORT = ["delta_t", "label_delta_t", "use_label_freq", "models", "preprocess_channels", "height", "width",
                    "preprocess", "preprocess_kwargs"]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_classifier(lightning_model, out_directory, tseq, batch_size, precision=32):
    """Exports Jitted classifier
    & json parameter files
    Args:
        lightning_model : Pytorch lightning classification model
        out_directory: output directory
        tseq (int): time sequence of one random input tensor
        batch_size (int): batch size of one random input tensor
        precision (int): set to 16 to export in half precision (float16)  
    """
    assert precision in (16,32), "only 16 and 32 precision (float) are supported"

    classifier = lightning_model.net.cpu()
    classifier.eval()
    params = lightning_model.hparams
    if is_rnn(params.models):
        label_map = ['background'] + params['classes']
    else:
        label_map = params['classes']

    jit_model = torch.jit.script(classifier)
    if precision == 16:
        jit_model.half()
    jit_model.save(os.path.join(out_directory, "model_classifier.ptjit"))

    #for compatibility with old rnn models
    if not "preprocess_kwargs" in params:
        assert "max_incr_per_pixel" in params, "something wrong with params, either 'preprocess_kwargs' or 'max_incr_per_pixel' should be given"
        params["preprocess_kwargs"] = {"max_incr_per_pixel": params["max_incr_per_pixel"]}
    
    # export relevant params for inference
    dic_json = {key: int(params[key]) if type(params[key]).__module__ == np.__name__ else params[key] for key in
                PARAMS_TO_EXPORT}
    dic_json["label_map"] = label_map
    dic_json["num_classes"] = len(label_map)

    for key in dic_json["preprocess_kwargs"]:
        if key == "preprocess_dtype":
            dic_json["preprocess_kwargs"].pop('preprocess_dtype')
        if type(dic_json["preprocess_kwargs"][key]).__module__ == np.__name__:
            dic_json["preprocess_kwargs"][key] = int(dic_json["preprocess_kwargs"][key])

    filename_json = os.path.join(out_directory, "info_classifier_jit.json")
    json.dump(dic_json, open(filename_json, "w"), indent=4, default=lambda o: o.__dict__, sort_keys=True)

    # sanity check
    if is_rnn(params.models):
        x = torch.rand((tseq, batch_size, params['preprocess_channels'], params['height'], params['width']))
    else:
        x = torch.rand((batch_size, params['preprocess_channels'] * params['num_ev_reps'],
                        params['height'], params['width']))
    if precision == 16:
        if not torch.cuda.is_available():
            print("Warning: Model exported with half precision, but no GPU available! Tests will be skipped")
            return
        else:
            device = torch.device("cuda")
            x = x.to(device).half()
            classifier = classifier.to(device).half()
            jit_model = jit_model.to(device)
    # for test we need to disable jit optimizations 
    # otherwise there will be some difference in the results (especially for half precision)
    # see https://github.com/pytorch/pytorch/issues/74534
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_nvfuser_enabled(False)

    ckpt_out = classifier(x)
    jit_out = jit_model(x)
    np.testing.assert_allclose(to_numpy(ckpt_out), to_numpy(jit_out), rtol=1e-6, atol=1e-6)
    print("torchjit result has been tested, OK")


def main(
        checkpoint_path,
        out_directory,
        tseq=1,
        batch_size=12,
        precision=32):
    """
    Performs the export of a model

    Args:
        checkpoint_path (str): path to checkpoint file saved during training
        out_directory (str): output directory where the exported model will be saved
        tseq (int): time sequence of one random input tensor
        batch_size (int): batch size of one random input tensor
        precision (int): set to 16 to export in half precision (float16)  
    """
    # 1. create directory
    if not os.path.exists(out_directory):
        print('Creating destination folder: {}'.format(out_directory))
        os.makedirs(out_directory)

    # 2. load classification model
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hparams = argparse.Namespace(**checkpoint['hyper_parameters'])

    # for compatibility with old models
    if not hasattr(hparams, "preprocess_channels"):
        hparams.preprocess_channels = hparams.in_channels

    model_class = get_model_class(hparams.models)
    model = model_class(hparams)
    model.load_state_dict(checkpoint['state_dict'])

    # 3. export
    export_classifier(model, out_directory, tseq, batch_size, precision)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
