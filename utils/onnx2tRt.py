import os
import sys
import warnings
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
import numpy as np
from .torch2onnx import ONNXRuntimeModel
from .tensorrt import TRTWrapper, load_tensorrt_plugin, save_trt_engine
from .tensorrt import onnx2trt as _onnx2trt
from .utils import to_device


class TensorRTModel(nn.Module):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, engine_file, input_names, output_names, device_id):
        super(TensorRTModel, self).__init__()
        self.device_id = device_id
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with TensorRT from source.')
        model = TRTWrapper(engine_file, input_names, output_names)
        self.model = model

    def forward(self, img):
        inputs = {'img': img}
        with torch.cuda.device(self.device_id), torch.no_grad():
            trt_outputs = self.model(inputs)
            trt_outputs = [trt_outputs[name] for name in self.model.output_names]
        trt_outputs = [out.detach().cpu() for out in trt_outputs]
        return trt_outputs


def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def onnx2trt(model,
             onnx_file,
             trt_file,
             input_dict,
             input_config=None,
             fp16_mode=False,
             workspace_size=2,
             verify=True,
             verbose=True):

    img, img_metas = input_dict['img'], input_dict['img_metas']
    img, img_metas = img.unsqueeze(0), [img_metas]

    img = img * torch.tensor([[[58.395]], [[57.12]], [[57.375]]]) + torch.tensor([[[123.675]], [[116.28]], [[103.53]]])
    img = F.interpolate(img, size=(1080, 1920), mode='bilinear', align_corners=True).int()     # (1, 3, H, W)

    # img = img[0][[2, 1, 0]].permute(1, 2, 0).contiguous()

    img = img.permute(0, 2, 3, 1).contiguous()
    print(img.shape)
    print(img.dtype)

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    if input_config is not None:
        max_shape = input_config['max_shape']
        min_shape = input_config['min_shape']
        opt_shape = input_config['opt_shape']
    else:
        max_shape = img.shape
        min_shape = img.shape
        opt_shape = img.shape

    # create trt engine and wraper
    opt_shape_dict = {'img': [min_shape, opt_shape, max_shape]}
    max_workspace_size = get_GiB(workspace_size)
    trt_engine = _onnx2trt(
        onnx_model,
        opt_shape_dict,
        log_level=trt.Logger.VERBOSE if verbose else trt.Logger.ERROR,
        fp16_mode=fp16_mode,
        max_workspace_size=max_workspace_size)
    save_dir, _ = os.path.split(trt_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_trt_engine(trt_engine, trt_file)
    print(f'Successfully created TensorRT engine: {trt_file}')

    if verify:
        # wrap ONNX and TensorRT model
        input_names = model.input_names
        output_names = model.output_names
        onnx_model = ONNXRuntimeModel(onnx_file, device_id=0)
        trt_model = TensorRTModel(trt_file, input_names, output_names, device_id=0)

        # # prepare input
        # keys = input_dict.keys()
        # for key in keys:
        #     input_dict[key] = to_device(input_dict[key], device='cuda')

        img = img.cuda()

        # inference with wrapped model
        onnx_results = onnx_model(img)
        trt_results = trt_model(img)

        # visualize predictions
        compare_pairs = list(zip(onnx_results, trt_results))

        err_msg = 'The numerical values are different between ONNX' + \
                  ' and TRT, but it does not necessarily mean the' + \
                  ' exported TRT model is problematic.'
        # check the numerical value
        for onnx_res, trt_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, trt_res):
                np.testing.assert_allclose(
                    o_res, p_res, rtol=1e-03, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')

    sys.exit()
