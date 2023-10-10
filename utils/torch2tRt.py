import sys
import warnings
import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
from torch2trt import torch2trt as _torch2trt
from .tensorrt import TRTWrapper, load_tensorrt_plugin, save_trt_engine


class TensorRTModel(nn.Module):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, engine_file, device_id):
        super(TensorRTModel, self).__init__()
        self.device_id = device_id
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with TensorRT from source.')

        input_names = ['img']
        output_names = ['logits', 'regs', 'masks', 'lane_ranges']
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


def torch2trt(model,
              trt_file,
              input_dict,
              fp16_mode=False,
              workspace_size=2,
              verify=True):

    img, img_metas = input_dict['img'], input_dict['img_metas']
    img, img_metas = img.unsqueeze(0).cuda(), [img_metas]

    input_ = (img,)

    model.eval()
    model.cuda()

    max_workspace_size = get_GiB(workspace_size)

    model_trt = _torch2trt(
        model,
        input_,
        fp16_mode=fp16_mode,
        log_level=trt.Logger.INFO,
        max_workspace_size=max_workspace_size,
        max_batch_size=1,
    )
    save_trt_engine(model_trt.engine, trt_file)
    print(f'Successfully exported TensorRT model: {trt_file}')

    if verify:
        trt_model = TensorRTModel(trt_file, device_id=0)

        # inference with wrapped model
        torch_results = model(img)
        trt_results = trt_model(img)

        # visualize predictions
        compare_pairs = list(zip(torch_results, trt_results))

        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res.cpu().numpy(), p_res.cpu().numpy(), rtol=1e-03, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')

    sys.exit()
