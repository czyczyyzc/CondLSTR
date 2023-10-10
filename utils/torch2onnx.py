import warnings
import onnx
import onnxsim
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ONNXRuntimeModel(nn.Module):
    """Wrapper for model's inference with ONNXRuntime."""

    def __init__(self, onnx_file, device_id):
        super(ONNXRuntimeModel, self).__init__()
        self.device_id = device_id

        sess_options = ort.SessionOptions()
        is_cuda_available = ort.get_device() == 'GPU'
        providers = ['CPUExecutionProvider']
        provider_options = [{}]
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            provider_options.insert(0, {'device_id': device_id})
        sess = ort.InferenceSession(onnx_file, sess_options, providers, provider_options)

        self.sess = sess
        self.io_binding = sess.io_binding()
        self.output_names = [out.name for out in sess.get_outputs()]
        self.is_cuda_available = is_cuda_available

    def forward(self, img):
        inputs = [img]
        input_names = ['img']
        input_types = [np.int32]  # [np.float32]

        # set io binding for inputs/outputs
        device_type = 'cuda' if self.is_cuda_available else 'cpu'
        for input_, input_name, input_type in list(zip(inputs, input_names, input_types)):
            self.io_binding.bind_input(
                name=input_name,
                device_type=device_type,
                device_id=self.device_id,
                element_type=input_type,
                shape=input_.shape,
                buffer_ptr=input_.data_ptr())

        for name in self.output_names:
            self.io_binding.bind_output(name)
        # run session to get outputs
        self.sess.run_with_iobinding(self.io_binding)
        ort_outputs = self.io_binding.copy_outputs_to_cpu()
        ort_outputs = [torch.from_numpy(x) for x in ort_outputs]
        return ort_outputs


def torch2onnx(model,
               jit_file,
               onnx_file,
               input_dict,
               opset_version=11,
               do_simplify=True,
               dynamic_export=True,
               do_constant_folding=True,
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

    input_ = (img,)
    input_names = model.input_names
    output_names = model.output_names
    print(input_names)
    print(output_names)

    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = model.dynamic_axes
    print(dynamic_axes)

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_,
            onnx_file,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=do_constant_folding,
            verbose=verbose,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes)

    if do_simplify:
        input_dic = {'img': img.detach().cpu().numpy()}
        model_opt, check_ok = onnxsim.simplify(
            onnx_file,
            input_data=input_dic,
            dynamic_input_shape=dynamic_export)
        if check_ok:
            onnx.save(model_opt, onnx_file)
            print(f'Successfully simplified ONNX model: {onnx_file}')
        else:
            warnings.warn('Failed to simplify ONNX model.')
    print(f'Successfully exported ONNX model: {onnx_file}')

    if verify:
        # check by onnx
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)

        img = img.cuda()

        # wrap onnx model
        onnx_model = ONNXRuntimeModel(onnx_file, 0)

        # get onnx output
        onnx_results = onnx_model(img)

        print('###############################################3')
        for x in onnx_results:
            print(x.shape)
            print(x.dtype)

        # get pytorch output
        model = model.cuda()
        with torch.no_grad():
            torch_results = model(img)

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        for x in torch_results:
            print(x.shape)
            print(x.dtype)

        # visualize predictions
        compare_pairs = list(zip(onnx_results, torch_results))

        # compare a part of result
        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res.cpu().numpy(), p_res.cpu().numpy(), rtol=1e-03, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')
