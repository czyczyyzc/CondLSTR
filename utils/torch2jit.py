import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile


class JITRuntimeModel(nn.Module):
    """Wrapper for model's inference with ONNXRuntime."""
    def __init__(self, jit_file, device_id, postprocess):
        super(JITRuntimeModel, self).__init__()
        self.device_id = device_id
        self.model = torch.jit.load(jit_file, map_location='cpu').to(device='cuda:{}'.format(device_id))  #.cuda(device=device_id)
        self.postprocess = postprocess

    def forward(self, data_dict):
        img, img_metas = data_dict['img'], data_dict['img_metas']
        self.model.eval()
        output = self.model(img)
        ret_dict = self.postprocess(output, img_metas)
        return ret_dict


def torch2jit(model,
              input_dict,
              output_file='tmp.jit',
              verify=True):

    img, img_metas = input_dict['img'], input_dict['img_metas']
    img, img_metas = img.unsqueeze(0), [img_metas]
    input_dict['img'], input_dict['img_metas'] = img, img_metas

    input_shape = img.shape[-2:]
    # update real input shape of each single img
    for img_meta in img_metas:
        img_meta.update(input_shape=input_shape)

    img = img * torch.tensor([[[58.395]], [[57.12]], [[57.375]]]) + torch.tensor([[[123.675]], [[116.28]], [[103.53]]])
    img = F.interpolate(img, size=(1080, 1920), mode='bilinear', align_corners=True).int()     # (1, 3, H, W)

    img = img[0][[2, 1, 0]].permute(1, 2, 0).contiguous()

    # img = img.permute(0, 2, 3, 1).contiguous()
    print(img.shape)
    print(img.dtype)

    # macs, params = get_model_complexity_info(model, (3, 768, 1344), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print(macs)
    # print(params)
    #
    # sys.exit()

    # macs, params = profile(model, inputs=(img,))
    # print(macs, params)

    # device = torch.device("cuda:0")
    # example_inputs = img.to(device=device)
    # model = model.to(device=device)

    example_inputs = img
    model.eval()

    model = torch.jit.trace(model.forward, example_inputs, optimize=None, check_trace=True,
                            check_inputs=None, check_tolerance=1e-08, strict=True)
    model.save(output_file)

    sys.exit()

    # if verify:
    #     keys = input_dict.keys()
    #     for key in keys:
    #         input_dict[key] = to_device(input_dict[key], device='cuda')
    #
    #     # wrap onnx model
    #     jit_model = JITRuntimeModel(output_file, 0)
    #
    #     # get onnx output
    #     jit_results = jit_model(input_dict, model.postprocess_jit)
    #
    #     # get pytorch output
    #     model = model.cuda()
    #     with torch.no_grad():
    #         pytorch_results = model(input_dict)
    #
    #     # visualize predictions
    #
    #     pytorch_keys = pytorch_results.keys()
    #     jit_keys = jit_results.keys()
    #     assert pytorch_keys == jit_keys
    #
    #     compare_pairs = [(jit_results[k], pytorch_results[k]) for k in pytorch_keys]
    #     # compare a part of result
    #     err_msg = 'The numerical values are different between Pytorch' + \
    #               ' and ONNX, but it does not necessarily mean the' + \
    #               ' exported ONNX model is problematic.'
    #     # check the numerical value
    #     for onnx_res, pytorch_res in compare_pairs:
    #         for o_res, p_res in zip(onnx_res, pytorch_res):
    #             np.testing.assert_allclose(
    #                 o_res, p_res, rtol=1e-03, atol=1e-05, err_msg=err_msg)
    #     print('The numerical values are the same between Pytorch and ONNX')
