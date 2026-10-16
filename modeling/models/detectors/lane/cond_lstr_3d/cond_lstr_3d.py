import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_locations, cal_num_params
from .loss import CondLSTR3DLoss
from .postprocess import CondLSTR3DPost


prior_prob = 0.01
bias_value = -math.log((1 - prior_prob) / prior_prob)


class DynamicMaskHead(nn.Module):

    def __init__(self,
                 num_layers,
                 in_channels,
                 channels,
                 mask_out_stride,
                 weight_nums,
                 bias_nums,
                 disable_coords=False,
                 out_channels=1):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.channels = channels
        self.mask_out_stride = mask_out_stride
        self.disable_coords = disable_coords
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.out_channels = out_channels

    def forward(self, x, mask_head_params, is_mask=True):
        N, _, H, W = x.size()
        if not self.disable_coords:
            locations = compute_locations(x.size(2), x.size(3), stride=1, device=x.device)
            locations = locations.unsqueeze(0).permute(0, 2, 1).contiguous().float().view(1, 2, H, W)
            locations[:, 0, :, :] /= H
            locations[:, 1, :, :] /= W
            locations = locations.repeat(N, 1, 1, 1)
            x = torch.cat([locations, x], dim=1)

        weights, biases = self.parse_dynamic_params(mask_head_params, mask=is_mask)
        mask_logits = self.mask_heads_forward(x, weights, biases)
        return mask_logits

    def parse_dynamic_params(self, params, mask=True):
        assert params.dim() == 3
        assert len(self.weight_nums) == len(self.bias_nums)
        assert params.size(2) == sum(self.weight_nums) + sum(self.bias_nums)

        batch_size, num_ins = params.shape[:2]
        num_layers = len(self.weight_nums)
        params_list = list(params.split(self.weight_nums + self.bias_nums, dim=2))

        assert num_layers == 1, "If num_layers > 1, you cannot use this accelerated form of dynamic conv!"

        weight_list = params_list[:num_layers]                                                        # [(N, L, C)] * M
        bias_list = params_list[num_layers:]                                                          # [(N, L, C)] * M
        if mask:
            bias_list[-1] = bias_list[-1] + bias_value

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_list[l] = weight_list[l].reshape(batch_size, num_ins * self.channels, -1).contiguous()      # (N, L * Co, Ci)
                bias_list[l] = bias_list[l].reshape(batch_size, num_ins * self.channels, 1).contiguous()           # (N, L * Co, 1)
            else:
                weight_list[l] = weight_list[l].reshape(batch_size, num_ins * self.out_channels, -1).contiguous()  # (N, L * Co, Ci)
                bias_list[l] = bias_list[l].reshape(batch_size, num_ins * self.out_channels, 1).contiguous()       # (N, L * Co, 1)
        return weight_list, bias_list

    def mask_heads_forward(self, features, weights, biases):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        N, C, H, W = features.size()
        n_layers = len(weights)

        x = features.view(N, C, H * W)
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = w.bmm(x) + b
            if i < n_layers - 1:
                x = F.relu(x)
        x = x.view(N, -1, H, W)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CtnetHead(nn.Module):
    def __init__(self, heads, in_channels, head_channels=256):
        super(CtnetHead, self).__init__()

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_channels > 0:
                fc = nn.Sequential(
                    nn.Linear(in_channels, head_channels, bias=True),
                    nn.ReLU(inplace=False),
                    nn.Linear(head_channels, classes, bias=True))
                if 'logits' in head:
                    fc[-1].bias.data.fill_(bias_value)
                else:
                    self.fill_fc_weights(fc)
            else:
                fc = nn.Linear(in_channels, classes, bias=True)
                if 'logits' in head:
                    fc.bias.data.fill_(bias_value)
                else:
                    self.fill_fc_weights(fc)
            self.__setattr__(head, fc)

    @staticmethod
    def fill_fc_weights(layers):
        for m in layers.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return z

    def init_weights(self):
        # ctnet_head will init weights during building
        pass


class CondLSTR3D(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 head_layers=1,
                 disable_coords=False,
                 branch_channels=64,
                 min_points=5,
                 line_width=13,
                 score_thresh=0.3,
                 eos_coef=0.1,
                 mask_downscale=4,
                 with_smooth=False,
                 with_2d=True):
        super(CondLSTR3D, self).__init__()
        self.num_classes = num_classes
        self.min_points = min_points
        self.line_width = line_width
        self.score_thresh = score_thresh
        self.mask_downscale = mask_downscale
        self.with_2d = with_2d

        # parameters for bev detection
        self.mask_weight_nums_bev, self.mask_bias_nums_bev = cal_num_params(
            head_layers, disable_coords, in_channels[0], branch_channels, out_channels=1)
        self.num_mask_params_bev = sum(self.mask_weight_nums_bev) + sum(self.mask_bias_nums_bev)

        self.reg_weight_nums_bev, self.reg_bias_nums_bev = cal_num_params(
            head_layers, disable_coords, in_channels[0], branch_channels, out_channels=2)
        self.num_reg_params_bev = sum(self.reg_weight_nums_bev) + sum(self.reg_bias_nums_bev)

        self.num_gen_params_bev = self.num_mask_params_bev + self.num_reg_params_bev

        # heads for bev detection
        self.mask_head_bev = DynamicMaskHead(
            head_layers,
            in_channels[0],
            branch_channels,
            1,
            self.mask_weight_nums_bev,
            self.mask_bias_nums_bev,
            disable_coords=disable_coords,
            out_channels=1)

        self.reg_head_bev = DynamicMaskHead(
            head_layers,
            in_channels[0],
            branch_channels,
            1,
            self.reg_weight_nums_bev,
            self.reg_bias_nums_bev,
            disable_coords=disable_coords,
            out_channels=2)

        if self.with_2d:
            # parameters for img detection
            self.mask_weight_nums_img, self.mask_bias_nums_img = cal_num_params(
                head_layers, disable_coords, in_channels[0], branch_channels, out_channels=1)
            self.num_mask_params_img = sum(self.mask_weight_nums_img) + sum(self.mask_bias_nums_img)

            self.reg_weight_nums_img, self.reg_bias_nums_img = cal_num_params(
                head_layers, disable_coords, in_channels[0], branch_channels, out_channels=2)
            self.num_reg_params_img = sum(self.reg_weight_nums_img) + sum(self.reg_bias_nums_img)

            self.num_gen_params_img = self.num_mask_params_img + self.num_reg_params_img

            # heads for img detection
            self.mask_head_img = DynamicMaskHead(
                head_layers,
                in_channels[0],
                branch_channels,
                1,
                self.mask_weight_nums_img,
                self.mask_bias_nums_img,
                disable_coords=disable_coords,
                out_channels=1)

            self.reg_head_img = DynamicMaskHead(
                head_layers,
                in_channels[0],
                branch_channels,
                1,
                self.reg_weight_nums_img,
                self.reg_bias_nums_img,
                disable_coords=disable_coords,
                out_channels=2)

        heads = {
            'logits_obj': 2,
            'logits_cls': self.num_classes,
            'params_bev': self.num_gen_params_bev,
            'ranges_bev': 2,
        }
        if self.with_2d:
            heads.update({
                'params_img': self.num_gen_params_img,
                'ranges_img': 2,
            })

        self.ctnet_head = CtnetHead(
            heads,
            in_channels=in_channels[1],
            head_channels=branch_channels)

        loss_weights = dict(
            obj_weight=10.,
            cls_weight=10.,
            loc_weight=1.,
            reg_weight=1.,
            rng_weight=20.)

        self.loss = CondLSTR3DLoss(
            min_points=self.min_points,
            line_width=self.line_width,
            weight_dict=loss_weights,
            eos_coef=eos_coef,
            with_2d=with_2d)

        self.postprocess = CondLSTR3DPost(
            min_points=self.min_points,
            line_width=self.line_width,
            score_thresh=self.score_thresh,
            mask_downscale=self.mask_downscale,
            with_smooth=with_smooth,
            with_2d=False)

    def _forward(self, f_bev, f_img, f_ins, bev_shape, img_shape):
        # f_ins:  (N, L, C)
        z = self.ctnet_head(f_ins)
        logits_obj, logits_cls = z['logits_obj'], z['logits_cls']                                              # (N, L, 2), (N, L, C)
        params_bev, ranges_bev = z['params_bev'], z['ranges_bev']                                              # (N, L, Ci * Co), (N, L, 2)

        mask_params_bev = params_bev[:, :, :self.num_mask_params_bev]                                          # (N, L, Ci * Co)
        reg_params_bev = params_bev[:, :, self.num_mask_params_bev:]                                           # (N, L, Ci * Co)

        masks_bev = self.mask_head_bev(f_bev, mask_params_bev, is_mask=True)                                   # (N, L, H, W)
        regs_bev = self.reg_head_bev(f_bev, reg_params_bev, is_mask=False)                                     # (N, L * 2, H, W)

        ranges_bev = ranges_bev.sigmoid()                                                                      # (N, L, 2)
        mask_shape = (int(bev_shape[0] // self.mask_downscale), int(bev_shape[1] // self.mask_downscale))
        masks_bev = F.interpolate(masks_bev, size=mask_shape, mode='bilinear', align_corners=True)             # (N, L, H, W)
        regs_bev = F.interpolate(regs_bev, size=mask_shape, mode='bilinear', align_corners=True)               # (N, L, H, W)

        if self.with_2d:
            params_img, ranges_img = z['params_img'], z['ranges_img']                                          # (N, L, Ci * Co), (N, L, 2)

            mask_params_img = params_img[:, :, :self.num_mask_params_img]                                      # (N, L, Ci * Co)
            reg_params_img = params_img[:, :, self.num_mask_params_img:]                                       # (N, L, Ci * Co)

            masks_img = self.mask_head_img(f_img, mask_params_img, is_mask=True)                               # (N, L, H, W)
            regs_img = self.reg_head_img(f_img, reg_params_img, is_mask=False)                                 # (N, L * 2, H, W)

            ranges_img = ranges_img.sigmoid()                                                                  # (N, L, 2)
            mask_shape = (int(img_shape[0] // self.mask_downscale), int(img_shape[1] // self.mask_downscale))
            masks_img = F.interpolate(masks_img, size=mask_shape, mode='bilinear', align_corners=True)         # (N, L, H, W)
            regs_img = F.interpolate(regs_img, size=mask_shape, mode='bilinear', align_corners=True)           # (N, L * 2, H, W)
        else:
            masks_img = None
            regs_img = None
            ranges_img = None
        return logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img, ranges_bev, ranges_img

    def forward_raw(self, f_bev, f_img, f_ins, bev_shape, img_shape):
        if isinstance(f_ins, (list, tuple)):
            if self.training:
                logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img, ranges_bev, ranges_img = \
                    list(zip(*[self._forward(f_b, f_i, f_s, bev_shape, img_shape)
                               for (f_b, f_i, f_s) in zip(f_bev, f_img, f_ins)]))
            else:
                logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img, ranges_bev, ranges_img = \
                    self._forward(f_bev[-1], f_img[-1], f_ins[-1], bev_shape, img_shape)
        else:
            logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img, ranges_bev, ranges_img = \
                self._forward(f_bev, f_img, f_ins, bev_shape, img_shape)
        return logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img, ranges_bev, ranges_img

    def forward(self, f_bev, f_img, f_ins, img_metas, gt_masks_bev=None, gt_masks_img=None, gt_labels=None):
        img_shape = img_metas[0]['input_shape']
        bev_shape = img_metas[0]['bev_shape']
        logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img, ranges_bev, ranges_img = \
            self.forward_raw(f_bev, f_img, f_ins, bev_shape, img_shape)

        if self.training:
            loss_obj, loss_cls, loss_loc, loss_reg, loss_rng = self.loss(
                logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img, ranges_bev, ranges_img,
                gt_masks_bev, gt_masks_img, gt_labels, img_metas)
            loss_dict = {
                'obj': loss_obj,
                'cls': loss_cls,
                'reg': loss_reg,
                'loc': loss_loc,
                'rng': loss_rng,
            }
            return loss_dict
        else:
            lane_points, lane_attris, lane_scores = self.postprocess(
                logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img,
                ranges_bev, ranges_img, img_metas)
            pred_dict = {
                'lane_points': lane_points,
                'lane_attris': lane_attris,
                'lane_scores': lane_scores,
            }
            return pred_dict


# class DynamicMaskHead(nn.Module):
#
#     def __init__(self,
#                  num_layers,
#                  channels,
#                  in_channels,
#                  mask_out_stride,
#                  weight_nums,
#                  bias_nums,
#                  disable_coords=False,
#                  out_channels=1):
#         super(DynamicMaskHead, self).__init__()
#         self.num_layers = num_layers
#         self.channels = channels
#         self.in_channels = in_channels
#         self.mask_out_stride = mask_out_stride
#         self.disable_coords = disable_coords
#         self.weight_nums = weight_nums
#         self.bias_nums = bias_nums
#         self.num_gen_params = sum(weight_nums) + sum(bias_nums)
#         self.out_channels = out_channels
#
#     def forward(self, x, mask_head_params, num_ins, is_mask=True):
#         N, _, H, W = x.size()
#         if not self.disable_coords:
#             locations = compute_locations(x.size(2), x.size(3), stride=1, device=x.device)
#             locations = locations.unsqueeze(0).permute(0, 2, 1).contiguous().float().view(1, 2, H, W)
#             locations[:, 0, :, :] /= H
#             locations[:, 1, :, :] /= W
#             locations = locations.repeat(N, 1, 1, 1)
#             x = torch.cat([locations, x], dim=1)
#
#         mask_head_inputs = []
#         for idx in range(N):
#             mask_head_inputs.append(x[idx:idx + 1, ...].repeat(1, num_ins, 1, 1))
#         mask_head_inputs = torch.cat(mask_head_inputs, 1)                                    # (1, N * L * C, H, W)
#
#         num_insts = num_ins * N
#         weights, biases = self.parse_dynamic_params(mask_head_params, mask=is_mask)
#
#         mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, num_insts)  # (1, N * L * C, H, W)
#         mask_logits = mask_logits.view(N, -1, H, W)                                          # (N, L * C, H, W)
#         return mask_logits
#
#     def parse_dynamic_params(self, params, mask=True):
#         assert params.dim() == 2
#         assert len(self.weight_nums) == len(self.bias_nums)
#         assert params.size(1) == sum(self.weight_nums) + sum(self.bias_nums)
#
#         num_insts = params.size(0)                                                           # (N * L,)
#         num_layers = len(self.weight_nums)
#         params_splits = list(params.split(self.weight_nums + self.bias_nums, dim=1))
#
#         weight_splits = params_splits[:num_layers]                                           # [(N * L, C)] * M
#         bias_splits = params_splits[num_layers:]                                             # [(N * L, C)] * M
#         if mask:
#             bias_splits[-1] = bias_splits[-1] - 2.19
#
#         # out_channels x in_channels x 1 x 1
#         for l in range(num_layers):
#             if l < num_layers - 1:
#                 weight_splits[l] = weight_splits[l].reshape(num_insts * self.channels, -1, 1, 1)      # (N * L * C1, C2, 1, 1)
#                 bias_splits[l] = bias_splits[l].reshape(num_insts * self.channels)                    # (N * L * C1)
#             else:
#                 weight_splits[l] = weight_splits[l].reshape(num_insts * self.out_channels, -1, 1, 1)  # (N * L * C1, C2, 1, 1)
#                 bias_splits[l] = bias_splits[l].reshape(num_insts * self.out_channels)                # (N * L * C1)
#         return weight_splits, bias_splits
#
#     def mask_heads_forward(self, features, weights, biases, num_insts):
#         '''
#         :param features
#         :param weights: [w0, w1, ...]
#         :param bias: [b0, b1, ...]
#         :return:
#         '''
#         assert features.dim() == 4
#         n_layers = len(weights)
#         x = features
#         for i, (w, b) in enumerate(zip(weights, biases)):
#             x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
#             if i < n_layers - 1:
#                 x = F.relu(x)
#         return x
