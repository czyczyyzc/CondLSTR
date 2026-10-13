import math
import copy
import torch
import PIL.Image
import PIL.ImageDraw
import numpy as np
from functools import cmp_to_key
from shapely.geometry import Polygon, LineString, MultiLineString


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_label(mask,
               polygon_in,
               val,
               shape_type='polygon',
               width=3,
               convert=False):
    polygon = copy.deepcopy(polygon_in)
    mask = PIL.Image.fromarray(mask)
    xy = []
    if convert:
        for i in range(len(polygon) // 2):
            xy.append((polygon[2 * i], polygon[2 * i + 1]))
    else:
        for i in range(len(polygon)):
            xy.append((polygon[i][0], polygon[i][1]))

    if shape_type == 'polygon':
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=val, fill=val)
    else:
        PIL.ImageDraw.Draw(mask).line(xy=xy, fill=val, width=width)
    mask = np.array(mask, dtype=np.uint8)
    return mask


def cal_num_params(num_layers,
                   disable_coords,
                   in_channels,
                   channels,
                   out_channels=1):
    weight_nums, bias_nums = [], []
    for l in range(num_layers):
        if l == num_layers - 1:
            if num_layers == 1 and not disable_coords:
                weight_nums.append((in_channels + 2) * out_channels)
            else:
                weight_nums.append(in_channels * out_channels)
            bias_nums.append(out_channels)
        elif l == 0:
            if not disable_coords:
                weight_nums.append((in_channels + 2) * channels)
            else:
                weight_nums.append(in_channels * channels)
            bias_nums.append(channels)
        else:
            weight_nums.append(channels * channels)
            bias_nums.append(channels)
    return weight_nums, bias_nums


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(
        0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def select_mask_points(ct, r, shape, device, max_sample=5):
    h, w = shape[:2]
    r = max((r.item() // 2), 1)
    start_x, end_x = max(ct[0].item() - r, 0), min(ct[0].item() + r, w - 1)
    start_y, end_y = max(ct[1].item() - r, 0), min(ct[1].item() + r, h - 1)

    range_x = torch.arange(start_x, end_x + 1, device=device)
    range_y = torch.arange(start_y, end_y + 1, device=device)
    point_y, point_x = torch.meshgrid(range_y, range_x, indexing='ij')
    point_m = torch.stack([point_x, point_y], dim=-1)                         # (M, 2)
    point_m = point_m.reshape(-1, 2)                                          # (M, 2)
    point_c = ct.reshape(1, 2)                                                # (1, 2)
    dis_a_c = torch.cdist(point_m.float(), point_c.float(), p=2.0)[:, 0]      # (M,)
    dis_kep = (dis_a_c > 0) & (dis_a_c <= r + 0.1)                            # (M,)
    point_m = point_m[dis_kep]                                                # (K, 2)
    if len(point_m) > max_sample - 1:
        idx_kep = np.random.choice(len(point_m), max_sample - 1, replace=False)
        point_m = point_m[idx_kep]
    point_m = torch.cat([point_m, point_c], dim=0)                            # (K, 2)
    return point_m


# def select_mask_points(ct, r, shape, max_sample=5):
#     h, w = shape[:2]
#     r = max(int(r // 2), 1)
#     start_x, end_x = max(ct[0] - r, 0), min(ct[0] + r, w - 1)
#     start_y, end_y = max(ct[1] - r, 0), min(ct[1] + r, h - 1)
#
#     range_x = np.arange(start_x, end_x + 1)
#     range_y = np.arange(start_y, end_y + 1)
#     point_x, point_y = np.meshgrid(range_x, range_y)
#     point_m = np.stack([point_x, point_y], axis=-1)
#     point_m = point_m.reshape(-1, 2)                       # (M, 2)
#     point_c = np.array(ct).reshape(1, 2)                   # (1, 2)
#     dis_a_c = point_m - point_c                            # (M, 2)
#     dis_a_c = np.sqrt((dis_a_c * dis_a_c).sum(axis=1))     # (M,)
#     dis_kep = (dis_a_c > 0) & (dis_a_c <= r + 0.1)         # (M,)
#     point_m = point_m[dis_kep]                             # (K, 2)
#     if len(point_m) > max_sample - 1:
#         idx_kep = np.random.choice(len(point_m), max_sample - 1, replace=False)
#         point_m = point_m[idx_kep]
#     point_m = np.concatenate([point_m, point_c], axis=0)
#     return point_m


# def select_mask_points(ct, r, shape, max_sample=5):
#
#     def in_range(pt, w, h):
#         if 0 <= pt[0] < w and 0 <= pt[1] < h:
#             return True
#         else:
#             return False
#
#     h, w = shape[:2]
#     valid_points = []
#     r = max(int(r // 2), 1)
#     start_x, end_x = ct[0] - r, ct[0] + r
#     start_y, end_y = ct[1] - r, ct[1] + r
#     for x in range(start_x, end_x + 1):
#         for y in range(start_y, end_y + 1):
#             if x == ct[0] and y == ct[1]:
#                 continue
#             if in_range((x, y), w, h) and cal_dis((x, y), ct) <= r + 0.1:
#                 valid_points.append([x, y])
#     if len(valid_points) > max_sample - 1:
#         valid_points = random.sample(valid_points, max_sample - 1)
#     valid_points.append([ct[0], ct[1]])
#     return valid_points


def cal_dis(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def min_dis_one_point(points, idx):
    min_dis = 1e6
    for i in range(len(points)):
        if i == idx:
            continue
        else:
            d = cal_dis(points[idx], points[i])
            if d < min_dis:
                min_dis = d
    return min_dis


def get_line_intersection(x, y, line):
    def in_line_range(val, start, end):
        s = min(start, end)
        e = max(start, end)
        if s <= val <= e and s != e:
            return True
        else:
            return False

    def choose_min_reg(val, ref):
        min_val = 1e5
        index = -1
        if len(val) == 0:
            return None
        else:
            for i, v in enumerate(val):
                if abs(v - ref) < min_val:
                    min_val = abs(v - ref)
                    index = i
        return val[index]

    reg_y = []
    reg_x = []

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(x, point_start[0], point_end[0]):
            k = (point_end[1] - point_start[1]) / (
                point_end[0] - point_start[0])
            reg_y.append(k * (x - point_start[0]) + point_start[1])
    reg_y = choose_min_reg(reg_y, y)

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(y, point_start[1], point_end[1]):
            k = (point_end[0] - point_start[0]) / (
                point_end[1] - point_start[1])
            reg_x.append(k * (y - point_start[1]) + point_start[0])
    reg_x = choose_min_reg(reg_x, x)
    return reg_x, reg_y


def convert_list(p, downscale=None):
    xy = list()
    if downscale is None:
        for i in range(len(p) // 2):
            xy.append((p[2 * i], p[2 * i + 1]))
    else:
        for i in range(len(p) // 2):
            xy.append((p[2 * i] / downscale, p[2 * i + 1] / downscale))
    return xy


def clamp_line(line, box, min_length=0):
    left, top, right, bottom = box
    loss_box = Polygon([[left, top], [right, top], [right, bottom],
                        [left, bottom]])
    line_coords = np.array(line).reshape((-1, 2))
    if line_coords.shape[0] < 2:
        return None
    try:
        line_string = LineString(line_coords)
        I = line_string.intersection(loss_box)
        if I.is_empty:
            return None
        if I.length < min_length:
            return None
        if isinstance(I, LineString):

            pts = list(I.coords)
            return pts
        elif isinstance(I, MultiLineString):
            pts = []
            Istrings = list(I)
            for Istring in Istrings:
                pts += list(Istring.coords)
            return pts
    except:
        return None


def extend_line(line, dis=10):
    extended = copy.deepcopy(line)
    start = line[1]
    end = line[0]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = math.sqrt(dx**2 + dy**2)
    dx = dx / norm
    dy = dy / norm
    extend_point = (start[0] + dx * dis, start[1] + dy * dis)
    extended.insert(0, extend_point)
    return extended


def sort_line_func(a, b):

    def get_line_intersection(y, line):

        def in_line_range(val, start, end):
            s = min(start, end)
            e = max(start, end)
            if s == e and val == s:
                return 1
            elif s <= val <= e and s != e:
                return 2
            else:
                return 0

        reg_x = []
        # 水平线的交点
        for i in range(len(line) - 1):
            point_start, point_end = line[i], line[i + 1]
            flag = in_line_range(y, point_start[1], point_end[1])
            if flag == 2:
                k = (point_end[0] - point_start[0]) / (
                    point_end[1] - point_start[1])
                reg_x.append(k * (y - point_start[1]) + point_start[0])
            elif flag == 1:
                reg_x.append((point_start[0] + point_end[0]) / 2)
        reg_x = min(reg_x)

        return reg_x

    line1 = np.array(copy.deepcopy(a))
    line2 = np.array(copy.deepcopy(b))
    line1_ymin = min(line1[:, 1])
    line1_ymax = max(line1[:, 1])
    line2_ymin = min(line2[:, 1])
    line2_ymax = max(line2[:, 1])
    if line1_ymax <= line2_ymin or line2_ymax <= line1_ymin:
        y_ref1 = (line1_ymin + line1_ymax) / 2
        y_ref2 = (line2_ymin + line2_ymax) / 2
        x_line1 = get_line_intersection(y_ref1, line1)
        x_line2 = get_line_intersection(y_ref2, line2)
    else:
        ymin = max(line1_ymin, line2_ymin)
        ymax = min(line1_ymax, line2_ymax)
        y_ref = (ymin + ymax) / 2
        x_line1 = get_line_intersection(y_ref, line1)
        x_line2 = get_line_intersection(y_ref, line2)

    if x_line1 < x_line2:
        return -1
    elif x_line1 == x_line2:
        return 0
    else:
        return 1


def nms_endpoints(lane_ends, thr):

    def cal_dis(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def search_groups(coord, groups, thr):
        for idx_group, group in enumerate(groups):
            for group_point in group:
                group_point_coord = group_point[1]
                if cal_dis(coord, group_point_coord) <= thr:
                    return idx_group
        return -1

    def update_coords(points_info, thr=4):
        groups = []
        for idx, coord in enumerate(points_info):
            idx_group = search_groups(coord, groups, thr)
            if idx_group < 0:
                groups.append([(idx, coord)])
            else:
                groups[idx_group].append((idx, coord))

        return groups

    results = []

    points = [item[0] for item in lane_ends]
    groups = update_coords(points, thr=thr)
    for group in groups:
        group_points = []
        lanes = []
        for idx, coord in group:
            group_points.append(coord)
            lanes.append(lane_ends[idx][1])
        group_points = np.array(group_points)
        center_x = (np.min(group_points[:, 0]) +
                    np.max(group_points[:, 0])) / 2
        center_y = (np.min(group_points[:, 1]) +
                    np.max(group_points[:, 1])) / 2
        center = (center_x, center_y)
        max_dis = 0
        for point in group_points:
            dis = cal_dis(center, point)
            if dis > max_dis:
                max_dis = dis
        lanes = sorted(lanes, key=cmp_to_key(sort_line_func))
        results.append([center, lanes, dis])

    return results


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
#         mask_head_inputs = []
#         for idx in range(N):
#             mask_head_inputs.append(x[idx:idx + 1, ...].repeat(1, num_ins[idx], 1, 1))
#         mask_head_inputs = torch.cat(mask_head_inputs, 1)
#         num_insts = sum(num_ins)
#         mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
#         weights, biases = self.parse_dynamic_params(mask_head_params, mask=is_mask)
#         mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, num_insts)
#         return mask_logits
#
#     def parse_dynamic_params(self, params, mask=True):
#         assert params.dim() == 2
#         assert len(self.weight_nums) == len(self.bias_nums)
#         assert params.size(1) == sum(self.weight_nums) + sum(self.bias_nums)
#
#         num_insts = params.size(0)
#         num_layers = len(self.weight_nums)
#         params_splits = list(params.split(self.weight_nums + self.bias_nums, dim=1))
#
#         weight_splits = params_splits[:num_layers]
#         bias_splits = params_splits[num_layers:]
#         if mask:
#             bias_splits[-1] = bias_splits[-1] - 2.19
#
#         # out_channels x in_channels x 1 x 1
#         for l in range(num_layers):
#             if l < num_layers - 1:
#                 weight_splits[l] = weight_splits[l].reshape(num_insts * self.channels, -1, 1, 1)
#                 bias_splits[l] = bias_splits[l].reshape(num_insts * self.channels)
#             else:
#                 weight_splits[l] = weight_splits[l].reshape(num_insts * self.out_channels, -1, 1, 1)
#                 bias_splits[l] = bias_splits[l].reshape(num_insts * self.out_channels)
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
