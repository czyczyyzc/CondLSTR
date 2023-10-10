# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of lane detection task."""
import os
import sys
import cv2
import json
import pickle
import argparse
import imagesize
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from itertools import product


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def calc_x(f, t):
    """Calc x from t.
    :param f: the param of interp
    :type f: dict
    :param t: step of interp
    :type t: int
    :return: x corrdinate
    :rtype: float
    """
    return f['a_x'] + f['b_x'] * t + f['c_x'] * t * t + f['d_x'] * t * t * t


def calc_y(f, t):
    """Calc y from t.
    :param f: the param of interp
    :type f: dict
    :param t: step of interp
    :type t: int
    :return: y corrdinate
    :rtype: float
    """
    return f['a_y'] + f['b_y'] * t + f['c_y'] * t * t + f['d_y'] * t * t * t


def spline_interp(*, lane, step_t=1):
    """Interp a line.
    :param lane: the lane to be interp
    :type lane: a list of dict
    :param step_t: the interp step
    :type step_t: int
    :return: the interp lane
    :rtype: list
    """
    interp_lane = []
    if len(lane) < 2:
        return lane
    interp_param = calc_params(lane)
    for f in interp_param:
        t = 0
        while t < f['h']:
            x = calc_x(f, t)
            y = calc_y(f, t)
            interp_lane.append({"x": x, "y": y})
            t += step_t
    interp_lane.append(lane[-1])
    return interp_lane


def calc_params(lane):
    """Calc params of a line.
    :param lane: the lane of which the param to be calculated.
    :type lane: list of dicts
    :return: param of the lane
    :rtype: list
    """
    params = []
    n_pt = len(lane)
    if n_pt < 2:
        return params
    if n_pt == 2:
        h0 = np.sqrt((lane[0]['x'] - lane[1]['x']) * (lane[0]['x'] - lane[1]['x']) +
                     (lane[0]['y'] - lane[1]['y']) * (lane[0]['y'] - lane[1]['y']))
        a_x = lane[0]['x']
        a_y = lane[0]['y']
        b_x = (lane[1]['x'] - a_x) / h0
        b_y = (lane[1]['y'] - a_y) / h0
        params.append({"a_x": a_x, "b_x": b_x, "c_x": 0, "d_x": 0, "a_y": a_y, "b_y": b_y, "c_y": 0, "d_y": 0, "h": h0})
        return params
    h = []
    for i in range(n_pt - 1):
        dx = lane[i]['x'] - lane[i + 1]['x']
        dy = lane[i]['y'] - lane[i + 1]['y']
        h.append(np.sqrt(dx * dx + dy * dy))
    A = []
    B = []
    C = []
    D_x = []
    D_y = []
    for i in range(n_pt - 2):
        A.append(h[i])
        B.append(2 * (h[i] + h[i + 1]))
        C.append(h[i + 1])
        dx1 = (lane[i + 1]['x'] - lane[i]['x']) / h[i]
        dx2 = (lane[i + 2]['x'] - lane[i + 1]['x']) / h[i + 1]
        tmpx = 6 * (dx2 - dx1)
        dy1 = (lane[i + 1]['y'] - lane[i]['y']) / h[i]
        dy2 = (lane[i + 2]['y'] - lane[i + 1]['y']) / h[i + 1]
        tmpy = 6 * (dy2 - dy1)
        if i == 0:
            C[i] /= B[i]
            D_x.append(tmpx / B[i])
            D_y.append(tmpy / B[i])
        else:
            base_v = B[i] - A[i] * C[i - 1]
            C[i] /= base_v
            D_x.append((tmpx - A[i] * D_x[i - 1]) / base_v)
            D_y.append((tmpy - A[i] * D_y[i - 1]) / base_v)

    Mx = np.zeros(n_pt)
    My = np.zeros(n_pt)
    Mx[n_pt - 2] = D_x[n_pt - 3]
    My[n_pt - 2] = D_y[n_pt - 3]
    for i in range(n_pt - 4, -1, -1):
        Mx[i + 1] = D_x[i] - C[i] * Mx[i + 2]
        My[i + 1] = D_y[i] - C[i] * My[i + 2]

    Mx[0] = 0
    Mx[-1] = 0
    My[0] = 0
    My[-1] = 0

    for i in range(n_pt - 1):
        a_x = lane[i]['x']
        b_x = (lane[i + 1]['x'] - lane[i]['x']) / h[i] - (2 * h[i] * Mx[i] + h[i] * Mx[i + 1]) / 6
        c_x = Mx[i] / 2
        d_x = (Mx[i + 1] - Mx[i]) / (6 * h[i])

        a_y = lane[i]['y']
        b_y = (lane[i + 1]['y'] - lane[i]['y']) / h[i] - (2 * h[i] * My[i] + h[i] * My[i + 1]) / 6
        c_y = My[i] / 2
        d_y = (My[i + 1] - My[i]) / (6 * h[i])

        params.append(
            {"a_x": a_x, "b_x": b_x, "c_x": c_x, "d_x": d_x, "a_y": a_y, "b_y": b_y, "c_y": c_y, "d_y": d_y, "h": h[i]})

    return params


def resize_lane(lane, x_ratio, y_ratio):
    """Resize the coordinate of a lane accroding image resize ratio.
    :param lane: the lane need to be resized
    :type lane: a list of dicts
    :param x_ratio: correspond image resize ratio in x axes.
    :type x_ratio: float
    :param y_ratio: correspond image resize ratio in y axes.
    :type y_ratio: float
    :return: resized lane
    :rtype: list
    """
    return [{"x": float(p['x']) / x_ratio, "y": float(p['y']) / y_ratio} for p in lane]


def calc_iou(lane1, lane2, hyperp):
    """Calc iou of two lane.
    :param lane1: the first lane to be calc.
    :type lane1: list of dict.
    :param lane2: the first lane to be calc.
    :type lane2: list of dict.
    :return: iou ratio.
    :rtype: float
    """
    new_height = hyperp['eval_height']
    new_width = hyperp['eval_width']
    lane_width = hyperp['lane_width']

    im1 = np.zeros((new_height, new_width), np.uint8)
    im2 = np.zeros((new_height, new_width), np.uint8)
    interp_lane1 = spline_interp(lane=lane1, step_t=1)
    interp_lane2 = spline_interp(lane=lane2, step_t=1)
    for i in range(0, len(interp_lane1) - 1):
        cv2.line(im1, (int(interp_lane1[i]['x']), int(interp_lane1[i]['y'])),
                 (int(interp_lane1[i + 1]['x']), int(interp_lane1[i + 1]['y'])), 255, lane_width)
    for i in range(0, len(interp_lane2) - 1):
        cv2.line(im2, (int(interp_lane2[i]['x']), int(interp_lane2[i]['y'])),
                 (int(interp_lane2[i + 1]['x']), int(interp_lane2[i + 1]['y'])), 255, lane_width)
    union_im = cv2.bitwise_or(im1, im2)
    union_sum = union_im.sum()
    intersection_sum = im1.sum() + im2.sum() - union_sum
    if union_sum == 0:
        return 0
    else:
        return intersection_sum / float(union_sum)


def evaluate_core(*, gt_lanes, pr_lanes, gt_wh, pr_wh, hyperp):
    """Core function of evaluate for every image.
    :param gt_lanes: groundtruth lanes of an image
    :type gt_lanes:  a list of lanes in an image
    :param pr_lanes: predict lanes of an image
    :type pr_lanes: a list of lanes in an image
    :return: a dict contain a series of parameters, which is:
            gt_num: groundtruth lanes number of an image
            pr_num: predict lanes number of an image
            hit_num: the matched number of groundtruth and predict lanes
            gt_curr_num: groundtruth current lanes number of an image
            pr_curr_num: predict current lanes number of an image
            hit_curr_num: the matched number of groundtruth and predict lanes in current domin
            left_error: the error of current left matched lane in x axes
            right_error: the error of current right matched lane in x axes
            census_error: the error of matched lane in x axes
    :rtype: dict
    """
    gt_num = len(gt_lanes)
    pr_num = len(pr_lanes)
    hit_num = 0
    pr_list = [False for i in range(pr_num)]
    gt_list = [False for i in range(gt_num)]

    if gt_num > 0 and pr_num > 0:
        iou_thresh = hyperp['iou_thresh']
        new_height = hyperp['eval_height']
        new_width = hyperp['eval_width']

        gt_y_ratio = np.true_divide(gt_wh['height'], new_height)
        gt_x_ratio = np.true_divide(gt_wh['width'], new_width)
        pr_y_ratio = np.true_divide(pr_wh['height'], new_height)
        pr_x_ratio = np.true_divide(pr_wh['width'], new_width)
        # resize lanes and interp lanes,
        # all the gt and pr are mapping to src img, so the scale ratio is same,
        # note that the scale ratio is not a factor but a divisor
        # print('gt_lane',gt_lanes)
        gt_lanes = list(map(lambda lane: resize_lane(lane, gt_x_ratio, gt_y_ratio), gt_lanes))
        pr_lanes = list(map(lambda lane: resize_lane(lane, pr_x_ratio, pr_y_ratio), pr_lanes))

        sorted_gt_lanes = gt_lanes
        sorted_pr_lanes = pr_lanes
        iou_mat = np.zeros((gt_num, pr_num))

        for (index_gt, gt_lane), (index_pr, pr_lane) in product(enumerate(sorted_gt_lanes), enumerate(sorted_pr_lanes)):
            iou_mat[index_gt][index_pr] = calc_iou(gt_lane, pr_lane, hyperp)

        # match_idx = Munkres().compute(make_cost_matrix(iou_mat, lambda iou: float(1.0 - iou)))
        cost_matrix = 1 - np.array(iou_mat)
        match_index_list = linear_sum_assignment(cost_matrix)

        for gt_index, pr_index in zip(*match_index_list):
            iou_val = iou_mat[gt_index][pr_index]
            if iou_val > iou_thresh:
                hit_num += 1
                pr_list[pr_index] = True
                gt_list[gt_index] = True
    return dict(gt_num=gt_num, pr_num=pr_num, hit_num=hit_num, pr_list=pr_list, gt_list=gt_list)


class LaneMetricCore(object):
    """Save and summary metric for lane metric."""

    def __init__(self, *, eval_width, eval_height, iou_thresh, lane_width, prob_thresh=None):
        """Initialize metric params, we use bitwise IOU as the judgment condition.
        :param eval_width
        :param eval_height
        :param iou_thresh
        :param lane_width
        :param thresh_list
        :type list
        """
        self.eval_params = dict(
            eval_width=eval_width,
            eval_height=eval_height,
            iou_thresh=iou_thresh,
            lane_width=lane_width,
        )
        self.prob_thresh = prob_thresh
        self.result_record = []
        self.results = []

    def __call__(self, gt_result, pr_result, *args, **kwargs):
        """Append input into result record cache.
        :param output: output data
        :param target: target data
        :return:
        """
        prob_thresh = self.prob_thresh
        predict_spec = pr_result
        target_spec = gt_result

        gt_wh = target_spec['Shape']
        pr_wh = predict_spec['Shape']
        gt_lanes = []
        for line_spec in target_spec['Lines']:
            if len(line_spec) > 0:
                gt_lanes.append(line_spec)

        pr_lanes = []
        for line_spec in predict_spec['Lines']:
            if 'score' in line_spec:
                if float(line_spec['score']) > prob_thresh:
                    line_spec = line_spec['points']
                else:
                    line_spec = []
            if len(line_spec) > 0:
                pr_lanes.append(line_spec)

        result = evaluate_core(gt_lanes=gt_lanes, pr_lanes=pr_lanes, gt_wh=gt_wh, pr_wh=pr_wh, hyperp=self.eval_params)

        self.result_record.append(result)

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.result_record = []

    def summary(self):
        """Summary all record from result cache, and get performance."""
        hit_num = sum(result['hit_num'] for result in self.result_record)
        pr_num = sum(result['pr_num'] for result in self.result_record)
        gt_num = sum(result['gt_num'] for result in self.result_record)
        precision = hit_num / (pr_num + sys.float_info.epsilon)
        recall = hit_num / (gt_num + sys.float_info.epsilon)
        f1_measure = 2 * precision * recall / (precision + recall + sys.float_info.epsilon)
        return dict(f1_measure=f1_measure, precision=precision, recall=recall, pr_num=pr_num, gt_num=gt_num)


def main():
    args = parse_args()

    if args.translate:
        with open(os.path.join(args.pred_dir, 'results.pkl'), 'rb') as f:
            results_list = pickle.load(f)
        for result in tqdm(results_list):
            res_path = os.path.join(args.pred_dir,
                                    result['image_name'].replace('.jpg', '.lines.json'))
            ann_path = os.path.join(args.anno_dir, 'labels',
                                    result['image_name'].replace('.jpg', '.lines.json'))
            img_path = os.path.join(args.anno_dir, 'images', result['image_name'])
            if not (os.path.exists(ann_path) and os.path.exists(img_path)):
                print(ann_path)
                print(img_path)
                continue

            res_list = result['2d_res']
            if args.show:
                vis_path = os.path.join(args.pred_dir, 'shows', result['image_name'])
                os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                img = cv2.imread(img_path)
                for idx, lane in enumerate(res_list):
                    lane = [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                    lane = np.array(lane).round().astype(np.int32)
                    cv2.polylines(img, [lane], False, COLORS[idx + 1], thickness=args.line_width)
                cv2.imwrite(vis_path, img)

            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            with open(res_path, 'w') as f:
                data_dict = {'Lines': []}
                for lane in res_list:
                    lane = list(map(str, lane))
                    lane = [{'x': lane[i], 'y': lane[i + 1]} for i in range(0, len(lane), 2)]
                    data_dict['Lines'].append(lane)
                json.dump(data_dict, f)

    evaluator = LaneMetricCore(
        eval_width=args.eval_width,
        eval_height=args.eval_height,
        iou_thresh=0.5,
        lane_width=args.lane_width)

    print('#########################################')
    print(args.pred_dir)
    print(args.anno_dir)

    for list_path in args.list:
        with open(list_path, 'r') as f:
            for line in tqdm(f.readlines()):
                img_name = line.strip().replace('images/', '')
                res_path = os.path.join(args.pred_dir, img_name.replace('.jpg', '.lines.json'))
                ann_path = os.path.join(args.anno_dir, 'labels', img_name.replace('.jpg', '.lines.json'))
                img_path = os.path.join(args.anno_dir, 'images', img_name)
                try:
                    with open(res_path, 'r') as f:
                        pred = json.load(f)['Lines']
                    with open(ann_path, 'r') as f:
                        anno = json.load(f)['Lines']

                    width, height = imagesize.get(img_path)

                    gt_wh = dict(height=height, width=width)
                    predict_spec = dict(Lines=pred, Shape=gt_wh)
                    target_spec = dict(Lines=anno, Shape=gt_wh)
                    evaluator(target_spec, predict_spec)
                except FileNotFoundError:
                    print(res_path)
                    print(ann_path)
                    continue

    print(evaluator.summary())


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CurveLanes's metric")
    parser.add_argument(
        "--pred_dir",
        default='/data/chenziye/HDMapNet/output/test_cond_lstr_20lane_50epoch_curvelanes_new_vote_raw_s0.7',
        help="Path to directory containing the predicted lanes"
    )
    parser.add_argument(
        "--anno_dir",
        default='/data/sets/curvelanes/valid/',
        help="Path to directory containing the annotated lanes"
    )
    parser.add_argument("--lane-width",
                        type=int,
                        default=5,
                        help="Width of the lane")
    parser.add_argument("--eval-width",
                        type=int,
                        default=224,
                        help="Width of the lane")
    parser.add_argument("--eval-height",
                        type=int,
                        default=224,
                        help="Width of the lane")
    parser.add_argument("--list",
                        nargs='+',
                        default=['s3://czyczyyzc/datasets/curvelanes/valid.txt'],
                        help="Path to txt file containing the list of files"
    )
    parser.add_argument("--line-width",
                        type=int,
                        default=13,
                        help="Width of the lane line for drawing")
    parser.add_argument('--translate', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    main()


# from functools import partial
# from scipy.interpolate import splprep, splev
# from scipy.optimize import linear_sum_assignment
# from shapely.geometry import LineString, Polygon
#
#
# def draw_lane(lane, img=None, img_shape=None, width=30):
#     if img is None:
#         img = np.zeros(img_shape, dtype=np.uint8)
#     lane = lane.astype(np.int32)
#     for p1, p2 in zip(lane[:-1], lane[1:]):
#         cv2.line(img,
#                  tuple(p1),
#                  tuple(p2),
#                  color=(255, 255, 255),
#                  thickness=width)
#     return img
#
#
# def discrete_cross_iou(xs, ys, width=30, img_shape=(1440, 2560, 3)):
#     xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
#     ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]
#
#     ious = np.zeros((len(xs), len(ys)))
#     for i, x in enumerate(xs):
#         for j, y in enumerate(ys):
#             ious[i, j] = (x & y).sum() / (x | y).sum()
#     return ious
#
#
# def continuous_cross_iou(xs, ys, width=30, img_shape=(1440, 2560, 3)):
#     h, w, _ = img_shape
#     image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
#     xs = [
#         LineString(lane).buffer(distance=width / 2., cap_style=1,
#                                 join_style=2).intersection(image)
#         for lane in xs
#     ]
#     ys = [
#         LineString(lane).buffer(distance=width / 2., cap_style=1,
#                                 join_style=2).intersection(image)
#         for lane in ys
#     ]
#
#     ious = np.zeros((len(xs), len(ys)))
#     for i, x in enumerate(xs):
#         for j, y in enumerate(ys):
#             ious[i, j] = x.intersection(y).area / x.union(y).area
#
#     return ious
#
#
# def interp(points, n=50):
#     x = [x for x, _ in points]
#     y = [y for _, y in points]
#     tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))
#
#     u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
#     return np.array(splev(u, tck)).T
#
#
# def curvelanes_metric(pred,
#                       anno,
#                       width=30,
#                       iou_thresholds=[0.5],
#                       official=True,
#                       img_shape=(1440, 2560, 3)):
#     _metric = {}
#     for thr in iou_thresholds:
#         tp = 0
#         fp = 0 if len(anno) != 0 else len(pred)
#         fn = 0 if len(pred) != 0 else len(anno)
#         _metric[thr] = [tp, fp, fn]
#
#     interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred],
#                            dtype=object)  # (4, 50, 2)
#     interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno],
#                            dtype=object)  # (4, 50, 2)
#
#     if official:
#         ious = discrete_cross_iou(interp_pred,
#                                   interp_anno,
#                                   width=width,
#                                   img_shape=img_shape)
#     else:
#         ious = continuous_cross_iou(interp_pred,
#                                     interp_anno,
#                                     width=width,
#                                     img_shape=img_shape)
#
#     row_ind, col_ind = linear_sum_assignment(1 - ious)
#
#     _metric = {}
#     for thr in iou_thresholds:
#         tp = int((ious[row_ind, col_ind] > thr).sum())
#         fp = len(pred) - tp
#         fn = len(anno) - tp
#         _metric[thr] = [tp, fp, fn]
#     return _metric
#
#
# def load_curvelanes_img_data(path):
#     with open(path, 'r') as f:
#         data = json.load(f)['Lines']
#
#     lanes = [[(float(point['x']), float(point['y'])) for point in lane] for lane in data]
#     lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
#     lanes = [lane for lane in lanes if len(lane) >= 2]  # remove lanes with less than 2 points
#     lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
#     return lanes
#
#
# def load_curvelanes_data(data_dir, file_list_path):
#     with refile.smart_open(file_list_path, 'r') as file_list:
#         filepaths = [
#             os.path.join(
#                 data_dir, line.strip().replace('images/', '').replace(
#                     '.jpg', '.lines.json')) for line in file_list.readlines()
#         ]
#
#     data = []
#     for path in filepaths:
#         try:
#             img_data = load_curvelanes_img_data(path)
#         except FileNotFoundError:
#             print(path)
#             continue
#         data.append(img_data)
#
#     return data
#
#
# def eval_predictions(pred_dir,
#                      anno_dir,
#                      list_path,
#                      iou_thresholds=[0.5],
#                      width=30,
#                      official=True,
#                      sequential=False):
#     logger = logging.getLogger(__name__)
#     logger.info('Calculating metric for List: {}'.format(list_path))
#
#     predictions = load_curvelanes_data(pred_dir, list_path)
#     annotations = load_curvelanes_data(anno_dir, list_path)
#     img_shape = (1440, 2560, 3)
#     if sequential:
#         results = map(
#             partial(curvelanes_metric,
#                     width=width,
#                     official=official,
#                     iou_thresholds=iou_thresholds,
#                     img_shape=img_shape), predictions, annotations)
#     else:
#         from multiprocessing import Pool, cpu_count
#         from itertools import repeat
#         with Pool(cpu_count()) as p:
#             results = p.starmap(curvelanes_metric, zip(predictions, annotations,
#                                                        repeat(width),
#                                                        repeat(iou_thresholds),
#                                                        repeat(official),
#                                                        repeat(img_shape)))
#
#     mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
#     ret = {}
#     for thr in iou_thresholds:
#         tp = sum(m[thr][0] for m in results)
#         fp = sum(m[thr][1] for m in results)
#         fn = sum(m[thr][2] for m in results)
#         precision = float(tp) / (tp + fp) if tp != 0 else 0
#         recall = float(tp) / (tp + fn) if tp != 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if tp !=0 else 0
#         logger.info('iou thr: {:.2f}, tp: {}, fp: {}, fn: {}, precision: {}, recall: {}, f1: {}'.format(
#             thr, tp, fp, fn, precision, recall, f1))
#         mean_f1 += f1 / len(iou_thresholds)
#         mean_prec += precision / len(iou_thresholds)
#         mean_recall += recall / len(iou_thresholds)
#         total_tp += tp
#         total_fp += fp
#         total_fn += fn
#         ret[thr] = {
#             'TP': tp,
#             'FP': fp,
#             'FN': fn,
#             'Precision': precision,
#             'Recall': recall,
#             'F1': f1
#         }
#     if len(iou_thresholds) > 2:
#         logger.info('mean result, total_tp: {}, total_fp: {}, total_fn: {}, '
#                     'precision: {}, recall: {}, f1: {}'.format(
#             total_tp, total_fp, total_fn, mean_prec, mean_recall, mean_f1))
#         ret['mean'] = {
#             'TP': total_tp,
#             'FP': total_fp,
#             'FN': total_fn,
#             'Precision': mean_prec,
#             'Recall': mean_recall,
#             'F1': mean_f1
#         }
#     return ret
