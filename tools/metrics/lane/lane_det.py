import os
import json
import refile
import pickle
import jsonlines
import torch
import numpy as np
from tqdm import tqdm
from itertools import product
from scipy.optimize import linear_sum_assignment
from chamfer_distance import ChamferDistance


class LaneDetMetric(object):

    def __init__(self, distance_thr=10, num_classes=1):
        self.lane_distance = LaneDistance(distance_thr)

    def add_batch(self, pred_points_list, gt_points_list):
        self.lane_distance.add_batch(pred_points_list, gt_points_list)

    def dump_state(self):
        state = np.array([])
        state = self.lane_distance.dump_state(state)
        return state

    def load_state(self, state):
        state = self.lane_distance.load_state(state)
        assert len(state) == 0

    def __call__(self):
        F1, recall, acc, chamfer_dist = self.lane_distance()
        eval_dict = {'F1': F1, 'recall': recall, 'acc': acc, 'chamfer_dist': chamfer_dist}
        eval_dict['avg'] = eval_dict['F1']
        return eval_dict


class LaneDistance(object):

    def __init__(self, distance_thr, inter_lane=True):
        self.distance_thr = distance_thr
        self.inter_lane = inter_lane
        self.chamfer_distance = ChamferDistance()
        self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images = 0, 0, 0, 0, 0

    @staticmethod
    def lane_interpolate(lane):
        if len(lane) == 2:
            return lane
        x = lane[0::2]
        y = lane[1::2]
        y_min, y_max = int(min(y)), int(max(y))

        y_x = list(zip(y, x))
        y_x.sort(key=lambda x: x[0])
        x, y = [], []
        for i in y_x:
            y.append(i[0])
            x.append(i[1])
        y_new = list(range(y_min, y_max + 1, 1))
        x_inter = np.interp(y_new, y, x)

        lane_new = [0 for _ in range(len(y_new) * 2)]
        lane_new[0::2] = x_inter
        lane_new[1::2] = y_new
        return lane_new

    def _single_lane_chamfer_distance(self, source_point, target_point):

        source_point_num = len(source_point) // 2
        target_point_num = len(target_point) // 2

        source_point = np.asarray(source_point)
        target_point = np.asarray(target_point)

        source_xyz = np.zeros((1, source_point_num, 3))
        target_xyz = np.zeros((1, target_point_num, 3))

        source_xyz[:, :, 0:2] = source_point.reshape(source_point_num, 2)
        target_xyz[:, :, 0:2] = target_point.reshape(target_point_num, 2)

        source_xyz = torch.FloatTensor(source_xyz).cuda()
        target_xyz = torch.FloatTensor(target_xyz).cuda()

        dist1, dist2 = self.chamfer_distance(source_xyz, target_xyz)
        mean_chamfer_distance = (torch.mean(dist1)) + (torch.mean(dist2))
        mean_chamfer_distance = mean_chamfer_distance.cpu().detach()
        return mean_chamfer_distance

    def _eval_chamfer_distance(self, pred_point, gt_point):
        hit_num = 0
        pr_num = len(pred_point)
        gt_num = len(gt_point)
        pr_list = [False for _ in range(pr_num)]
        gt_list = [False for _ in range(gt_num)]
        if len(pred_point) > 0 and len(gt_point) > 0:
            dist_mat = np.zeros((len(pred_point), len(gt_point)))
        else:
            return dict(chamfer_distance=0, gt_num=gt_num, pr_num=pr_num, hit_num=hit_num,
                        pr_list=pr_list, gt_list=gt_list)   # 当无法计算chamfer distance 的时候需要返回一个什么？

        if self.inter_lane:
            pred_point_tmp, gt_point_tmp = [], []
            for lane in pred_point:
                new_lane = self.lane_interpolate(lane)
                pred_point_tmp.append(new_lane)
            for lane in gt_point:
                new_lane = self.lane_interpolate(lane)
                gt_point_tmp.append(new_lane)
            pred_point = pred_point_tmp
            gt_point = gt_point_tmp

        for s_idx, t_idx in product(range(len(pred_point)), range(len(gt_point))):
            s = pred_point[s_idx]
            t = gt_point[t_idx]
            s_t_dist = self._single_lane_chamfer_distance(s, t)
            dist_mat[s_idx, t_idx] = s_t_dist

        cost_matrix = dist_mat
        match_index_list = linear_sum_assignment(cost_matrix)

        distance_sum = []
        for pr_index, gt_index in zip(*match_index_list):
            dis_val = dist_mat[pr_index, gt_index]
            if dis_val < self.distance_thr:
                hit_num += 1
                pr_list[pr_index] = True
                gt_list[gt_index] = True
                distance_sum.append(dis_val)
        if len(distance_sum) > 0:
            distance_sum = sum(distance_sum) / len(distance_sum)
        else:
            distance_sum = 0
        res = dict(chamfer_distance=distance_sum, gt_num=gt_num, pr_num=pr_num, hit_num=hit_num,
                   pr_list=pr_list, gt_list=gt_list)
        return res

    def _calc_single_img(self, pred_point, gt_point):
        res = self._eval_chamfer_distance(pred_point, gt_point)
        res.update(dict(pred=pred_point, gt=gt_point))
        return res

    def add_batch(self, pred_points_list, gt_points_list):
        batch_size = len(pred_points_list)
        for i in range(batch_size):
            res = self._calc_single_img(pred_points_list[i], gt_points_list[i])
            self.chamfer_dist += res['chamfer_distance']
            self.gt_num += res['gt_num']
            self.pr_num += res['pr_num']
            self.hit_num += res['hit_num']
            self.num_images += 1

    def reset(self):
        self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images = 0, 0, 0, 0, 0

    def dump_state(self, state):
        state = np.concatenate([state, np.array([self.chamfer_dist, self.gt_num, self.pr_num,
                                                 self.hit_num, self.num_images])], axis=0)
        return state

    def load_state(self, state):
        self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images = state[-5:]
        return state[:-5]

    def __call__(self):
        acc = self.hit_num / (self.pr_num + 1e-5)
        recall = self.hit_num / (self.gt_num + 1e-5)
        F1 = acc * recall * 2.0 / (acc + recall + 1e-5)
        chamfer_dist = self.chamfer_dist / self.num_images
        self.reset()
        return F1, recall, acc, chamfer_dist


if __name__ == "__main__":
    metric = LaneDetMetric(distance_thr=60)

    anno_dict = {}
    pred_dict = {}

    anno_file = './annotation.pkl'
    pred_file = './prediction.pkl'

    anno_list = ['s3://perceptor-share/data/lane_data_car2/0520_reformat_RV.json']
    pred_list = ['/data/chenziye/HDMapNet/modeling/models/detectors/cond_lstr_attr/temp_avg/lane_points.json']

    if not os.path.exists(pred_file):
        with open(pred_file, 'wb') as ff:
            for fpath in pred_list:
                with refile.smart_open(fpath, "r") as f:
                    for line in f.readlines():
                        try:
                            sample = json.loads(line)
                            nori_id = sample['image_name'].replace('.jpg', '')
                            pred_dict[nori_id] = []
                            for points_2d in sample['2d_res']:
                                pred_dict[nori_id].append(np.array(points_2d).reshape(-1))
                        except:
                            continue
            pickle.dump(pred_dict, ff)
    else:
        with open(pred_file, 'rb') as ff:
            pred_dict = pickle.load(ff)
    print('Prediction has {} samples'.format(len(pred_dict.keys())))

    if not os.path.exists(anno_file):
        with open(anno_file, 'wb') as ff:
            for fpath in anno_list:
                with refile.smart_open(fpath, "r") as f:
                    frames = json.load(f)["frame_data"]
                    for i, frame in enumerate(frames):
                        for idx in ['0_6']:  # [15]:
                            nori_id = frame['rv']['camera_{}'.format(idx)]['nori_id']
                            if nori_id in pred_dict:
                                anno_dict[nori_id] = []
                                for idx, lane in frame['rv']['camera_{}'.format(idx)]['lane'].items():
                                    points_2d = np.array(lane['points'])
                                    if len(points_2d) < 2:
                                        continue
                                    anno_dict[nori_id].append(points_2d.reshape(-1))

                # with refile.smart_open(fpath, "r") as f:
                #     reader = jsonlines.Reader(f)
                #     frames = list(reader)
                #     for i, frame in enumerate(frames):
                #         nori_id = frame["nori_id"]
                #         if nori_id in pred_dict:
                #             anno_dict[nori_id] = []
                #             for lane in frame['lane']:
                #                 points_2d = np.array(lane['lane_points'])
                #                 if len(points_2d) < 2:
                #                     continue
                #                 anno_dict[nori_id].append(points_2d.reshape(-1))

            pickle.dump(anno_dict, ff)
    else:
        with open(anno_file, 'rb') as ff:
            anno_dict = pickle.load(ff)
    print('Annotation has {} samples'.format(len(anno_dict.keys())))

    for nori_id in tqdm(pred_dict):
        pre_points_list = [pred_dict[nori_id]]
        tgt_points_list = [anno_dict[nori_id]]
        metric.add_batch(pre_points_list, tgt_points_list)

    eval_dict = metric()
    print_info = 'Result:'
    for key, value in eval_dict.items():
        print_info += ' ' + key + ': ' + '{N_acc:.3f}'.format(N_acc=value)
    print(print_info)
