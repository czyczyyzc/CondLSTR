import torch
import torch.nn.functional as F
import numpy as np
from itertools import product
from scipy.optimize import linear_sum_assignment
from .chamfer_distance import ChamferDistance


class LaneDet3DMetric(object):

    def __init__(self, distance_thr=1, num_classes=1, **kwargs):
        self.num_classes = num_classes
        self.lane_distance = LaneDistance(distance_thr, num_classes=num_classes)

    def add_batch(self, pred_dict, data_dict):
        pre_points_batch = pred_dict['lane_points']
        pre_attris_batch = pred_dict['lane_attris']
        tgt_points_batch = data_dict['lane_points']
        tgt_attris_batch = data_dict['lane_attris']
        img_metas_batch = data_dict['img_metas']
        
        pre_points_list = []
        tgt_points_list = []
        for pre_points, tgt_points, pre_attris, tgt_attris, img_meta in \
            zip(pre_points_batch, tgt_points_batch, pre_attris_batch, tgt_attris_batch, img_metas_batch):
            P_bev2ego = img_meta['calibration']['trans_bev2ego']

            pre_points_new = []
            for points_3d, attr in zip(pre_points, pre_attris):
                attr = int(attr)
                points_3d = points_3d[0] if isinstance(points_3d, tuple) else points_3d
                points_3d = np.asarray(points_3d)
                points_3d = np.concatenate([points_3d, np.ones_like(points_3d[:, 0:1])], axis=1).T
                points_3d = np.matmul(P_bev2ego, points_3d)
                points_3d = points_3d.T
                points_3d = points_3d[:, :3]
                points_3d = points_3d.reshape(-1)
                pre_points_new.append((points_3d, attr))

            tgt_points_new = []
            for points_3d, attr in zip(tgt_points, tgt_attris):
                attr = int(attr.cpu().numpy())
                points_3d = points_3d.cpu().numpy()
                points_3d = np.concatenate([points_3d, np.ones_like(points_3d[:, 0:1])], axis=1).T
                points_3d = np.matmul(P_bev2ego, points_3d)
                points_3d = points_3d.T
                points_3d = points_3d[:, :3]
                points_3d = points_3d.reshape(-1)
                tgt_points_new.append((points_3d, attr))

            pre_points_list.append(pre_points_new)
            tgt_points_list.append(tgt_points_new)

        self.lane_distance.add_batch(pre_points_list, tgt_points_list)

    def dump_state(self):
        state = np.array([])
        state = self.lane_distance.dump_state(state)
        return state

    def load_state(self, state):
        state = self.lane_distance.load_state(state)
        assert len(state) == 0

    def __call__(self):
        eval_dict = {}
        F1, recall, acc, chamfer_dist, F1_attr, recall_attr, acc_attr, recall_attr_all = self.lane_distance()
        eval_dict.update({'F1': F1, 'recall': recall, 'acc': acc, 'chamfer_dist': chamfer_dist})

        for i in range(self.num_classes):
            eval_dict.update({'F1_'+str(i): F1_attr[i], 'recall_'+str(i): recall_attr[i], 'acc_'+str(i): acc_attr[i],
                             'recall_all_'+str(i): recall_attr_all[i]})

        eval_dict['avg'] = eval_dict['F1']
        return eval_dict


class LaneDistance(object):

    def __init__(self, distance_thr, inter_lane=True, num_classes=4):
        self.distance_thr = distance_thr
        self.inter_lane = inter_lane
        self.num_classes = num_classes
        self.chamfer_distance = ChamferDistance()
        self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images = 0, 0, 0, 0, 0
        self.gt_num_attr  = [0 for _ in range(self.num_classes)]
        self.pr_num_attr  = [0 for _ in range(self.num_classes)]
        self.hit_num_attr = [0 for _ in range(self.num_classes)]
        self.tgt_num_attr = [0 for _ in range(self.num_classes)]

    @staticmethod
    def lane_interpolate(lane):
        if len(lane) == 3:
            return lane
        x = lane[0::3]
        y = lane[1::3]
        z = lane[2::3]
        y_min, y_max = int(min(y)), int(max(y))

        y_x_z = list(zip(y, x, z))
        y_x_z.sort(key=lambda x: x[0])
        x, y, z = [], [], []
        for i in y_x_z:
            y.append(i[0])
            x.append(i[1])
            z.append(i[2])
        
        y_new = list(range(y_min, y_max + 1, 1))
        x_inter = np.interp(y_new, y, x)
        z_inter = np.interp(y_new, y, z)
        
        lane_new = [0 for _ in range(len(y_new) * 3)]
        lane_new[0::3] = x_inter
        lane_new[1::3] = y_new
        lane_new[2::3] = z_inter
        return lane_new

    def _single_lane_chamfer_distance(self, source_point, target_point):

        source_point_num = len(source_point) // 3
        target_point_num = len(target_point) // 3

        source_point = np.asarray(source_point)
        target_point = np.asarray(target_point)

        source_xyz = np.zeros((1, source_point_num, 3))
        target_xyz = np.zeros((1, target_point_num, 3))

        source_xyz[:, :, 0:3] = source_point.reshape(source_point_num, 3)
        target_xyz[:, :, 0:3] = target_point.reshape(target_point_num, 3)

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

        distance_sum = []
        pr_list = [False for _ in range(pr_num)]
        gt_list = [False for _ in range(gt_num)]

        gt_num_attr  = [0 for _ in range(self.num_classes)]
        pr_num_attr  = [0 for _ in range(self.num_classes)]
        hit_num_attr = [0 for _ in range(self.num_classes)]
        tgt_num_attr = [0 for _ in range(self.num_classes)]

        if len(pred_point) > 0 and len(gt_point) > 0:
            dist_mat = np.zeros((len(pred_point), len(gt_point)))
        else:
            # 当无法计算chamfer distance 的时候需要返回一个什么？
            return dict(chamfer_distance=0, gt_num=gt_num, pr_num=pr_num, hit_num=hit_num,
                        pr_list=pr_list, gt_list=gt_list, gt_num_attr=gt_num_attr, pr_num_attr=pr_num_attr,
                        hit_num_attr=hit_num_attr, tgt_num_attr=tgt_num_attr)

        pred_point, pred_attrs = list(zip(*pred_point))
        gt_point, gt_attrs = list(zip(*gt_point))

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

        for gt_attr in gt_attrs:
            if gt_attr == 255:
                continue
            tgt_num_attr[int(gt_attr)] += 1

        for s_idx, t_idx in product(range(len(pred_point)), range(len(gt_point))):
            s = pred_point[s_idx]
            t = gt_point[t_idx]
            s_t_dist = self._single_lane_chamfer_distance(s, t)
            dist_mat[s_idx, t_idx] = s_t_dist

        cost_matrix = dist_mat
        match_index_list = linear_sum_assignment(cost_matrix)

        for pr_index, gt_index in zip(*match_index_list):
            dis_val = dist_mat[pr_index, gt_index]

            if dis_val < self.distance_thr:
                hit_num += 1
                pr_list[pr_index] = True
                gt_list[gt_index] = True
                distance_sum.append(dis_val)

            pr_attr = pred_attrs[pr_index]
            gt_attr = gt_attrs[gt_index]

            if gt_attr == 255:
                continue
            gt_num_attr[int(gt_attr)] += 1

            if pr_attr == 255:
                continue
            pr_num_attr[int(pr_attr)] += 1

            if gt_attr == pr_attr:
                hit_num_attr[int(gt_attr)] += 1

        if len(distance_sum) > 0:
            distance_sum = sum(distance_sum) / len(distance_sum)
        else:
            distance_sum = 0
        
        res = dict(chamfer_distance=distance_sum, gt_num=gt_num, pr_num=pr_num, hit_num=hit_num,
                   pr_list=pr_list, gt_list=gt_list, gt_num_attr=gt_num_attr, pr_num_attr=pr_num_attr,
                   hit_num_attr=hit_num_attr, tgt_num_attr=tgt_num_attr)
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

            for j in range(self.num_classes):
                self.gt_num_attr[j] += res['gt_num_attr'][j]
                self.pr_num_attr[j] += res['pr_num_attr'][j]
                self.hit_num_attr[j] += res['hit_num_attr'][j]
                self.tgt_num_attr[j] += res['tgt_num_attr'][j]

            self.num_images += 1

    def reset(self):
        self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images = 0, 0, 0, 0, 0
        self.gt_num_attr  = [0 for _ in range(self.num_classes)]
        self.pr_num_attr  = [0 for _ in range(self.num_classes)]
        self.hit_num_attr = [0 for _ in range(self.num_classes)]
        self.tgt_num_attr = [0 for _ in range(self.num_classes)]

    def dump_state(self, state):
        state = np.concatenate([state, np.array(
            [self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images] +
            self.gt_num_attr + self.pr_num_attr + self.hit_num_attr + self.tgt_num_attr)], axis=0)
        return state

    def load_state(self, state):
        self.tgt_num_attr = list(state[-self.num_classes:])
        state = state[:-self.num_classes]

        self.hit_num_attr = list(state[-self.num_classes:])
        state = state[:-self.num_classes]

        self.pr_num_attr = list(state[-self.num_classes:])
        state = state[:-self.num_classes]

        self.gt_num_attr = list(state[-self.num_classes:])
        state = state[:-self.num_classes]

        self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images = state[-5:]
        return state[:-5]

    def __call__(self):
        acc = self.hit_num / (self.pr_num + 1e-5)
        recall = self.hit_num / (self.gt_num + 1e-5)
        F1 = acc * recall * 2.0 / (acc + recall + 1e-5)
        chamfer_dist = self.chamfer_dist / self.num_images

        F1_attr = [0. for _ in range(self.num_classes)]
        acc_attr = [0. for _ in range(self.num_classes)]
        recall_attr = [0. for _ in range(self.num_classes)]
        recall_attr_all = [0. for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            acc_attr[i] = self.hit_num_attr[i] / (self.pr_num_attr[i] + 1e-5)
            recall_attr[i] = self.hit_num_attr[i] / (self.gt_num_attr[i] + 1e-5)
            F1_attr[i] = acc_attr[i] * recall_attr[i] * 2.0 / (acc_attr[i] + recall_attr[i] + 1e-5)
            recall_attr_all[i] = self.hit_num_attr[i] / (self.tgt_num_attr[i] + 1e-5)

        self.reset()
        return F1, recall, acc, chamfer_dist, F1_attr, recall_attr, acc_attr, recall_attr_all



# class LaneDet3DMetric(object):

#     def __init__(self, distance_thr=1, num_classes=1):
#         self.num_classes = num_classes
#         self.lane_distance = LaneDistance(distance_thr, num_classes=num_classes)

#     def add_batch(self, pred_dict, data_dict):
#         if '3d' in pred_dict:
#             seg_preds = pred_dict['3d']['seg_pred'].float()                                       # (N, 5, H, W)
#             img_masks = data_dict['map_mask'].float()                                             # (N, 5, H, W)
#         elif '2d' in pred_dict:
#             seg_preds = pred_dict['2d']['seg_pred'].float()                                       # (N, 5, H, W)
#             img_masks = data_dict['img_mask'].float()                                             # (N, 5, H, W)
#         else:
#             raise NotImplementedError

#         pre_points_list = []
#         tgt_points_list = []

#         for seg_pred, img_mask in zip(seg_preds, img_masks):                                       # (5, H, W)  (5, H, W)
#             pre_mask = F.one_hot(seg_pred[0].long(), num_classes=-1).permute(2, 0, 1)[1:].float()  # (M, H, W)
#             tgt_mask = F.one_hot(img_mask[0].long(), num_classes=-1).permute(2, 0, 1)[1:].float()  # (M, H, W)
#             pre_mask = pre_mask[pre_mask.sum(dim=(1, 2)) > 0]                                      # (M, H, W)
#             tgt_mask = tgt_mask[tgt_mask.sum(dim=(1, 2)) > 0]                                      # (M, H, W)

#             pre_attr = pre_mask.flatten(1, 2).mm(seg_pred[2:].flatten(1, 2).t())                   # (M, D)
#             tgt_attr = tgt_mask.flatten(1, 2).mm(img_mask[2:].flatten(1, 2).t())                   # (M, D)
#             pre_attr = pre_attr / pre_mask.sum(dim=(1, 2))[:, None]                                # (M, D)
#             tgt_attr = tgt_attr / tgt_mask.sum(dim=(1, 2))[:, None]                                # (M, D)
#             pre_attr = pre_attr.long().cpu().numpy()                                               # (M, D)
#             tgt_attr = tgt_attr.long().cpu().numpy()                                               # (M, D)

#             pre_points = []
#             for mask, attr in zip(pre_mask, pre_attr):
#                 points = mask.nonzero(as_tuple=False).flip(1).reshape(-1)
#                 points = points.cpu().numpy()
#                 pre_points.append((points, attr))

#             tgt_points = []
#             for mask, attr in zip(tgt_mask, tgt_attr):
#                 points = mask.nonzero(as_tuple=False).flip(1).reshape(-1)
#                 points = points.cpu().numpy()
#                 tgt_points.append((points, attr))

#             pre_points_list.append(pre_points)
#             tgt_points_list.append(tgt_points)

#         self.lane_distance.add_batch(pre_points_list, tgt_points_list)

#     def dump_state(self):
#         state = np.array([])
#         state = self.lane_distance.dump_state(state)
#         return state

#     def load_state(self, state):
#         state = self.lane_distance.load_state(state)
#         assert len(state) == 0

#     def __call__(self):
#         eval_dict = {}
#         F1, recall, acc, chamfer_dist, F1_attr, recall_attr, acc_attr, recall_attr_all = self.lane_distance()
#         eval_dict.update({'F1': F1, 'recall': recall, 'acc': acc, 'chamfer_dist': chamfer_dist})

#         for i in range(self.num_classes * 2):
#             eval_dict.update({'F1_'+str(i): F1_attr[i], 'recall_'+str(i): recall_attr[i], 'acc_'+str(i): acc_attr[i],
#                              'recall_all_'+str(i): recall_attr_all[i]})

#         eval_dict['avg'] = eval_dict['F1']
#         return eval_dict


# class LaneDistance(object):

#     def __init__(self, distance_thr, inter_lane=True, num_classes=4):
#         self.distance_thr = distance_thr
#         self.inter_lane = inter_lane
#         self.num_classes = num_classes
#         self.chamfer_distance = ChamferDistance()
#         self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images = 0, 0, 0, 0, 0
#         self.gt_num_attr  = [0 for _ in range(self.num_classes * 2)]
#         self.pr_num_attr  = [0 for _ in range(self.num_classes * 2)]
#         self.hit_num_attr = [0 for _ in range(self.num_classes * 2)]
#         self.tgt_num_attr = [0 for _ in range(self.num_classes * 2)]

#     @staticmethod
#     def lane_interpolate(lane):
#         if len(lane) == 2:
#             return lane
#         x = lane[0::2]
#         y = lane[1::2]
#         y_min, y_max = int(min(y)), int(max(y))

#         y_x = list(zip(y, x))
#         y_x.sort(key=lambda x: x[0])
#         x, y = [], []
#         for i in y_x:
#             y.append(i[0])
#             x.append(i[1])
#         y_new = list(range(y_min, y_max + 1, 1))
#         x_inter = np.interp(y_new, y, x)

#         lane_new = [0 for _ in range(len(y_new) * 2)]
#         lane_new[0::2] = x_inter
#         lane_new[1::2] = y_new
#         return lane_new

#     def _single_lane_chamfer_distance(self, source_point, target_point):

#         source_point_num = len(source_point) // 2
#         target_point_num = len(target_point) // 2

#         source_point = np.asarray(source_point)
#         target_point = np.asarray(target_point)

#         source_xyz = np.zeros((1, source_point_num, 3))
#         target_xyz = np.zeros((1, target_point_num, 3))

#         source_xyz[:, :, 0:2] = source_point.reshape(source_point_num, 2)
#         target_xyz[:, :, 0:2] = target_point.reshape(target_point_num, 2)

#         source_xyz = torch.FloatTensor(source_xyz).cuda()
#         target_xyz = torch.FloatTensor(target_xyz).cuda()

#         dist1, dist2 = self.chamfer_distance(source_xyz, target_xyz)
#         mean_chamfer_distance = (torch.mean(dist1)) + (torch.mean(dist2))
#         mean_chamfer_distance = mean_chamfer_distance.cpu().detach()
#         return mean_chamfer_distance

#     def _eval_chamfer_distance(self, pred_point, gt_point):
#         hit_num = 0
#         pr_num = len(pred_point)
#         gt_num = len(gt_point)

#         distance_sum = []
#         pr_list = [False for _ in range(pr_num)]
#         gt_list = [False for _ in range(gt_num)]

#         gt_num_attr  = [0 for _ in range(self.num_classes * 2)]
#         pr_num_attr  = [0 for _ in range(self.num_classes * 2)]
#         hit_num_attr = [0 for _ in range(self.num_classes * 2)]
#         tgt_num_attr = [0 for _ in range(self.num_classes * 2)]

#         if len(pred_point) > 0 and len(gt_point) > 0:
#             dist_mat = np.zeros((len(pred_point), len(gt_point)))
#         else:
#             # 当无法计算chamfer distance 的时候需要返回一个什么？
#             return dict(chamfer_distance=0, gt_num=gt_num, pr_num=pr_num, hit_num=hit_num,
#                         pr_list=pr_list, gt_list=gt_list, gt_num_attr=gt_num_attr, pr_num_attr=pr_num_attr,
#                         hit_num_attr=hit_num_attr, tgt_num_attr=tgt_num_attr)

#         pred_point, pred_attrs = list(zip(*pred_point))
#         gt_point, gt_attrs = list(zip(*gt_point))

#         if self.inter_lane:
#             pred_point_tmp, gt_point_tmp = [], []
#             for lane in pred_point:
#                 new_lane = self.lane_interpolate(lane)
#                 pred_point_tmp.append(new_lane)
#             for lane in gt_point:
#                 new_lane = self.lane_interpolate(lane)
#                 gt_point_tmp.append(new_lane)
#             pred_point = pred_point_tmp
#             gt_point = gt_point_tmp

#         for gt_attrs_ in gt_attrs:
#             for i, gt_attr in enumerate(gt_attrs_):
#                 if gt_attr == 255:
#                     continue
#                 tgt_num_attr[2 * i + int(gt_attr)] += 1

#         for s_idx, t_idx in product(range(len(pred_point)), range(len(gt_point))):
#             s = pred_point[s_idx]
#             t = gt_point[t_idx]
#             s_t_dist = self._single_lane_chamfer_distance(s, t)
#             dist_mat[s_idx, t_idx] = s_t_dist

#         cost_matrix = dist_mat
#         match_index_list = linear_sum_assignment(cost_matrix)

#         for pr_index, gt_index in zip(*match_index_list):
#             dis_val = dist_mat[pr_index, gt_index]

#             if dis_val < self.distance_thr:
#                 hit_num += 1
#                 pr_list[pr_index] = True
#                 gt_list[gt_index] = True
#                 distance_sum.append(dis_val)

#             for i, (pr_attr, gt_attr) in enumerate(zip(pred_attrs[pr_index], gt_attrs[gt_index])):
#                 if gt_attr == 255:
#                     continue
#                 gt_num_attr[2 * i + int(gt_attr)] += 1

#                 if pr_attr == 255:
#                     continue
#                 pr_num_attr[2 * i + int(pr_attr)] += 1

#                 if gt_attr == pr_attr:
#                     hit_num_attr[2 * i + int(gt_attr)] += 1

#         if len(distance_sum) > 0:
#             distance_sum = sum(distance_sum) / len(distance_sum)
#         else:
#             distance_sum = 0
#         res = dict(chamfer_distance=distance_sum, gt_num=gt_num, pr_num=pr_num, hit_num=hit_num,
#                    pr_list=pr_list, gt_list=gt_list, gt_num_attr=gt_num_attr, pr_num_attr=pr_num_attr,
#                    hit_num_attr=hit_num_attr, tgt_num_attr=tgt_num_attr)
#         return res

#     def _calc_single_img(self, pred_point, gt_point):
#         res = self._eval_chamfer_distance(pred_point, gt_point)
#         res.update(dict(pred=pred_point, gt=gt_point))
#         return res

#     def add_batch(self, pred_points_list, gt_points_list):
#         batch_size = len(pred_points_list)

#         for i in range(batch_size):
#             res = self._calc_single_img(pred_points_list[i], gt_points_list[i])

#             self.chamfer_dist += res['chamfer_distance']
#             self.gt_num += res['gt_num']
#             self.pr_num += res['pr_num']
#             self.hit_num += res['hit_num']

#             for j in range(self.num_classes * 2):
#                 self.gt_num_attr[j] += res['gt_num_attr'][j]
#                 self.pr_num_attr[j] += res['pr_num_attr'][j]
#                 self.hit_num_attr[j] += res['hit_num_attr'][j]
#                 self.tgt_num_attr[j] += res['tgt_num_attr'][j]

#             self.num_images += 1

#     def reset(self):
#         self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images = 0, 0, 0, 0, 0
#         self.gt_num_attr  = [0 for _ in range(self.num_classes * 2)]
#         self.pr_num_attr  = [0 for _ in range(self.num_classes * 2)]
#         self.hit_num_attr = [0 for _ in range(self.num_classes * 2)]
#         self.tgt_num_attr = [0 for _ in range(self.num_classes * 2)]

#     def dump_state(self, state):
#         state = np.concatenate([state, np.array(
#             [self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images] +
#             self.gt_num_attr + self.pr_num_attr + self.hit_num_attr + self.tgt_num_attr)], axis=0)
#         return state

#     def load_state(self, state):
#         self.tgt_num_attr = list(state[-self.num_classes * 2:])
#         state = state[:-self.num_classes * 2]

#         self.hit_num_attr = list(state[-self.num_classes * 2:])
#         state = state[:-self.num_classes * 2]

#         self.pr_num_attr = list(state[-self.num_classes * 2:])
#         state = state[:-self.num_classes * 2]

#         self.gt_num_attr = list(state[-self.num_classes * 2:])
#         state = state[:-self.num_classes * 2]

#         self.chamfer_dist, self.gt_num, self.pr_num, self.hit_num, self.num_images = state[-5:]
#         return state[:-5]

#     def __call__(self):
#         acc = self.hit_num / (self.pr_num + 1e-5)
#         recall = self.hit_num / (self.gt_num + 1e-5)
#         F1 = acc * recall * 2.0 / (acc + recall + 1e-5)
#         chamfer_dist = self.chamfer_dist / self.num_images

#         F1_attr = [0. for _ in range(self.num_classes * 2)]
#         acc_attr = [0. for _ in range(self.num_classes * 2)]
#         recall_attr = [0. for _ in range(self.num_classes * 2)]
#         recall_attr_all = [0. for _ in range(self.num_classes * 2)]
#         for i in range(self.num_classes * 2):
#             acc_attr[i] = self.hit_num_attr[i] / (self.pr_num_attr[i] + 1e-5)
#             recall_attr[i] = self.hit_num_attr[i] / (self.gt_num_attr[i] + 1e-5)
#             F1_attr[i] = acc_attr[i] * recall_attr[i] * 2.0 / (acc_attr[i] + recall_attr[i] + 1e-5)
#             recall_attr_all[i] = self.hit_num_attr[i] / (self.tgt_num_attr[i] + 1e-5)

#         self.reset()
#         return F1, recall, acc, chamfer_dist, F1_attr, recall_attr, acc_attr, recall_attr_all
