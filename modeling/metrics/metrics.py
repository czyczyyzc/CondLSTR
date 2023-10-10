import cv2
import torch
import numpy as np
# from .chamfer_dist_utils import chamfer_dist


class SegIoU(object):

    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        mIoU = np.nanmean(MIoU[1:])
        MIoU[0] = mIoU
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def dump_state(self, state):
        state = np.concatenate([state, self.confusion_matrix.reshape(-1)], axis=0)
        return state

    def load_state(self, state):
        position = self.confusion_matrix.reshape(-1).shape[0]
        self.confusion_matrix = state[-position:].reshape(self.confusion_matrix.shape)
        return state[:-position]

    def __call__(self):
        MIoU = self.Mean_Intersection_over_Union()
        self.reset()
        return MIoU


class ChamferDistance(object):

    def __init__(self, erode_ratio, meter_per_pixel, threshold):
        self.threshold = threshold  # m
        self.meter_per_pixel = meter_per_pixel  # m/pixel
        self.erode_kernel = np.ones((erode_ratio, erode_ratio), np.uint8)
        self.num_correct_g2d, self.num_points_g2d = np.array([]), np.array([])
        self.num_correct_d2g, self.num_points_d2g = np.array([]), np.array([])

    def add_batch(self, targets, outputs):

        for i in range(len(targets)):

            gt_points = [[x, y, 0] for y, x in zip(*np.where((targets[i] != 0)))]
            # outputs[i] = cv2.erode(outputs[i].astype(np.uint8), self.erode_kernel, iterations=1)
            dt_points = [[x, y, 0] for y, x in zip(*np.where((outputs[i] != 0)))]

            gt_points = torch.from_numpy(np.array(gt_points)).unsqueeze(0).cuda().float()  # (1, num_points, 3)
            dt_points = torch.from_numpy(np.array(dt_points)).unsqueeze(0).cuda().float()  # (1, num_points, 3)

            if gt_points.shape[1] > 0 and dt_points.shape[1] > 0:
                dists_g2d, dists_d2g, _, _ = chamfer_dist(gt_points, dt_points)  # ()
                # Precision
                dists_g2d = torch.sqrt(dists_g2d + 1e-8) * self.meter_per_pixel
                num_correct_g2d_batch = torch.sum((dists_g2d <= self.threshold), dim=1).cpu().numpy()
                num_points_g2d_batch = torch.sum((dists_g2d >= 0), dim=1).cpu().numpy()
                # for Recall
                dists_d2g = torch.sqrt(dists_d2g + 1e-8) * self.meter_per_pixel
                num_correct_d2g_batch = torch.sum((dists_d2g <= self.threshold), dim=1).cpu().numpy()
                num_points_d2g_batch = torch.sum((dists_d2g >= 0), dim=1).cpu().numpy()
            elif gt_points.shape[1] == 0 and dt_points.shape[1] > 0:
                num_correct_g2d_batch, num_points_g2d_batch = np.array([0]), np.array([0])
                num_correct_d2g_batch, num_points_d2g_batch = np.array([0]), np.array([dt_points.shape[1]])
            elif gt_points.shape[1] > 0 and dt_points.shape[1] == 0:
                num_correct_g2d_batch, num_points_g2d_batch = np.array([0]), np.array([gt_points.shape[1]])
                num_correct_d2g_batch, num_points_d2g_batch = np.array([0]), np.array([0])
            else:
                num_correct_g2d_batch, num_points_g2d_batch = np.array([0]), np.array([0])
                num_correct_d2g_batch, num_points_d2g_batch = np.array([0]), np.array([0])

            self.num_correct_g2d = np.concatenate([self.num_correct_g2d, num_correct_g2d_batch])
            self.num_points_g2d = np.concatenate([self.num_points_g2d, num_points_g2d_batch])
            self.num_correct_d2g = np.concatenate([self.num_correct_d2g, num_correct_d2g_batch])
            self.num_points_d2g = np.concatenate([self.num_points_d2g, num_points_d2g_batch])

    def reset(self):
        self.num_correct_g2d, self.num_points_g2d = np.array([]), np.array([])
        self.num_correct_d2g, self.num_points_d2g = np.array([]), np.array([])

    def dump_state(self, state):
        state = np.concatenate([state,
                                self.num_correct_g2d, self.num_points_g2d,
                                self.num_correct_d2g, self.num_points_d2g], axis=0)
        return state

    def load_state(self, state):
        offset = self.num_points_d2g.shape[0]
        self.num_points_d2g = state[-offset * 1:]
        self.num_correct_d2g = state[-offset * 2: -offset * 1]
        self.num_points_g2d = state[-offset * 3: -offset * 2]
        self.num_correct_g2d = state[-offset * 4: -offset * 3]
        return state[:-4 * offset]

    def __call__(self):
        p = np.sum(self.num_correct_g2d) / np.sum(self.num_points_g2d)
        r = np.sum(self.num_correct_d2g) / np.sum(self.num_points_d2g)
        score = 2 * p * r / (p + r)
        self.reset()
        return [score]


class SegCDs(object):

    def __init__(self, pixels_per_meter=50, max_points=30):
        self.pixels_per_meter = pixels_per_meter
        self.max_points = max_points
        self.num_batches = 0
        self.cds_total = np.zeros((2,))

    def _chanfer_dist(self, image1, image2):
        assert image1.shape[0] == image2.shape[0], \
            "The number of classes for image1 and image2 are different!"
        line_dist = LineDist(image1, image2, pixels_per_meter=self.pixels_per_meter,
                             max_points=self.max_points)

        dist3 = []
        for class_index, (img1, img2) in enumerate(zip(image1, image2)):
            lanes1 = set(np.unique(img1)) - {0}
            lanes2 = set(np.unique(img2)) - {0}
            lane_num1 = len(lanes1)
            lane_num2 = len(lanes2)
            if lane_num1 == 0 or lane_num2 == 0:
                continue
            dist1 = []
            for lane1 in lanes1:
                dist2 = []
                for lane2 in lanes2:
                    dist = line_dist(lane1, lane2)
                    dist2.append(dist)
                dist1.append(dist2)
            dist = np.array(dist1)
            dist = np.mean(np.min(dist, axis=1))
            dist3.append(dist)
        if len(dist3) > 0:
            dist = sum(dist3) / len(dist3)
        else:
            dist = 0
        return dist

    def add_batch(self, gt_image, pre_image):
        cd_pl = 0
        cd_lp = 0
        for i in range(len(gt_image)):
            cd_pl += self._chanfer_dist(pre_image[i], gt_image[i])
            cd_lp += self._chanfer_dist(gt_image[i], pre_image[i])
        cd_pl /= len(gt_image)
        cd_lp /= len(gt_image)
        self.cds_total[0] += cd_pl
        self.cds_total[1] += cd_lp
        self.num_batches += 1

    def reset(self):
        self.num_batches = 0
        self.cds_total = np.zeros((2,))

    def dump_state(self, state):
        state = np.concatenate([state, self.cds_total, np.array([self.num_batches])], axis=0)
        return state

    def load_state(self, state):
        self.cds_total = state[-3:-1]
        self.num_batches = state[-1]
        return state[:-3]

    def __call__(self):
        cds = self.cds_total / self.num_batches
        self.reset()
        return cds


class SegmAP(object):

    def __init__(self, cd_thresholds=[0.2, 0.5, 1.0], pixels_per_meter=50, max_points=30):
        self.cd_thresholds = cd_thresholds
        self.pixels_per_meter = pixels_per_meter
        self.max_points = max_points
        self.num_batches = 0
        self.mAP_total = np.zeros((len(self.cd_thresholds,)))

    def _mAP(self, gt_image, pre_image, cd_threshold=0.2):
        """
         Calculate the AP for each class
        """
        assert gt_image.shape[0] == pre_image.shape[0], \
            "The number of classes for ground truth and prediction are different!"
        sum_AP = 0.0
        n_classes = 0
        for class_index, (dr_data, ground_truth_data) in enumerate(zip(pre_image, gt_image)):
            """
             Assign detection-results to ground-truth objects
            """
            line_dist = LineDist(dr_data, ground_truth_data, pixels_per_meter=self.pixels_per_meter,
                                 max_points=self.max_points)

            ds = set(np.unique(dr_data)) - {0}
            nd = len(ds)
            
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd

            gs = set(np.unique(ground_truth_data)) - {0}
            ng = len(gs)

            gt_used = [False] * ng
            
            if ng == 0:
                continue
            n_classes += 1
            
            if nd == 0:
                continue

            for d_i, d_lane in enumerate(ds):
                # assign detection-results to ground truth object if any
                dist_min = np.inf
                gt_match = -1
                # load detected object bounding-box
                for g_j, g_lane in enumerate(gs):
                    dist = line_dist(d_lane, g_lane)
                    if dist < dist_min:
                        dist_min = dist
                        gt_match = g_j
                if dist_min <= cd_threshold:
                    if not gt_used[gt_match]:
                        # true positive
                        tp[d_i] = 1
                        gt_used[gt_match] = True
                    else:
                        # false positive (multiple detection)
                        fp[d_i] = 1
                else:
                    # false positive
                    fp[d_i] = 1
            # print(tp)
            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            # print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / ng  # gt_counter_per_class[class_name]
            # print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            # print(prec)
            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
        if n_classes > 0:
            mAP = sum_AP / n_classes
        else:
            mAP = 0.0
        return mAP

    def add_batch(self, gt_image, pre_image):
        for i, cd_threshold in enumerate(self.cd_thresholds):
            mAP = 0.0
            for j in range(len(gt_image)):
                mAP += self._mAP(gt_image[j], pre_image[j], cd_threshold)
            mAP /= len(gt_image)
            self.mAP_total[i] += mAP
        self.num_batches += 1

    def reset(self):
        self.num_batches = 0
        self.mAP_total = np.zeros((len(self.cd_thresholds,)))

    def dump_state(self, state):
        state = np.concatenate([state, self.mAP_total, np.array([self.num_batches])], axis=0)
        return state

    def load_state(self, state):
        self.mAP_total = state[-4:-1]
        self.num_batches = state[-1]
        return state[:-4]

    def __call__(self):
        mAP = self.mAP_total / self.num_batches
        self.reset()
        return mAP


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


class LineDist(object):

    def __init__(self, image1, image2, pixels_per_meter=100, max_points=-1):
        self.image1 = torch.as_tensor(image1, device=torch.device('cuda'))
        self.image2 = torch.as_tensor(image2, device=torch.device('cuda'))
        self.pixels_per_meter = pixels_per_meter
        self.max_points = max_points

    def __call__(self, line1, line2):
        pts1 = torch.stack(torch.where(self.image1 == line1), dim=1).float()   # (M, 2)
        pts2 = torch.stack(torch.where(self.image2 == line2), dim=1).float()   # (N, 2)
        if self.max_points > 0:
            npt1 = min(len(pts1), self.max_points)
            npt2 = min(len(pts2), self.max_points)
            pts1 = pts1[np.random.choice(len(pts1), npt1, replace=False)]
            pts2 = pts2[np.random.choice(len(pts2), npt2, replace=False)]
        dist = torch.linalg.norm(pts1[:, None] - pts2[None, :], ord=2, dim=2)  # (M, N)
        dst1 = torch.mean(torch.min(dist, dim=1)[0])
        dst2 = torch.mean(torch.min(dist, dim=0)[0])
        dist = (dst1 + dst2) / self.pixels_per_meter
        dist = dist.cpu().numpy()
        return dist


if __name__ == "__main__":
    map_masks = np.load('/data/sets/nuscenes/nuscenes_maps/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489296547795.pcd.npz')['arr_0']
    seg_iou = SegIoU()
    seg_cds = SegCDs()
    seg_map = SegmAP()
    seg_iou.add_batch(map_masks[None, 0], map_masks[None, 0])
    iou = seg_iou()
    print(iou)

    seg_cds.add_batch(map_masks[None, 4:], map_masks[None, 4:])
    cds = seg_cds()
    print(cds)

    seg_map.add_batch(map_masks[None, 4:], map_masks[None, 4:])
    mAP = seg_map()
    print(mAP)

