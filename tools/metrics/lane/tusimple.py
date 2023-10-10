import os
import sys
import json
import refile
import pickle
import argparse
import imagesize
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1.
        angles = [
            LaneEval.get_angle(np.array(x_gts), np.array(y_samples))
            for x_gts in gt
        ]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [
                LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts),
                                       thresh) for x_preds in pred
            ]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)),
                       1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(
                           min(len(gt), 4.), 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [
                json.loads(line) for line in open(pred_file).readlines()
            ]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception(
                'We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred:  # or 'run_time' not in pred:
                raise Exception(
                    'raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time'] if 'run_time' in pred else 0
            if raw_file not in gts:
                raise Exception(
                    'Some raw_file from your predictions do not exist in the test tasks.'
                )
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples,
                                         run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter

        fp = fp / num
        fn = fn / num
        tp = 1 - fp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        return json.dumps([{
            'name': 'Accuracy',
            'value': accuracy / num,
            'order': 'desc'
        }, {
            'name': 'F1_score',
            'value': f1,
            'order': 'desc'
        }, {
            'name': 'FP',
            'value': fp,
            'order': 'asc'
        }, {
            'name': 'FN',
            'value': fn,
            'order': 'asc'
        }]), accuracy / num


def main():
    args = parse_args()

    if args.translate:
        with refile.smart_open(os.path.join(args.pred_dir, 'results.pkl'), 'rb') as f:
            results_list = pickle.load(f)

        h_samples_dict = {}
        with refile.smart_open(os.path.join(args.anno_dir, 'test_label.json'), 'rb') as f:
            for line in f.readlines():
                ann_dict = json.loads(line)
                raw_file = ann_dict['raw_file']
                y_sample = ann_dict['h_samples']
                h_samples_dict[raw_file] = y_sample

        predictions = []
        for result in tqdm(results_list):
            res_list = result['2d_res']
            img_name = result['image_name'].replace(',', '/')
            assert img_name in h_samples_dict, "Wrong image name {} !".format(img_name)

            y_new = h_samples_dict[img_name]
            res_dict = {'run_time': 0, 'lanes': [], 'raw_file': None, 'h_samples': None}
            for lane in res_list:
                x = lane[0::2]
                y = lane[1::2]
                y_x = list(zip(y, x))
                y_x.sort(key=lambda x: x[0])
                x, y = [], []
                for i in y_x:
                    y.append(i[0])
                    x.append(i[1])

                y_min = min(y)
                y_max = max(y)
                x_new = []
                for y_ in y_new:
                    if y_ < y_min or y_ > y_max:
                        x_new.append(-2)
                    else:
                        x_new.append(np.interp([y_], y, x)[0])
                res_dict['lanes'].append(x_new)

            res_dict['raw_file'] = img_name
            res_dict['h_samples'] = y_new
            res_line = json.dumps(res_dict)
            predictions.append(res_line + '\n')

        with open(os.path.join(args.pred_dir, 'results.json'), 'w') as f:
            f.writelines(predictions)

    print('#########################################')
    print(args.pred_dir)
    print(args.anno_dir)

    for fpath in args.list:
        res_path = os.path.join(args.pred_dir, 'results.json')
        ann_path = os.path.join(args.anno_dir, fpath)
        print(LaneEval.bench_one_submit(res_path, ann_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Tusimple's metric")
    parser.add_argument(
        "--pred_dir",
        default='/data/chenziye/HDMapNet/output/test_cond_lstr_80lane_250epoch_old_great_res18_tusimple_s0.85',
        help="Path to directory containing the predicted lanes"
    )
    parser.add_argument(
        "--anno_dir",
        default='/data/sets/tusimple',
        help="Path to directory containing the annotated lanes"
    )
    parser.add_argument("--list",
                        nargs='+',
                        default=['test_label.json'],
                        help="Path to txt file containing the list of files"
    )
    parser.add_argument('--translate', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    main()
