"""Checks for assignment correctness, ignored labels and empty-target gradients."""
import os
os.environ['HDMAPNET_TASK_FAMILY'] = 'lane'
import unittest
import numpy as np
import torch
from experiments.ijcv_3d.loss import heat_loss, CondLSTR3DMatcher, CondLSTR3DCriterion
from experiments.ijcv_3d.geometry import bev_points_to_ego
from modeling.models.detectors.lane.cond_lstr_vote_3d.postprocess import CondLSTRVote3DPost


class MatchingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.weights = dict(obj_weight=5., cls_weight=1., msk_weight=1., reg_weight=1.)
        self.outputs = [torch.randn(1, 5, 2), torch.randn(1, 5, 4), torch.randn(1, 5, 4, 6),
                        torch.randn(1, 5, 3, 4, 6), torch.randn(1, 5, 3, 7), torch.randn(1, 5, 3, 3, 7)]
        self.target = dict(gt_lbl_obj=torch.zeros(3, dtype=torch.long), gt_lbl_cls=torch.tensor([1, 255, 3]),
                           gt_msk_bev=torch.rand(3, 4, 6), gt_reg_bev=torch.randn(3, 3, 4, 6),
                           gt_msk_img=torch.rand(3, 3, 7), gt_reg_img=torch.randn(3, 3, 3, 7))

    def test_chunking_matches_pairwise_reference_and_does_not_mutate(self):
        matcher = CondLSTR3DMatcher(self.weights, query_chunk=2)
        obj, cls, bev, br, img, ir = [x[0] for x in self.outputs]
        actual = matcher.cost_matrix(obj, cls, bev, br, img, ir, self.target)
        expected = torch.empty_like(actual)
        for q in range(5):
            for t in range(3):
                label = self.target['gt_lbl_cls'][t]
                value = -5 * obj[q].log_softmax(0)[0]
                if label != 255:
                    value -= cls[q].log_softmax(0)[label]
                for pred, reg, view, channels in [(bev, br, 'bev', 3), (img, ir, 'img', 2)]:
                    mask = self.target['gt_msk_' + view][t]
                    value += heat_loss(pred[q:q+1], mask[None])[0]
                    offset = (reg[q, :channels] - self.target['gt_reg_' + view][t, :channels]).abs().sum(0)
                    value += (offset * mask).sum() / mask.sum().clamp_min(1)
                expected[q, t] = value
        torch.testing.assert_close(actual, expected)
        self.assertEqual(self.target['gt_lbl_cls'].tolist(), [1, 255, 3])

    def test_empty_targets_have_finite_backward(self):
        target = {k: v[:0] for k, v in self.target.items()}
        outputs = [x.requires_grad_() for x in self.outputs]
        criterion = CondLSTR3DCriterion(CondLSTR3DMatcher(self.weights), **self.weights)
        losses = criterion(*outputs, [target], 1.)
        total = sum(losses)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(outputs[0].grad).all())

    def test_good_heatmap_has_lower_loss(self):
        target = torch.zeros(2, 7, 9)
        target[:, :, 4] = 1
        good = (target * 2 - 1) * 12
        self.assertTrue((heat_loss(good, target) < heat_loss(-good, target)).all())

    def test_nonempty_voting_and_metric_coordinates(self):
        post = CondLSTRVote3DPost(min_points=2, line_width=1, score_thresh=.7, mask_downscale=1, with_2d=False)
        masks = torch.full((1, 1, 8, 8), 12.)
        regs = torch.zeros(1, 1, 3, 8, 8)
        regs[:, :, 0] = 4 - torch.arange(8).float()
        regs[:, :, 2] = 40
        lanes, attrs, scores = post(torch.tensor([[[12., -12.]]]), torch.tensor([[[0., 12.]]]),
                                   masks, regs, None, None, [{}])
        self.assertEqual(len(lanes[0]), 1)
        bev = lanes[0][0][0]
        transform = np.array([[20/384, 0, 0, -10], [0, 100/768, 0, 3], [0, 0, 1/20, 0], [0, 0, 0, 1]])
        ego = bev_points_to_ego(bev, transform)
        np.testing.assert_allclose(ego[:, 0], -10 + 4 * 20/384)
        np.testing.assert_allclose(ego[:, 2], 2.)
        np.testing.assert_allclose(ego[:, 1], 3 + np.arange(8) * 100/768)
        self.assertEqual(attrs, [[1]])


if __name__ == '__main__':
    unittest.main()
