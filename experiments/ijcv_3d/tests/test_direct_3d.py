"""Tests the direct-coordinate control without requiring a GPU or dataset."""
import os
os.environ['HDMAPNET_TASK_FAMILY'] = 'lane'
import unittest

import numpy as np
import torch

from experiments.ijcv_3d.direct_3d import (
    Direct3DHead, Direct3DLoss, Direct3DMatcher, LaneDirect3D, interpolate_lane,
)


class Direct3DTests(unittest.TestCase):
    def test_interpolation_sorts_and_averages_duplicate_y(self):
        lane = torch.tensor([[4., 5., 2.], [0., 3., 0.], [2., 5., 0.], [6., 7., 4.]])
        anchors = torch.tensor([3., 4., 5., 6., 7., 8.])
        coords, visible = interpolate_lane(lane, anchors)
        self.assertEqual(visible.tolist(), [True, True, True, True, True, False])
        # The two samples at y=5 average to x/z=(3,1).
        torch.testing.assert_close(coords[:5], torch.tensor([
            [0., 0.], [1.5, .5], [3., 1.], [4.5, 2.5], [6., 4.]]))
        torch.testing.assert_close(coords[-1], torch.zeros(2))

    def test_matcher_prefers_correct_geometry(self):
        matcher = Direct3DMatcher(dict(obj=5., cls=1., visibility=1., coordinates=1.))
        target = dict(labels=torch.tensor([2]), visible=torch.tensor([[True, True, True]]),
                      coords=torch.tensor([[[1., 0.], [2., .2], [3., .4]]]))
        obj = torch.tensor([[8., -8.], [8., -8.]])
        cls = torch.full((2, 21), -8.)
        cls[:, 2] = 8.
        coords = torch.stack((target['coords'][0], target['coords'][0] + 4))
        vis = torch.full((2, 3), 8.)
        cost = matcher.cost_matrix(obj, cls, coords, vis, target)
        self.assertLess(cost[0, 0], cost[1, 0])

    def test_empty_target_loss_is_finite_and_differentiable(self):
        loss = Direct3DLoss(points=5, y_min=3, y_max=103)
        outputs = dict(object=torch.randn(1, 4, 2, requires_grad=True),
                       category=torch.randn(1, 4, 21, requires_grad=True),
                       coordinates=torch.randn(1, 4, 5, 2, requires_grad=True),
                       visibility=torch.randn(1, 4, 5, requires_grad=True))
        losses = loss(outputs, [[]], [torch.empty(0, dtype=torch.long)])
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(outputs['object'].grad).all())

    def test_postprocess_exports_sorted_ego_coordinates(self):
        model = object.__new__(LaneDirect3D)
        torch.nn.Module.__init__(model)
        model.config = dict(score_threshold=.7, direct_visibility_threshold=.5)
        model.loss = Direct3DLoss(points=3, y_min=3, y_max=103)
        outputs = dict(object=torch.tensor([[[12., -12.]]]),
                       category=torch.tensor([[[-12., 12.] + [-12.] * 19]]),
                       coordinates=torch.tensor([[[[1., .1], [2., .2], [3., .3]]]]),
                       visibility=torch.tensor([[[12., -12., 12.]]]))
        result = model.postprocess(outputs)
        self.assertEqual(result['lane_points_frame'], 'ego')
        np.testing.assert_allclose(result['lane_points'][0][0],
                                   [[1., 3., .1], [3., 103., .3]], rtol=1e-6)
        self.assertEqual(result['lane_attris'], [[1]])

    def test_head_shapes_and_coordinate_bounds(self):
        head = Direct3DHead(8, queries=4, points=6, x_scale=10, z_scale=5)
        output = head(torch.randn(2, 4, 8))
        self.assertEqual(output['coordinates'].shape, (2, 4, 6, 2))
        self.assertLessEqual(output['coordinates'][..., 0].abs().max(), 10)
        self.assertLessEqual(output['coordinates'][..., 1].abs().max(), 5)


if __name__ == '__main__':
    unittest.main()
