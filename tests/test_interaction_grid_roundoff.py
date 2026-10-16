"""Regression: nearly identical independent transform grids are safe at GL refinement."""

import unittest

import numpy as np

from quartic2d._numerics import (
    common_grid,
    composite_gauss_nodes_weights,
    subdivide_grid,
)


class NearDuplicateGridTests(unittest.TestCase):
    def test_ulps_separated_hankel_grids(self):
        q1 = np.array([0.0, 0.025, 0.05])
        q2 = q1.copy()
        q2[1] = np.nextafter(q2[1], np.inf)
        merged = common_grid(q1, q2)
        self.assertTrue(np.all(np.diff(merged) > 0.0))
        self.assertEqual(merged.size, 3)
        for subdivisions in (1, 2, 4, 8, 16, 64):
            panels = subdivide_grid(merged, subdivisions)
            self.assertTrue(np.all(np.diff(panels) > 0.0))
            nodes, weights = composite_gauss_nodes_weights(panels, 4)
            self.assertTrue(np.all(np.isfinite(nodes)))
            self.assertTrue(np.all(weights > 0.0))
            self.assertAlmostEqual(float(weights.sum()), q1[-1], places=13)

    def test_distinct_grid_nodes_are_retained(self):
        q1 = np.array([0.0, 0.01, 0.03])
        q2 = np.array([0.0, 0.02, 0.03])
        np.testing.assert_array_equal(common_grid(q1, q2),
                                      np.array([0.0, 0.01, 0.02, 0.03]))

    def test_subdivision_rounding_cannot_repeat_nodes(self):
        # Exercise the subdivision guard independently of common_grid.
        grid = np.array([0.0, 0.025,
                         np.nextafter(0.025, np.inf), 0.05])
        panels = subdivide_grid(grid, 16)
        self.assertTrue(np.all(np.diff(panels) > 0.0))
        composite_gauss_nodes_weights(panels, 4)


if __name__ == "__main__":
    unittest.main()
