import unittest
import numpy as np
from .context import magnetorquer_detumble

skew = magnetorquer_detumble.practical.skew
PC = magnetorquer_detumble.practical.PracticalController


class TestPraticalController(unittest.TestCase):

    def test_skew(self):
        vectors = [
            np.array([1, 0, 0]),
            np.array([0.3, 0.3, 0.6]),
            np.array([0.4, 0.1, -0.7])
        ]

        for v in vectors:
            for w in vectors:
                np.testing.assert_array_almost_equal(
                    np.cross(v, w),
                    np.dot(skew(v), w)
                )

    def test_calculate_control_regression(self):
        np.testing.assert_array_almost_equal(
            PC.calculate_control(
                [1, 2, 3], [0.01, 0.06, -0.03], [0.03, 0.07, -0.09]),
            np.array([-1.00000000e+00,  3.09880124e-17, -3.33333333e-01])
        )

    def test_bcross_control_regression(self):
        np.testing.assert_array_almost_equal(
            PC._bcross_control([0.05, -0.0213, 0.3], [0.1, 0.2, 0.4], 1.2),
            np.array([-0.39154286,  0.05714286,  0.06931429])
        )

    def test_scale_dipole_regression(self):
        np.testing.assert_array_almost_equal(
            PC._scale_dipole(np.array([0.5, 0.3, 0.7]), np.array(
                [1, 2, 3]), np.array([2, 1, 3])),
            np.array([1., 0.15, 0.7])
        )
