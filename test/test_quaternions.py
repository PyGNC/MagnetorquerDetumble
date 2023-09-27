import unittest
import numpy as np
from .context import magnetorquer_detumble
Quaternion = magnetorquer_detumble.Quaternion


class TestQuaternionLibrary(unittest.TestCase):

    def test_hat(self):
        vectors = [
            np.array([1, 0, 0]),
            np.array([0.3, 0.3, 0.6]),
            np.array([0.4, 0.1, -0.7])
        ]

        for v in vectors:
            for w in vectors:
                np.testing.assert_array_almost_equal(
                    np.cross(v, w),
                    np.dot(Quaternion.hat(v), w)
                )

    def test_quaternion_to_matrix_regression(self):
        q1 = np.array([0.70710678, 0.70710678, 0, 0])
        np.testing.assert_array_almost_equal(
            Quaternion.quaternion_to_matrix(q1),
            np.array([[1., 0., 0.],
                      [0., 1., 1.],
                      [0., 1., 1.]])
        )

        q2 = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_almost_equal(
            Quaternion.quaternion_to_matrix(q2),
            np.array([[1., 0.125, 0.125],
                      [0.125, 1., 0.125],
                      [0.125, 0.125, 1.]])
        )

        q3 = np.array([0.64054375, 0.25172911, 0.55149359, 0.47137138])
        np.testing.assert_almost_equal(
            Quaternion.quaternion_to_matrix(q3),
            np.array([[1., 0.44438196, 0.60829036],
                      [0.44438196, 1., 0.12673509],
                      [0.60829036, 0.12673509, 1.]])
        )

        q4 = np.array([1, 0, 0, 0])
        np.testing.assert_almost_equal(
            Quaternion.quaternion_to_matrix(q4),
            np.eye(3)
        )

    def test_qmul(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([1, 2, 4, 6])
        q3 = np.array([-3, 1, -1, 7])

        np.testing.assert_array_almost_equal(
            Quaternion.qmul(q1, q2),
            q2
        )

        np.testing.assert_array_almost_equal(
            Quaternion.qmul(q1, q3),
            q3
        )

        np.testing.assert_array_almost_equal(
            Quaternion.qmul(q2, q3),
            np.array([-43, 29, -21, -17])
        )

    def test_rotate_vector_by_quaternion(self):
        q1 = Quaternion.identity()
        v = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(
            Quaternion.rotate_vector_by_quaternion(q1, v),
            v
        )

    def test_qdot(self):
        q1 = np.array([0.5, 0.5, 0, 0])
        v1 = np.array([0.3, 0.4, 0.5])

        np.testing.assert_array_almost_equal(
            Quaternion.qdot(q1, v1),
            np.array([-0.075, 0.075, -0.024999999999999994, 0.225])
        )
