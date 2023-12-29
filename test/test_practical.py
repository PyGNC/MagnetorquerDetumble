import unittest
import numpy as np
from context import magnetorquer_detumble


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
            #add in new argument for which_controller
            PC.calculate_control(self, 
                [1, 2, 3], [0.01, 0.06, -0.03], [0.03, 0.07, -0.09], 0),
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

    def test_get_control(self):
        mag_data = np.array([1.0, 2.0, 3.0])
        gyro_data = np.array([-0.1, 0.1, 0.1])
        sun_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        controller = PC(np.array([10.0, 11.0, 12.0]), np.array([0.5, 0.6, 0.3]), mag_data, gyro_data, sun_data, 6*np.pi, which_controller=0)
        controller.new_mag = True
        ret = controller.get_control(0.2)
        controller.new_mag = False
        ret = controller.get_control(0.2)
        self.assertEqual(ret.shape[0], 3)
        self.assertEqual(len(ret.shape), 1)

    def test_update_bias(self):
        mag_data = np.array([1.0, 2.0, 3.0])
        gyro_data = np.array([-0.1, 0.1, 0.1])
        sun_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        bias_gyro_angle_threshold = 6*np.pi
        controller = PC(np.array([10.0, 11.0, 12.0]), np.array([0.5, 0.6, 0.3]), mag_data, gyro_data, sun_data, bias_calibration_gyro_threshold=bias_gyro_angle_threshold, which_controller=0)
        ret = controller.update_bias_estimate(0.2)
        np.testing.assert_array_almost_equal(controller.mag_bias, mag_data)
        self.assertFalse(controller.mag_bias_estimate_complete)
        self.assertEqual(controller.mag_bias_samples, 1)

        ret = controller.update_bias_estimate(0.2)
        np.testing.assert_array_almost_equal(controller.mag_bias, mag_data)
        self.assertFalse(controller.mag_bias_estimate_complete)
        self.assertEqual(controller.mag_bias_samples, 2)

        gyro_data += bias_gyro_angle_threshold * np.ones(3)
        ret = controller.update_bias_estimate(1.0)
        np.testing.assert_array_almost_equal(controller.mag_bias, mag_data)
        self.assertTrue(controller.mag_bias_estimate_complete)
        self.assertEqual(controller.mag_bias_samples, 3)

        controller.clear_bias_estimate()
        np.testing.assert_array_almost_equal(controller.mag_bias, np.zeros(3))
        self.assertFalse(controller.mag_bias_estimate_complete)
        np.testing.assert_array_almost_equal(controller.mag_bias_accumulator, np.zeros(3))
        self.assertEqual(controller.mag_bias_samples, 0)
        np.testing.assert_array_almost_equal(controller.gyro_accumulator, np.zeros(3))


if __name__ == "__main__":
    unittest.main()