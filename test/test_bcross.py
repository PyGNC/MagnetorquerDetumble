from .context import magnetorquer_detumble
# import magnetorquer_detumble

import numpy as np
import unittest


class TestBCross(unittest.TestCase):

    def test_gain_1(self):
        k = magnetorquer_detumble.bcross.Controller._bcross_gain(1, np.pi/4, 0.1)
        self.assertGreater(k, 0.0)

    def test_gain_2(self):
        k = magnetorquer_detumble.bcross.Controller._bcross_gain(1, -np.pi/4, 0.1)
        self.assertGreater(k, 0.0)

    def test_gain_3(self):
        k = magnetorquer_detumble.bcross.Controller._bcross_gain(1e4, np.pi/4, 0.001)
        self.assertGreater(k, 0.0)

    def test_minimum_inertia_1(self):
        J = np.eye(3)
        Jmin = magnetorquer_detumble.bcross.Controller._compute_minimum_inertia_moment(J)
        self.assertAlmostEqual(Jmin, 1.0)

    def test_minimum_inertia_2(self):
        J = np.array([
            [0.097487,     -0.00114518, 0.000315221],
            [-0.00114518,    0.0945149,  6.5994e-6],
            [0.000315221,   6.5994e-6,  0.0989731]
        ])
        Jmin = magnetorquer_detumble.bcross.Controller._compute_minimum_inertia_moment(J)
        self.assertGreater(Jmin, 0.0)
        self.assertLess(Jmin, 1.0)

    def test_saturate_dipole_1(self):
        m = 11.77 * np.ones(3)
        m_max = np.ones(3)

        ret = magnetorquer_detumble.bcross.Controller._saturate_dipole(m, m_max, False)
        np.testing.assert_array_almost_equal(ret, m_max)

    def test_saturate_dipole_2(self):
        m = -11.77 * np.ones(3)
        m_max = np.array([10, 9.99, 0.01])

        ret = magnetorquer_detumble.bcross.Controller._saturate_dipole(m, m_max, False)
        np.testing.assert_array_almost_equal(-ret, m_max)

    def test_saturate_dipole_3(self):
        m = np.array([4.32, 0.02, -2.1, -0.99])
        m_max = np.ones(4)

        ret = magnetorquer_detumble.bcross.Controller._saturate_dipole(m, m_max, False)
        np.testing.assert_array_equal(ret, np.array([1.0, 0.02, -1.0, -0.99]))

    def test_saturate_dipole_4(self):
        m = np.zeros(3)
        m_max = np.ones(3)

        ret = magnetorquer_detumble.bcross.Controller._saturate_dipole(m, m_max, False)
        np.testing.assert_array_almost_equal(ret, m)
    
    def test_saturate_dipole_always_saturate(self):
        m = np.array([1, 2, 3])
        m_max = np.array([3, 2, 1])

        ret = magnetorquer_detumble.bcross.Controller._saturate_dipole(m, m_max, True)
        np.testing.assert_array_almost_equal(ret, [1/3, 2/3, 1])

    def test_scale_dipole_1(self):
        m = 0.1 * np.ones(3)
        m_max = np.ones(3)
        output_range = 5.5*np.ones(3)

        ret = magnetorquer_detumble.bcross.Controller._scale_dipole(m, m_max, output_range)
        np.testing.assert_array_almost_equal(ret, 5.5*m)

    def test_scale_dipole_2(self):
        m = 0.1 * np.ones(3)
        m_max = 0.1 * np.ones(3)
        output_range = 5.5*np.ones(3)

        ret = magnetorquer_detumble.bcross.Controller._scale_dipole(m, m_max, output_range)
        np.testing.assert_array_almost_equal(ret, output_range)

    def test_scale_dipole_3(self):
        m = -0.1 * np.ones(3)
        m_max = 0.1 * np.ones(3)
        output_range = 5.5*np.ones(3)

        ret = magnetorquer_detumble.bcross.Controller._scale_dipole(m, m_max, output_range)
        np.testing.assert_array_almost_equal(ret, -output_range)

    def test_bcross_control_1(self):
        omega = np.array([0.0, 1.0, 0.0])
        B = np.array([1.0, 0, 0])
        k_gain = 1.0

        ret = magnetorquer_detumble.bcross.Controller._bcross_control(omega, B, k_gain)
        np.testing.assert_array_almost_equal(ret, np.array([0.0, 0.0, -1.0]))

    def test_bcross_control_2(self):
        omega = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 777.0, 0.0])
        k_gain = 1.0

        ret = magnetorquer_detumble.bcross.Controller._bcross_control(omega, B, k_gain)
        np.testing.assert_array_almost_equal(ret, np.array([0.0, 0.0, 1.0]) / 777.0)

    def test_bcross_control_3(self):
        omega = 53.0 * np.ones(3)
        B = np.array([0.0, 1.0, 0.0])
        k_gain = 1.0

        ret = magnetorquer_detumble.bcross.Controller._bcross_control(omega, B, k_gain)
        self.assertAlmostEqual(np.linalg.norm(ret), np.linalg.norm(omega))

    def test_bcross_control_3(self):
        omega = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 1.0, 0.0])
        k_gain = 7654e-5

        ret = magnetorquer_detumble.bcross.Controller._bcross_control(omega, B, k_gain)
        self.assertAlmostEqual(np.linalg.norm(ret), k_gain)

    def test_get_control(self):

        m_max = 7.55e-2 * np.ones(3)
        output_range = 1.123 * np.ones(3)

        bcc = magnetorquer_detumble.bcross.Controller(
            1e4,
            np.pi/2,
            1e-2,
            m_max,
            output_range,
        )

        self.assertGreater(bcc.k_gain, 0)
        np.testing.assert_array_almost_equal(bcc.maximum_dipoles, m_max)
        np.testing.assert_array_almost_equal(bcc.output_range, output_range)

        omega = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 777.0, 0.0])

        ret = bcc.get_control(omega, B)

        np.testing.assert_array_compare(np.greater_equal, ret, -output_range)
        np.testing.assert_array_compare(np.less_equal, ret, output_range)


if __name__ == "__main__":
    unittest.main()
