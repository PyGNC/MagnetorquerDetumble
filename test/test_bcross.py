from .context import magnetorquer_detumble
# import magnetorquer_detumble

import numpy as np
import unittest


class TestBCross(unittest.TestCase):

    def test_gain(self):

        k = magnetorquer_detumble.bcross.Controller._bcross_gain(1, np.pi/4, 0.1)
        self.assertGreater(k, 0.0)

        k = magnetorquer_detumble.bcross.Controller._bcross_gain(1, -np.pi/4, 0.1)
        self.assertGreater(k, 0.0)

        k = magnetorquer_detumble.bcross.Controller._bcross_gain(1e4, np.pi/4, 0.001)
        self.assertGreater(k, 0.0)

    def test_minimum_inertia(self):
        J = np.eye(3)
        Jmin = magnetorquer_detumble.bcross.Controller._compute_minimum_inertia_moment(J)
        self.assertAlmostEqual(Jmin, 1.0)

        J = np.array([
            [0.097487,     -0.00114518, 0.000315221],
            [-0.00114518,    0.0945149,  6.5994e-6],
            [0.000315221,   6.5994e-6,  0.0989731]
        ])
        Jmin = magnetorquer_detumble.bcross.Controller._compute_minimum_inertia_moment(J)
        self.assertGreater(Jmin, 0.0)
        self.assertLess(Jmin, 1.0)


if __name__ == "__main__":
    unittest.main()
