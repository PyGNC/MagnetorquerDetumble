try:
    import ulab.numpy as np
except:
    import numpy as np

GM_EARTH = 3.986004415e14  # TODO: move this to a constants repository


class Controller:
    """
    Implementation of the B-Cross detumble controller

    References:
    Magnetic Detumbling of a Rigid Spacecraft, Avanzini and Giulietti 2012
    Fundamentals of Spacecraft Attitude Determination and Control, Markley and Crassidis 2014
    """

    def __init__(self,
                 semi_major_axis,
                 inclination,
                 minimum_inertia_moment,
                 maximum_dipoles,
                 output_range):
        self.k_gain = self._bcross_gain(semi_major_axis, inclination, minimum_inertia_moment)
        self.maximum_dipoles = np.array(maximum_dipoles)
        self.output_range = np.array(output_range)

    @staticmethod
    def _compute_minimum_inertia_moment(inertia_matrix):
        return np.min(np.linalg.eigvals(inertia_matrix))

    @staticmethod
    def _bcross_gain(semi_major_axis, inclination, minimum_inertia_moment):
        """ bcross_gain
        Optimal gain for bcross_control, according to eq 30 in Avanzini and Giulietti 2012.
        We assume the Earth's magnetic dipole is aligned with its poles, 
        so the magnetic inclination == the orbit inclination
        """
        Jmin = minimum_inertia_moment
        Omega = 1 / np.sqrt(semi_major_axis**3 / GM_EARTH)
        xi_m = inclination

        k_bcross = 2 * Omega * (1 + np.sin(xi_m)) * Jmin  # equation 30
        return k_bcross

    def _bcross_control(self, angular_rate, magnetic_vector_body, k_gain=None):
        """ bcross_control(x, epc)
        Detumble controller
        See Markley and Crassidis eq 7.48, p 308

        :param angular_rate_body: the current angular rate in the body frame with units rad/s
        :param magnetic_vector_body: the current magnetic field measurement in the body frame, units in Tesla (T)
        :param k_gain: the controller gain; set to None to use the optimal default

        :return control_dipole: the dipole to produce, units in Am^2
        """
        if k_gain is None:
            k_gain = self.k_gain
        omega = angular_rate
        B = magnetic_vector_body
        Bnorm = np.norm(B)
        b = B / Bnorm
        m = (k_gain / Bnorm) * np.cross(omega, b)
        control_dipole = m
        return control_dipole

    def _saturate_dipole(self, control_dipole):
        """
        Clip the minimum and maximum values of `control_dipole` to be in the range
        [-self.maximum_dipoles, self.maximum_dipoles].

        :param control_dipole: the raw control dipole computed by _bcross_control
        """
        return np.clip(control_dipole, -self.maximum_dipoles, self.maximum_dipoles)

    def _scale_dipole(self, saturated_control_dipole):
        """
        Rescale `saturated_control_dipole` to be in the range set by self.output_range.
        The input should be saturated with `_saturate_dipole` prior to calling this function.
        The return value will be in the range [-self.output_range, self.output_range]
        """
        return (saturated_control_dipole / self.maximum_dipoles) * self.output_range

    def get_control(self, angular_rate_body, magnetic_vector_body, k_gain=None):
        """
        Get the control command for producing a magnetic dipole that will detumble the satellite.

        :param angular_rate_body: the current angular rate in the body frame with units rad/s
        :param magnetic_vector_body: the current magnetic field measurement in the body frame, units in Tesla (T)
        :param k_gain: the controller gain; set to None to use the optimal default

        :return scaled_control_dipole: the computed control dipole, in the body frame, scaled to [-output_range, output_range]
        """
        control_dipole = self._bcross_control(angular_rate_body, magnetic_vector_body, k_gain=k_gain)
        saturated_control_dipole = self._saturate_dipole(control_dipole)
        scaled_control_dipole = self._scale_dipole(saturated_control_dipole)
        return scaled_control_dipole
