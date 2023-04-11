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
                 output_range,
                 k_gain=None):
        """
        Set up the constant parameters for the B-Cross controller.

        :param semi_major_axis: The semi-major axis of the orbit, units in meters (m)
        :param inclination: The inclination of the orbit, units in radians (rad)
        :param minimum_inertia_moment: the minimum moment of inertia of the satellite, units in kg*m^2
        :param maximum_dipoles: the maximum dipole the satellite can produce, units in Am^2
        :param output_range: the maximum output values to rescale the control dipole to
        :param k_gain: the controller gain parameter, leave as None for "optimal" default
        """

        if k_gain is None:
            # default to optimal gain
            self.k_gain = self._bcross_gain(semi_major_axis, inclination, minimum_inertia_moment)
        else:
            self.k_gain = k_gain

        self.maximum_dipoles = np.array(maximum_dipoles)
        self.output_range = np.array(output_range)

    @staticmethod
    def _compute_minimum_inertia_moment(inertia_matrix):
        """
        Given an inertia matrix, compute the minimum moment of inertia.
        This is the minimum eigenvalue of the inertia matrix.

        :param inertia_matrix: The inertia matrix for a rigid-body
        :return: the minimum moment of inertia
        """
        return np.min(np.linalg.eigvals(inertia_matrix))

    @staticmethod
    def _bcross_gain(semi_major_axis, inclination, minimum_inertia_moment):
        """ bcross_gain
        Optimal gain for bcross_control, according to eq 30 in Avanzini and Giulietti 2012.
        We assume the Earth's magnetic dipole is aligned with its poles, 
        so the magnetic inclination == the orbit inclination

        :param semi_major_axis: The semi-major axis of the orbit, units in meters (m)
        :param inclination: The inclination of the orbit, units in radians (rad)
        :param minimum_inertia_moment: the minimum moment of inertia of the satellite, units in kg*m^2

        :return k_gain: computed gain
        """
        Jmin = minimum_inertia_moment
        Omega = 1 / np.sqrt(semi_major_axis**3 / GM_EARTH)
        xi_m = inclination

        k_gain = 2 * Omega * (1 + np.sin(xi_m)) * Jmin  # equation 30
        return k_gain

    @staticmethod
    def _bcross_control(angular_rate, magnetic_vector_body, k_gain):
        """ bcross_control(x, epc)
        Detumble controller
        See Markley and Crassidis eq 7.48, p 308

        :param angular_rate_body: the current angular rate in the body frame with units rad/s
        :param magnetic_vector_body: the current magnetic field measurement in the body frame, units in Tesla (T)
        :param k_gain: the controller gain; set to None to use the optimal default

        :return control_dipole: the dipole to produce, units in Am^2
        """
        omega = angular_rate
        B = magnetic_vector_body
        Bnorm = np.linalg.norm(B)
        b = B / Bnorm
        m = (k_gain / Bnorm) * np.cross(omega, b)
        control_dipole = m
        return control_dipole

    @staticmethod
    def _saturate_dipole(control_dipole, maximum_dipoles):
        """
        Clip the minimum and maximum values of `control_dipole` to be in the range
        [-self.maximum_dipoles, self.maximum_dipoles].

        :param control_dipole: the raw control dipole computed by _bcross_control
        :param maximum_dipoles: the maximum dipole the satellite can produce, units in Am^2
        """
        return np.clip(control_dipole, -maximum_dipoles, maximum_dipoles)

    @staticmethod
    def _scale_dipole(saturated_control_dipole, maximum_dipoles, output_range):
        """
        Rescale `saturated_control_dipole` to be in the range set by self.output_range.
        The input should be saturated with `_saturate_dipole` prior to calling this function.
        The return value will be in the range [-self.output_range, self.output_range]

        :param saturated_control_dipole: the saturated control dipole computed by _saturate_dipole
        :param maximum_dipoles: the maximum dipole the satellite can produce, units in Am^2
        :param output_range: the maximum output values to rescale the control dipole to
        """
        return (saturated_control_dipole / maximum_dipoles) * output_range

    def get_control(self, angular_rate_body, magnetic_vector_body):
        """
        Get the control command for producing a magnetic dipole that will detumble the satellite.

        :param angular_rate_body: the current angular rate in the body frame with units rad/s
        :param magnetic_vector_body: the current magnetic field measurement in the body frame, units in Tesla (T)
        :param k_gain: the controller gain; set to None to use the optimal default

        :return scaled_control_dipole: the computed control dipole, in the body frame, scaled to [-output_range, output_range]
        """
        control_dipole = self._bcross_control(angular_rate_body, magnetic_vector_body, self.k_gain)
        saturated_control_dipole = self._saturate_dipole(control_dipole, self.maximum_dipoles)
        scaled_control_dipole = self._scale_dipole(saturated_control_dipole, self.maximum_dipoles, self.output_range)
        return scaled_control_dipole
