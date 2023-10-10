
try:
    import ulab.numpy as np
except:
    import numpy as np


class PracticalController:
    """
    Practical implementation of the B-Cross detumble controller.

    * Accounts for the magnetic torque coils saturating the magnetometer
    * Does not use linear feedback, instead just saturates the control dipole
    """

    def __init__(self, maximum_dipoles, output_range, sense_time=5.0, actuate_time=5.0):
        """
        :param maximum_dipoles: the maximum dipole the satellite can produce, units in Am^2 
        :param output_range: the maximum output values to rescale the control dipole to
        :param sense_time: the time taken to sense the magnetic field, while not actuating, units in seconds (s)
        :param actuate_time: the time taken to use the magnetic torque coils, while not sensing, units in seconds (s)
        """
        self.output_range = np.array(output_range)
        self.maximum_dipoles = np.array(maximum_dipoles)
        self.sense_time = sense_time
        self.actuate_time = actuate_time

        self.mode = None
        self.timer = 0.0

    @staticmethod
    def calculate_control(maximum_dipoles, angular_rate_body, magnetic_vector_body):
        k = 1.0  # Doesn't matter, we're just saturating
        control_dipole = PracticalController._bcross_control(
            angular_rate_body, magnetic_vector_body, k)
        scale_factor = np.min(np.abs(maximum_dipoles / control_dipole))

        return scale_factor * control_dipole

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

    def get_control(self, angular_rate_body, magnetic_vector_body, dt):
        """
       :param angular_rate_body: the current angular rate in the body frame with units rad/s 
       :param magnetic_vector_body: the current magnetic field measurement in the body frame, units in Tesla (T)
       :param dt: the time step since the last call to get_control, units in seconds (s)
       """
        control = np.zeros(3)
        angular_rate_body = np.array(angular_rate_body)
        magnetic_vector_body = np.array(magnetic_vector_body)
        dt = float(dt)

        if self.mode is None:
            self.mode = 'sense'
            self.timer = self.sense_time
        elif self.mode == 'sense':
            self.timer -= dt

            self.magnetic_vector = magnetic_vector_body

            if self.timer <= 0:
                self.mode = 'actuate'
                self.timer = self.actuate_time
        elif self.mode == 'actuate':
            # In this case we ignore the magnetic vector and use angular velocity to propogate the magnetic vector
            self.timer -= dt

            propogation_matrix = np.eye(3) + skew(angular_rate_body * dt)
            self.magnetic_vector = np.dot(
                propogation_matrix.T, self.magnetic_vector)
            # First order approximation I + hat(omega *dt) for propogating the attitude

            control = self.calculate_control(
                self.maximum_dipoles, angular_rate_body, self.magnetic_vector)
            control = PracticalController._scale_dipole(
                control, self.maximum_dipoles, self.output_range)

            if self.timer <= 0:
                self.mode = 'sense'
                self.timer = self.sense_time

        return control


def skew(v):
    """
    hat(v)

    Convert a vector to a skew-symmetric matrix.
    Such that hat(v) * w = v x w
    """
    return np.array([
        [0, -v[2],  v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
