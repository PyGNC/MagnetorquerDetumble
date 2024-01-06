
try:
    import ulab.numpy as np
except:
    import numpy as np


class PracticalController:
    """
    Practical implementation of the B-Cross detumble controller.

    * Accounts for the magnetic torque coils saturating the magnetometer
    """

    def __init__(self, maximum_dipoles, output_range,
        mag_data_body, gyro_data_body, sun_vector_body, # sensor data pointers
        bias_calibration_gyro_threshold=6*np.pi,
        which_controller=1,spin_major_axis=1, # default sun controller, spin major axis
        inertia_ovride=None,major_minor_ovride=None, # sun pointing overrides
        h_target=0.2*0.005,umax=1,sun_err=(0.26,0.17)): # sun pointing bounds
        """
        :param maximum_dipoles: the maximum dipole the satellite can produce, units in Am^2
        :param output_range: the maximum output values to rescale the control dipole to
        :param mag_data_body: the current magnetic field measurement in the body frame, units in Tesla (T)
        :param gyro_data_body: the current angular rate in the body frame with units rad/s
        :param bias_calibration_gyro_threshold: the angle the gyro measurements should integrate to for calibration to be considered complete
        """
        self.output_range = np.array(output_range)
        self.maximum_dipoles = np.array(maximum_dipoles)
        self.mag_data_body = mag_data_body # reference to mag_data measurement in body frame
        self.gyro_data_body = gyro_data_body # reference to gyro_data measurement in body frame
        self.sun_vector_body = sun_vector_body
        self.magnetic_vector_body = np.zeros(3) # best estimate of B-vector in body frame
        self.new_mag = True # flag indicating new magnetometer measurements
        self.new_sun = True

        # members used for calibrating magnetometer bias
        self.mag_bias_accumulator = np.zeros(3)
        self.mag_bias = np.zeros(3)
        self.mag_bias_samples = 0
        self.gyro_accumulator = np.zeros(3)
        self.bias_calibration_gyro_threshold = bias_calibration_gyro_threshold
        self.mag_bias_estimate_complete = False

        # members used for sun-pointing controller
        self.inertia_matrix = np.array([[0.0043, -0.0003, 0.0],[-0.0003, 0.0049, 0.0],[0.0, 0.0, 0.0035]])
        if inertia_ovride is not None: # override
            self.inertia_matrix = np.array(inertia_ovride)
        self.major_axis = np.array([0.3503, 0.9365, 0.0152])
        self.minor_axis = np.array([-0.0011, -0.0158, 0.9999])
        if major_minor_ovride is not None: # override
            self.major_axis = np.array(major_minor_ovride[0])
            self.minor_axis = np.array(major_minor_ovride[1])
        self.which_controller=which_controller
        self.desired_spin_axis=self.major_axis if spin_major_axis else self.minor_axis
        self.h_target=h_target
         # kept umax var (set to 1) in case we want tweak sun_point output at runtime
        self.umax=umax
        self.sun_err=sun_err
        # external stop flag
        self.sun_seeking = True

    def calculate_control(self,maximum_dipoles, angular_rate_body, magnetic_vector_body, which_controller):
        if which_controller == 1: # sun pointing
            control_dipole = self._sun_point_control(
                angular_rate_body, magnetic_vector_body,maximum_dipoles)
        elif which_controller == 0: # detumble
            control_dipole = self._bcross_control(
                angular_rate_body, magnetic_vector_body, 1) # k = 1 Doesn't matter, we're just saturating

        scale_factor = np.min(abs(maximum_dipoles / control_dipole))

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

    def _sun_point_control(self, angular_rate, magnetic_vector_body, maximum_dipoles):
        # Calculate body-frame angular momentum vector
        h = np.dot(self.inertia_matrix,angular_rate)

        # Calculate normalized magnetic field vector
        b = magnetic_vector_body/np.linalg.norm(magnetic_vector_body)

        # Calculate normalized sun vector
        s = self.sun_vector_body/np.linalg.norm(self.sun_vector_body)

        if np.linalg.norm(self.desired_spin_axis - h/self.h_target) > self.sun_err[0]:
            print(f'\ts1:{np.linalg.norm(self.desired_spin_axis - h/self.h_target)} > {self.sun_err[0]}') # TODO DEBUG REMOVE THIS
            u = np.dot(skew(b),(self.desired_spin_axis-h/self.h_target))
        elif np.linalg.norm(s - h/np.linalg.norm(h)) > self.sun_err[1]:
            print(f'\ts2:{np.linalg.norm(s - h/np.linalg.norm(h))} > {self.sun_err[1]}') # TODO DEBUG REMOVE THIS
            u = np.dot(skew(b),(s-h/np.linalg.norm(h)))
        else:
            # set flag for external stop
            self.sun_seeking = False
            return np.zeros(3)

        return self.umax*maximum_dipoles*u/np.linalg.norm(u)


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

    def clear_bias_estimate(self):
        """
        Clear the magnetometer and sun bias estimate
        """
        self.mag_bias_accumulator = np.zeros(3)
        self.mag_bias = np.zeros(3)
        self.mag_bias_samples = 0
        self.gyro_accumulator = np.zeros(3)
        self.mag_bias_estimate_complete = False

    def update_bias_estimate(self, dt):
        """
        :param dt: the time step since the last call to update_bias, units in seconds (s)
        sets self.mag_bias_estimate_complete to True if bias estimate is complete, and False if more updates needed
        """
        self.mag_bias_accumulator += self.mag_data_body
        self.mag_bias_samples += 1
        self.mag_bias[:] = self.mag_bias_accumulator / self.mag_bias_samples

        self.gyro_accumulator += dt*self.gyro_data_body
        gyro_accumulator_magnitude = np.sqrt(np.dot(self.gyro_accumulator, self.gyro_accumulator))

        if gyro_accumulator_magnitude > self.bias_calibration_gyro_threshold:
            self.mag_bias_estimate_complete = True # bias estimate is complete
        else:
            self.mag_bias_estimate_complete = False # bias estimate is not complete

        return self.mag_bias_estimate_complete

    def get_control(self, dt):
        """
        :param dt: the time step since the last call to get_control, units in seconds (s)
        """
        if self.new_mag:
            self.new_mag = False
            # mag data has updated - subtract bias and save it
            self.magnetic_vector_body = self.mag_data_body - self.mag_bias
        else:
            # propagate magnetic vector into current body frame using gyro data
            propagation_matrix = np.eye(3) + skew(self.gyro_data_body * dt)
            self.magnetic_vector_body = np.dot(
                propagation_matrix.transpose(), self.magnetic_vector_body)
        # no sun bias but still propagate it bt readings
        if self.which_controller == 1: # sun pointing
            if self.new_sun:
                self.new_sun = False
            else:
                # propagate sun vector into current body frame using gyro data
                propagation_matrix = np.eye(3) + skew(self.gyro_data_body * dt)
                self.sun_vector_body[:] = np.dot(
                    propagation_matrix.transpose(), self.sun_vector_body)
        print(f'\tsun:{self.sun_vector_body.tolist()}, mag:{self.magnetic_vector_body}, gyr: {self.gyro_data_body}') # TODO DEBUG REMOVE THIS

        control = self.calculate_control(
            self.maximum_dipoles, self.gyro_data_body, self.magnetic_vector_body, self.which_controller)
        control = PracticalController._scale_dipole(
            control, self.maximum_dipoles, self.output_range)
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
