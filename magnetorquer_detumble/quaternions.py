try:
    import ulab.numpy as np
except:
    import numpy as np


class Quaternion:

    def __init__(self) -> None:
        pass

    @staticmethod
    def hat(v):
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

    @staticmethod
    def quaternion_to_matrix(q):
        """
        quaternion_to_matrix(q)

        Convert a quaternion to a rotation matrix representing the same rotation.
        """

        s, v = q[0], q[1:]
        return np.eye(3) + 2 * Quaternion.hat(v) * (s * np.eye(3) + Quaternion.hat(v))

    @staticmethod
    def qmul(a, b):
        """
        qmul(a,b)

        Multiply two scalar first quaternions a and b.
        """
        return np.array([
            a[0] * b[0] - np.dot(a[1:], b[1:]),
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        ])

    @staticmethod
    def qdot(q, w):
        """
        qdot(q, w)

        Returns the time derivative of a quaternion q given angular velocity w.
        """

        return 0.5 * Quaternion.qmul(q, np.concatenate([[0], w]))

    @staticmethod
    def integrate_quaternion_with_angular_velocity(q, w, t):
        """
        integrate_quaternion_with_angular_velocity(q, w, t)

        Integrate a quaternion with angular velocity w over time t.
        Normalize the quaternion after integration.
        """

        dq = Quaternion.qdot(q, w) * t
        q += dq
        q /= np.linalg.norm(q)
        return q

    @staticmethod
    def identity():
        return np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def rotate_vector_by_quaternion(q, v):
        """
        rotate_vector_by_quaternion(q, v)

        Rotate a vector v by a quaternion q.
        """

        return np.dot(Quaternion.quaternion_to_matrix(q), v)
