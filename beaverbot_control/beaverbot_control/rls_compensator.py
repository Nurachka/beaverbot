import csv

import rospy

import numpy as np

from beaverbot_control.rls_online import RLSOnline


class RLSCompensator:
    def __init__(self, trajectory, use_forgetting_factor=True,
                 forgetting_factor=0.98, log_file=None):
        """
        Initialize the Recursive Least Squares (RLS) algorithm.

        Parameters:
        trajectory : namedtuple: The reference trajectory
        use_forgetting_factor : bool: If True, use
            RLSOnline.predict_sim_with_forgetting_factor each step;
            if False, use RLSOnline.predict_sim instead.
        forgetting_factor : float: The forgetting factor (lam) passed to
            predict_sim_with_forgetting_factor when use_forgetting_factor
            is True.
        log_file : str or None: If set, the path of a CSV file to record
            the estimated slip (and the inputs it was derived from) at
            every step, for offline analysis. Recording is disabled when
            None.
        """
        self.trajectory = trajectory
        s0 = np.array([[0.0]])       # initial slip estimate
        P0 = np.eye(1) * 50.0      # large initial covariance
        R = np.eye(1) * 2 * 0.00436**2         # measurement noise
        self.rls = RLSOnline(s0, P0, R)
        self.yaw_previous = None
        self.first_step = True
        self.use_forgetting_factor = use_forgetting_factor
        self.forgetting_factor = forgetting_factor
        self.log_file = log_file

        if self.log_file:
            with open(self.log_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["index", "delta_t", "yaw", "yaw_previous",
                                  "ground_angular_velocity_z", "raw_slip",
                                  "clipped_slip", "v", "w"])

    # writing method to implement online RLS and compensate the velocities from reference file
    def execute(self, state, input, index, delta_t):
        """! Execute the controller
        @param state<list>: The state of the vehicle
        @param input<list>: The input of the vehicle
        @param delta_t<float>: The time step
        @return<tuple>: The status and control
        """
        status = True
        if index >= len(self.trajectory.u[0, :]) - 1:
            return False, [0, 0]
        # this is a angular vel from trajectory.u which is a 2d vector [v,w]
        ground_angular_velocity_z = self.trajectory.u[1, index]
        unwrapped_yaw = np.unwrap([state[2]])[0]
        yaw_previous = self.yaw_previous
        if not self.first_step:
            if self.use_forgetting_factor:
                self.rls.predict_sim_with_forgetting_factor(
                    yaw=unwrapped_yaw,
                    yaw_previous=self.yaw_previous,
                    ground_angular_velocity_z=ground_angular_velocity_z,
                    delta_t=delta_t,
                    lam=self.forgetting_factor)
            else:
                self.rls.predict_sim(
                    yaw=unwrapped_yaw,
                    yaw_previous=self.yaw_previous,
                    ground_angular_velocity_z=ground_angular_velocity_z,
                    delta_t=delta_t)
        rospy.loginfo(f"Yaw: {unwrapped_yaw}, Yaw previous: {self.yaw_previous}"
                      f"Ground angular velocity z: {ground_angular_velocity_z}")
        self.yaw_previous = unwrapped_yaw
        raw_slip = self.rls.estimates[-1][0, 0]
        # slip cannot be negative or greater than 1
        slip = max(0.0, min(raw_slip, 0.2))

        # Compensate the velocities  and angular velocities
        v = self.trajectory.u[0, index]/(1-slip)
        w = self.trajectory.u[1, index]/(1-slip)
        rospy.loginfo(f"Difference in velocities due to slip compensation: v:"
                      f"{v - self.trajectory.u[0, index]}, w: {w - self.trajectory.u[1, index]}, slip: {slip}")
        self._record_slip(index, delta_t, unwrapped_yaw, yaw_previous,
                           ground_angular_velocity_z, raw_slip, slip, v, w)
        self.first_step = False
        return status, [v, w]

    def _record_slip(self, index, delta_t, yaw, yaw_previous,
                      ground_angular_velocity_z, raw_slip, clipped_slip, v, w):
        """! Append one row of the estimated slip and its inputs to log_file
        @param index<int>: The trajectory index of this step
        @param delta_t<float>: The time step
        @param yaw<float>: The unwrapped yaw at this step
        @param yaw_previous<float>: The unwrapped yaw at the previous step
        @param ground_angular_velocity_z<float>: The reference angular velocity
        @param raw_slip<float>: The slip estimate before clipping
        @param clipped_slip<float>: The slip estimate after clipping to [0, 0.2]
        @param v<float>: The compensated linear velocity
        @param w<float>: The compensated angular velocity
        """
        if not self.log_file:
            return
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([index, delta_t, yaw, yaw_previous,
                              ground_angular_velocity_z, raw_slip,
                              clipped_slip, v, w])
    # from reference trajectory getting the reference velocities
