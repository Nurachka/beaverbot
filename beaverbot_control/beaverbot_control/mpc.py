#!/usr/bin/env python3
##
# @file mpc.py
#
# @brief Provide implementation of a linear MPC controller with a fixed
# (known) slip factor for autonomous driving.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 08/12/2024.
#
# Copyright (c) 2024 System Engineering Laboratory.  All rights reserved.

# External library
import numpy as np

# Internal library
from beaverbot_control.linear_mpc import LinearMPC


class MPC:
    """! MPC controller

    The class provides implementation of a linear MPC controller that
    corrects wheel-velocity references from the reference trajectory,
    assuming a fixed (known) wheel slip factor.

    Requires the trajectory to be built with trajectory_type="wheel", so
    that trajectory.u[0, :] / trajectory.u[1, :] hold [v, w] with
    w = angular_velocity_sign * (vr - vl) / wheel_base (see
    BeaverbotControl._retrieve_u's "wheel" branch and its
    ~angular_velocity_sign rosparam). Default angular_velocity_sign=-1
    matches what the "feedforward" controller already publishes to
    cmd_vel and has been validated on the real robot. The raw per-wheel
    references (vr_ref, vl_ref) are recovered from [v, w], and MPC's own
    wheel-velocity corrections are converted back to [v, w] in the same
    convention before being returned, so cmd_vel stays directly
    comparable across the feedforward / mpc / rls_compensator controllers.
    """
    # ==================================================================================================
    # PUBLIC METHODS
    # ==================================================================================================
    def __init__(self, trajectory, wheel_base, sampling_time, N_horizon=10,
                 slip=0.0, vr_max=0.5, vl_max=0.5, du_max=0.05,
                 angular_velocity_sign=-1):
        """! Constructor
        @param trajectory<instance>: The trajectory
        @param wheel_base<float>: Distance between the wheels of the robot.
        @param sampling_time<float>: Time step of the control loop.
        @param N_horizon<int>: Number of time steps in the prediction horizon.
        @param slip<float>: Fixed slip factor used by the MPC's system model.
        @param vr_max<float>: Maximum velocity of the right wheel.
        @param vl_max<float>: Maximum velocity of the left wheel.
        @param du_max<float>: Maximum allowed change in each wheel's
        delta-velocity per step.
        @param angular_velocity_sign<int>: Sign relating trajectory.u's w
        to the standard (vr - vl) / wheel_base convention; must match
        BeaverbotControl's ~angular_velocity_sign.
        """
        self.trajectory = trajectory

        self._angular_velocity_sign = angular_velocity_sign

        self._mpc = LinearMPC(dt=sampling_time, wheel_base=wheel_base,
                              N_horizon=N_horizon, vr_max=vr_max, vl_max=vl_max,
                              s=slip, du_max=du_max)

    def execute(self, state, input, index, delta_t):
        """! Execute the controller
        @param state<list>: The state of the vehicle
        @param input<list>: The input of the vehicle
        @param index<int>: The index
        @param delta_t<float>: The time step
        @return<tuple>: The status and control
        """
        last_index = len(self.trajectory.u[0, :]) - 1

        if index >= last_index:
            return False, [0, 0]

        error_state = self._mpc.compute_error_state(
            np.array(state, dtype=float), np.array(self.trajectory.x[index], dtype=float))

        A_list, B_list, vr_ref_horizon, vl_ref_horizon = self._build_horizon(index, last_index)

        delta_vr, delta_vl = self._mpc.solve(
            error_state, A_list, B_list, vr_ref_horizon, vl_ref_horizon)

        vr_ref, vl_ref = self._wheel_velocities(index)

        vr_cmd = vr_ref + delta_vr

        vl_cmd = vl_ref + delta_vl

        v = (vr_cmd + vl_cmd) / 2

        w = self._angular_velocity_sign * (vr_cmd - vl_cmd) / self._mpc.l

        return True, [v, w]

    # ==================================================================================================
    # PRIVATE METHODS
    # ==================================================================================================
    def _wheel_velocities(self, index):
        """! Recover the raw right/left wheel-velocity references at the
        given trajectory index from trajectory.u = [v, w] (see class
        docstring for the angular_velocity_sign convention).
        @param index<int>: The trajectory index
        @return<tuple>: The right and left wheel-velocity references
        (vr_ref, vl_ref)
        """
        v_ref = self.trajectory.u[0, index]

        w_ref = self.trajectory.u[1, index]

        vr_ref = v_ref + self._angular_velocity_sign * w_ref * self._mpc.l / 2

        vl_ref = v_ref - self._angular_velocity_sign * w_ref * self._mpc.l / 2

        return vr_ref, vl_ref

    def _build_horizon(self, index, last_index):
        """! Build the linearized system matrices and reference
        wheel-velocity horizons starting at the given index.
        @param index<int>: The current trajectory index
        @param last_index<int>: The last valid trajectory index
        @return<tuple>: A_list, B_list, vr_ref_horizon, vl_ref_horizon
        """
        A_list, B_list = [], []

        vr_ref_horizon, vl_ref_horizon = [], []

        for i in range(self._mpc.N):
            future_index = min(index + i, last_index)

            theta_ref = self.trajectory.x[future_index, 2]

            vr_ref, vl_ref = self._wheel_velocities(future_index)

            A_i, B_i = self._mpc.define_AB_matrices(theta_ref, vr_ref, vl_ref)

            A_list.append(A_i)

            B_list.append(B_i)

            vr_ref_horizon.append(vr_ref)

            vl_ref_horizon.append(vl_ref)

        return A_list, B_list, vr_ref_horizon, vl_ref_horizon
