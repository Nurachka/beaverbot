#!/usr/bin/env python3
##
# @file mpc_rls.py
#
# @brief Provide implementation of a linear MPC controller whose slip
# factor is estimated online via recursive least squares.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 08/12/2024.
#
# Copyright (c) 2024 System Engineering Laboratory.  All rights reserved.

# External library
import numpy as np

# Internal library
from beaverbot_control.mpc import MPC
from beaverbot_control.rls_online import RLSOnline


class MPCRLS(MPC):
    """! MPC + RLS controller

    The class provides implementation of a linear MPC controller that
    estimates the wheel slip factor online via recursive least squares
    (RLSOnline) from the heading measurement, and feeds it into the MPC's
    linearized system model each step.

    Requires the trajectory to be built with trajectory_type="wheel", so
    that trajectory.u[0, :] / trajectory.u[1, :] hold [v, w] (see MPC).

    Like MPC, ignores the caller-supplied `index` and instead uses the
    nearest-point search (see MPC._search_nearest_index) to pick the
    reference angular velocity fed into the RLS slip estimate, so the
    estimate is computed against where the robot actually is rather than
    the raw elapsed-tick schedule.
    """
    # ==================================================================================================
    # PUBLIC METHODS
    # ==================================================================================================
    def __init__(self, trajectory, wheel_base, sampling_time, N_horizon=10,
                 vr_max=0.5, vl_max=0.5, du_max=0.05,
                 warmup_steps=50, slip_clip=0.5, lam=0.96):
        """! Constructor
        @param trajectory<instance>: The trajectory
        @param wheel_base<float>: Distance between the wheels of the robot.
        @param sampling_time<float>: Time step of the control loop.
        @param N_horizon<int>: Number of time steps in the prediction horizon.
        @param vr_max<float>: Maximum velocity of the right wheel.
        @param vl_max<float>: Maximum velocity of the left wheel.
        @param du_max<float>: Maximum allowed change in each wheel's
        delta-velocity per step.
        @param warmup_steps<int>: Number of initial steps the slip
        estimate is forced to 0.0 while RLS converges.
        @param slip_clip<float>: The estimated slip is clipped to
        [-slip_clip, slip_clip].
        @param lam<float>: Forgetting factor used by the RLS estimator.
        """
        super(MPCRLS, self).__init__(
            trajectory, wheel_base, sampling_time, N_horizon=N_horizon,
            slip=0.0, vr_max=vr_max, vl_max=vl_max, du_max=du_max)

        s0 = np.array([[0.0]])

        P0 = np.eye(1) * 50.0

        R = np.eye(1) * 2 * 0.00436 ** 2

        self._rls = RLSOnline(s0, P0, R)

        self._warmup_steps = warmup_steps

        self._slip_clip = slip_clip

        self._lam = lam

        self._yaw_previous = None

        self._step = 0

    def execute(self, state, input, index, delta_t):
        """! Execute the controller
        @param state<list>: The state of the vehicle
        @param input<list>: The input of the vehicle (unused)
        @param index<int>: Elapsed-tick counter from the node (unused --
        see class docstring).
        @param delta_t<float>: The time step
        @return<tuple>: The status and control
        """
        last_index = len(self.trajectory.u[0, :]) - 1

        nearest_index = self._search_nearest_index(state)

        if nearest_index >= last_index:
            return False, [0, 0]

        self._update_slip_estimate(state, nearest_index, delta_t)

        return super(MPCRLS, self).execute(state, input, index, delta_t)

    # ==================================================================================================
    # PRIVATE METHODS
    # ==================================================================================================
    def _update_slip_estimate(self, state, nearest_index, delta_t):
        """! Update the MPC's slip factor from the online RLS estimate.
        @param state<list>: The state of the vehicle
        @param nearest_index<int>: The trajectory index nearest to the
        robot's current position (see MPC._search_nearest_index)
        @param delta_t<float>: The time step
        """
        unwrapped_yaw = np.unwrap([state[2]])[0]

        if self._yaw_previous is not None:
            w_ref = self.trajectory.u[1, nearest_index]

            self._rls.predict_sim_with_forgetting_factor(
                yaw=unwrapped_yaw,
                yaw_previous=self._yaw_previous,
                ground_angular_velocity_z=w_ref,
                delta_t=delta_t,
                lam=self._lam)

        self._yaw_previous = unwrapped_yaw

        slip = float(np.clip(self._rls.estimates[-1][0, 0], -self._slip_clip, self._slip_clip))

        if self._step < self._warmup_steps:
            slip = 0.0

        self._step += 1

        self._mpc.s = slip
