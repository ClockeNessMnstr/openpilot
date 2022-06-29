#!/usr/bin/env python3
import math
import sys
from typing import Any, Dict

import numpy as np

from selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from selfdrive.locationd.models.constants import ObservationKind
from selfdrive.swaglog import cloudlog

from rednose.helpers.kalmanfilter import KalmanFilter

if __name__ == '__main__':  # Generating sympy
  import sympy as sp
  from rednose.helpers.ekf_sym import gen_code
else:
  from rednose.helpers.ekf_sym_pyx import EKF_sym  # pylint: disable=no-name-in-module, import-error


i = 0

def _slice(n):
  global i
  s = slice(i, i + n)
  i += n

  return s


class States():
  # Vehicle model params
  STIFFNESS_FRONT = _slice(1)  # [-]
  STIFFNESS_REAR = _slice(1)  # [-]
  ANGLE_OFFSET = _slice(1)  # [rad]

  VELOCITY = _slice(2)  # (x, y) [m/s]
  
  YAW_RATE = _slice(1)  # [rad/s]
  STEER_ANGLE = _slice(1)  # [rad]
  STEER_RATE = _slice(1)  # [rad/s]
  ROAD_ROLL = _slice(1)  # [rad]
  ACTUATOR = _slice(1)  # [-]
  K_ACTUATOR = _slice(1)  # [-]
  K_REST = _slice(1)  # [-]
  K_DAMP = _slice(1)  # [-]


class CarKalman(KalmanFilter):
  name = 'car'

  initial_x = np.array([
    1.0,
    1.0,
    0.0,

    10.0,
    0.0,
    
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.01,
    0.02,
    0.1,
  ])

  # process noise
  Q = np.diag([
    (0.5 / 10000)**2,        # Cf is modeled
    (0.5 / 10000)**2,        # Cr is modeled
    math.radians(0.125)**2, # angle offset, wind is most likely going here primarily

    5**2,                   # not confident in this prediction (missing accel)
    .01**2,                 # v is modeled

    math.radians(0.5)**2,   # r is modeled
    math.radians(10)**2,    # sa is modeled (state is tire angle) but input is not
    math.radians(.1)**2,    # sa_dot is modeled (state is tire rate) but input is not
    math.radians(10)**2,    # not confident in this prediction (roll)
    1.0**2,                 # not confident in this prediction (missing actuator_rate)
    (1.0 / 100000)**2,        # actuator effectiveness
    (1.0 / 100000)**2,        # centering effectiveness
    (1.0 / 100000)**2,        # damping (DIVERGING)
  ])
  
  # Variances? (stable values in logs indicate convergence with given variance)
  P_initial = np.diag([
    (0.005)**2,
    (0.005)**2,
    (0.008)**2,

    (0.450)**2, 
    (0.030)**2,

    (0.010)**2,
    (0.016)**2,
    (0.200)**2,
    (0.008)**2,
    (0.005)**2,
    (0.001)**2,
    (0.001)**2,
    (0.001)**2,
  ])
  
  # noise/variance with which to make predictions for these measurements if not given in the observation
  obs_noise: Dict[int, Any] = {
    ObservationKind.ROAD_FRAME_X_SPEED: np.atleast_2d(0.05**2),
    ObservationKind.STEER_ACTUATOR: np.atleast_2d(0.01**2),
  }

  global_vars = [
    'mass',
    'rotational_inertia',
    'center_to_front',
    'center_to_rear',
    'stiffness_front',
    'stiffness_rear',
    'steer_ratio',
  ]

  @staticmethod
  def generate_code(generated_dir):
    dim_state = CarKalman.initial_x.shape[0]
    name = CarKalman.name

    # vehicle models comes from The Science of Vehicle Dynamics: Handling, Braking, and Ride of Road and Race Cars
    # Model used is in 6.15 with formula from 6.198

    # globals
    global_vars = [sp.Symbol(name) for name in CarKalman.global_vars]
    m, j, aF, aR, cF_orig, cR_orig, sr = global_vars

    # make functions and jacobians with sympy
    # state variables
    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)

    # Vehicle model constants
    sF = state[States.STIFFNESS_FRONT, :][0, 0]
    sR = state[States.STIFFNESS_REAR, :][0, 0]

    cF, cR = sF * cF_orig, sR * cR_orig
    roll = state[States.ROAD_ROLL, :][0, 0]
    sa = state[States.STEER_ANGLE, :][0, 0]
    sa_obs = sr * state[States.STEER_ANGLE, :][0, 0] * sr + state[States.ANGLE_OFFSET, :][0, 0]
    sa_rate = state[States.STEER_RATE, :][0, 0]
    actuator = state[States.ACTUATOR, :][0, 0]
    k_actuator = state[States.K_ACTUATOR, :][0, 0]
    k_rest = state[States.K_REST, :][0, 0]
    #k_damp = state[States.K_DAMP, :][0, 0]

    u, v = state[States.VELOCITY, :]
    r = state[States.YAW_RATE, :][0, 0]

    A = sp.Matrix(np.zeros((4, 4)))
    A[0, 0] = -(cF + cR) / (m * u)
    A[0, 1] = -(cF * aF - cR * aR) / (m * u) - u
    A[0, 2] =  (cF) / m
    A[1, 0] = -(cF * aF - cR * aR) / (j * u)
    A[1, 1] = -(cF * aF**2 + cR * aR**2) / (j * u)
    A[1, 2] =  (cF * aF) / j
    A[2, 3] =  1
    A[3, 1] = -k_rest*u
    #A[3, 3] = -k_damp

    B = sp.Matrix(np.zeros((4, 1)))
    B[3, 0] = k_actuator

    C = sp.Matrix(np.zeros((4, 1)))
    C[0, 0] = ACCELERATION_DUE_TO_GRAVITY

    x = sp.Matrix([v, r, sa, sa_rate])  # lateral velocity, yaw rate, steer angle, steer rate
    x_dot = A * x + B * actuator + C * roll

    dt = sp.Symbol('dt')
    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[States.VELOCITY.start + 1, 0] = x_dot[0]
    state_dot[States.YAW_RATE.start, 0] = x_dot[1]
    state_dot[States.STEER_ANGLE.start, 0] = x_dot[2]
    state_dot[States.STEER_RATE.start, 0] = x_dot[3]

    # Basic descretization, 1st order integrator
    # Can be pretty bad if dt is big
    f_sym = state + dt * state_dot

    #
    # Observation functions
    #
    obs_eqs = [
      [sp.Matrix([r]), ObservationKind.ROAD_FRAME_YAW_RATE, None],
      [sp.Matrix([u, v]), ObservationKind.ROAD_FRAME_XY_SPEED, None],
      [sp.Matrix([u]), ObservationKind.ROAD_FRAME_X_SPEED, None],
      [sp.Matrix([sa_obs]), ObservationKind.STEER_ANGLE, None],
      [sp.Matrix([roll]), ObservationKind.ROAD_ROLL, None],
      [sp.Matrix([actuator]), ObservationKind.STEER_ACTUATOR, None],
    ]

    gen_code(generated_dir, name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state, global_vars=global_vars)

  def __init__(self, generated_dir, stiffness_front=1, stiffness_rear=1, angle_offset=0, k_actuator=1, k_rest=0.0, k_damp=0.0, P_initial=None):  # pylint: disable=super-init-not-called
    dim_state = self.initial_x.shape[0]
    dim_state_err = self.P_initial.shape[0]
    x_init = self.initial_x
    x_init[States.STIFFNESS_FRONT] = stiffness_front
    x_init[States.STIFFNESS_REAR] = stiffness_rear
    x_init[States.ANGLE_OFFSET] = angle_offset
    x_init[States.K_ACTUATOR] = k_actuator
    x_init[States.K_REST] = k_rest
    x_init[States.K_DAMP] = k_damp

    if P_initial is not None:
      self.P_initial = P_initial
    # init filter
    self.filter = EKF_sym(generated_dir, self.name, self.Q, self.initial_x, self.P_initial, dim_state, dim_state_err, global_vars=self.global_vars, logger=cloudlog)

if __name__ == "__main__":
  generated_dir = sys.argv[2]
  CarKalman.generate_code(generated_dir)
