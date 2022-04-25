#!/usr/bin/env python3
import sys
import math

import numpy as np

from selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from selfdrive.locationd.models.constants import ObservationKind
from rednose.helpers.kalmanfilter import KalmanFilter
from selfdrive.swaglog import cloudlog
from typing import Any, Dict

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
  # Vehicle model params (can drift / change but do not "vary" in the model)
  STIFFNESS_FRONT = _slice(1)  # [-]
  STIFFNESS_REAR = _slice(1)  # [-]
  ANGLE_OFFSET = _slice(1)  # [rad]
  FRONT_RATIO = _slice(1) # [-]
  MASS_RATIO = _slice(1) # [-]
  MOMENT_RATIO = _slice(1) # [-]

  # Vehicle states + unmodeled inputs
  VELOCITY = _slice(2)  # (x, y) [m/s]
  YAW_RATE = _slice(1)  # [rad/s]
  STEER_ANGLE = _slice(1)  # [rad]
  ROAD_ROLL = _slice(1)  # [rad]

class CarKalman(KalmanFilter):
  name = 'car'
  initial_x = np.array([
    1.0,
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,

    10.0, 0.0,
    0.0,
    0.0,
    0.0
  ])

  # process noise (lower numbers mean we can make good predictions)
  Q = np.diag([
    (.005 / 100)**2,
    (.005 / 100)**2,
    math.radians(0.02)**2, # wind is likely in here / not modeled
    (.001 / 100)**2,
    (.001 / 100)**2,
    (.001 / 100)**2,

    6**2, .03**2,            # u is not modeled (acceleration is unknown), v is modeled
    math.radians(.1)**2,   # r is modeled
    math.radians(12)**2,    # absolutely not confident in this prediction (torque is not modeled)
    math.radians(3)**2,     # absolutely not confident in this prediction (road roll is not modeled)
  ])
  
  # Variances? (stable values in logs indicate convergence with given variance)
  P_initial = np.diag([
    (2 / 100)**2,
    (2 / 100)**2,
    math.radians(0.08)**2,
    (.52 / 100)**2,
    (.84 / 100)**2,
    (.87 / 100)**2,

    .6**2, .03**2,
    math.radians(.1)**2,
    math.radians(1.2)**2,
    math.radians(.3)**2,
  ])
  
  # noise/variance with which to make predictions for these measurements if not given in the observation
  obs_noise: Dict[int, Any] = {
    ObservationKind.STEER_ANGLE: np.atleast_2d(math.radians(1.0)**2),
    ObservationKind.ROAD_FRAME_X_SPEED: np.atleast_2d(0.5**2),
  }

  global_vars = [
    'mass',
    'rotational_inertia',
    'center_to_front',
    'wheelbase',
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
    m_orig, j_orig, aF_orig, l, cF_orig, cR_orig, sr = global_vars

    # make functions and jacobians with sympy
    # state variables
    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)

    # Vehicle model constants
    sF = state[States.STIFFNESS_FRONT, :][0, 0]
    sR = state[States.STIFFNESS_REAR, :][0, 0]
    fR = state[States.FRONT_RATIO, :][0, 0]
    mR = state[States.MASS_RATIO, :][0, 0]
    jR = state[States.MOMENT_RATIO, :][0, 0]

    cF, cR = sF * cF_orig, sR * cR_orig
    aF, aR = aF_orig*fR, l - aF_orig*fR
    m, j = m_orig*mR, j_orig*jR*mR
    angle_offset = state[States.ANGLE_OFFSET, :][0, 0]
    theta = state[States.ROAD_ROLL, :][0, 0]
    sa = state[States.STEER_ANGLE, :][0, 0]
    u, v = state[States.VELOCITY, :]
    r = state[States.YAW_RATE, :][0, 0]

    A = sp.Matrix(np.zeros((2, 2)))
    A[0, 0] = -(cF + cR) / (m * u)
    A[0, 1] = -(cF * aF - cR * aR) / (m * u) - u
    A[1, 0] = -(cF * aF - cR * aR) / (j * u)
    A[1, 1] = -(cF * aF**2 + cR * aR**2) / (j * u)

    B = sp.Matrix(np.zeros((2, 1)))
    B[0, 0] = cF / m / sr
    B[1, 0] = (cF * aF) / j / sr

    C = sp.Matrix(np.zeros((2, 1)))
    C[0, 0] = ACCELERATION_DUE_TO_GRAVITY
    C[1, 0] = 0

    x = sp.Matrix([v, r])  # lateral velocity, yaw rate
    x_dot = A * x - B * (sa - angle_offset) + C * theta

    dt = sp.Symbol('dt')
    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[States.VELOCITY.start + 1, 0] = x_dot[0]
    state_dot[States.YAW_RATE.start, 0] = x_dot[1]

    # Basic descretization, 1st order integrator
    # Can be pretty bad if dt is big
    f_sym = state + dt * state_dot

    # Observation functions
    obs_eqs = [
      [sp.Matrix([r]), ObservationKind.ROAD_FRAME_YAW_RATE, None],
      [sp.Matrix([u, v]), ObservationKind.ROAD_FRAME_XY_SPEED, None],
      [sp.Matrix([u]), ObservationKind.ROAD_FRAME_X_SPEED, None],
      [sp.Matrix([sa]), ObservationKind.STEER_ANGLE, None],
      [sp.Matrix([theta]), ObservationKind.ROAD_ROLL, None],
    ]

    gen_code(generated_dir, name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state, global_vars=global_vars)

  def __init__(self, generated_dir, stiffness_front=1, stiffness_rear=1, front_ratio=1, mass_ratio=1, moment_ratio=1, angle_offset=0, P_initial=None):  # pylint: disable=super-init-not-called
    dim_state = self.initial_x.shape[0]
    dim_state_err = self.P_initial.shape[0]
    x_init = self.initial_x
    x_init[States.STIFFNESS_FRONT] = stiffness_front
    x_init[States.STIFFNESS_REAR] = stiffness_rear
    x_init[States.FRONT_RATIO] = front_ratio
    x_init[States.MASS_RATIO] = mass_ratio
    x_init[States.MOMENT_RATIO] = moment_ratio
    x_init[States.ANGLE_OFFSET] = angle_offset

    if P_initial is not None:
      self.P_initial = P_initial
    # init filter
    self.filter = EKF_sym(generated_dir, self.name, self.Q, self.initial_x, self.P_initial, dim_state, dim_state_err, global_vars=self.global_vars, logger=cloudlog)


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  CarKalman.generate_code(generated_dir)
