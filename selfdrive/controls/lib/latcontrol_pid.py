import math
import numpy as np

from cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.discrete import DiscreteController
from openpilot.common.realtime import DT_CTRL

class LatControlPID(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)

    i = 0.25
    p = 4.0
    d = 7.5
    gains = [g / 2.5 for g in [i, p, d]]

    N = 0.5
    Z = [[[1, 1], [1, -1]], [[1], [1]], [[1, -1], [1-1j, 1+1j    ]]]
    T = [[[1, 0], [    2]], [[1], [1]], [[2    ], [1   , (1/N)*2j]]]
    self.pid = DiscreteController(gains, Z, T, rate=(1 / DT_CTRL))

    N = 4.0
    Z = [[[1, 1], [1-1j, 1+1j    ]]]
    T = [[[1, 0], [1   , (1/N)*2j]]]
    self.desired = DiscreteController([1], Z, T, rate=(1 / DT_CTRL))

  def reset(self):
    super().reset()
    self.desired.reset()
    self.pid.reset()

  def update(self, active, CS, VM, params, steer_limited_by_controls, actuators, desired_curvature, calibrated_pose, curvature_limited):
    pid_log = log.ControlsState.LateralPIDState.new_message()

    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg

    actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    pid_log.steeringAngleDeg = float(actual_curvature)
    pid_log.steeringRateDeg = float(actual_curvature)

    if not active:
      output_steer = 0.0
      self.reset()
    else:
      desired = self.desired.update(desired_curvature, self.desired.u[0])
      error = -(desired - actual_curvature) * CS.vEgo ** 2
      output_steer = self.pid.update(error, actuators.torque)
      output_steer = np.clip(output_steer, -self.steer_max, self.steer_max)

      pid_log.active = True
      pid_log.i = float(self.pid.gains[0]*self.pid.d[0][1])
      pid_log.p = float(self.pid.gains[1]*self.pid.d[1][1])
      pid_log.f = float(self.pid.gains[2]*self.pid.d[2][1]) # d-term

      pid_log.angleError = float(self.pid.e[1])

      pid_log.output = float(output_steer)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_steer) < 1e-3, CS, steer_limited_by_controls, curvature_limited))

    return output_steer, angle_steers_des, pid_log
