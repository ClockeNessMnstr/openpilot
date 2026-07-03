import math
import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.discrete import DiscreteController

from openpilot.common.opedit_mini import read_param, write_param

class LatControlPID(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)

    i = 0.25
    p = 4.0
    d = 7.5
    self.gains = [g / 2.5 for g in [i, p, d]]
    self.Nf = [4.0, 0.5]

    self.update_controllers()
    self.save_params()

  def reset(self):
    super().reset()
    self.pid.reset()
    self.desired.reset()
    self.read_params()

  def save_params(self):
    write_param('gains', self.gains)
    write_param('filters', self.Nf)

  def read_params(self):
    gains = read_param('gains')
    if gains[1]:
      self.gains = gains[0]
      self.update_gains()

    Nf = read_param('filters')
    if Nf[1]:
      self.Nf = Nf[0]
      self.update_controllers()

  def update_controllers(self):
    N = self.Nf[0]
    Z = [[[1, 1], [1-1j, 1+1j    ]]]
    T = [[[1, 0], [1   , (1/N)*2j]]]
    self.desired = DiscreteController([1], Z, T, rate=(1 / self.dt))

    N = self.Nf[1]
    Z = [[[1, 1], [1, -1]], [[1], [1]], [[1, -1], [1-1j, 1+1j    ]]]
    T = [[[1, 0], [    2]], [[1], [1]], [[2    ], [1   , (1/N)*2j]]]
    self.pid = DiscreteController(self.gains, Z, T, rate=(1 / self.dt))

  def update_gains(self):
      self.pid.update_gains(self.gains)

  def update(self, active, CS, VM, params, steer_limited_by_safety, actuators, desired_curvature, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralPIDState.new_message()

    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg

    actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    pid_log.steeringAngleDeg = float(actual_curvature)
    pid_log.steeringRateDeg = float(actual_curvature)

    if not active:
      output_torque = 0.0
      self.reset()
    else:
      desired = self.desired.update(desired_curvature, self.desired.u[0])
      error = -(desired - actual_curvature) * CS.vEgo ** 2
      output_torque = self.pid.update(error, actuators.torque)
      output_torque = np.clip(output_torque, -self.steer_max, self.steer_max)

      pid_log.active = True
      pid_log.i = float(self.pid.gains[0]*self.pid.d[0][1])
      pid_log.p = float(self.pid.gains[1]*self.pid.d[1][1])
      pid_log.f = float(self.pid.gains[2]*self.pid.d[2][1]) # d-term

      pid_log.angleError = float(self.pid.e[1])

      pid_log.output = float(output_torque)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return output_torque, angle_steers_des, pid_log
