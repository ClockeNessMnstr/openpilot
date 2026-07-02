import math
import numpy as np

from cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.discrete import DiscreteController
from openpilot.common.realtime import DT_CTRL

from openpilot.common.opedit_mini import read_param, write_param

class LatControlPID(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)

    P0 = 0
    R_sys = 1 / 1000000
    R_obs = 1 / 100

    self.pid = DiscreteController()
    num = np.array([[ 1.5, -2.94696296, 1.447481479]])
    den = np.array([ 1.0, -1.97530864, 0.97530864])
    self.pid.set_dlti((num, den, DT_CTRL))

    delay = 0.1

    delta = 1.00 # 0, 1, 2
    sigma = 0.1

    self.pid.set_ref(sigma, delta, P0=P0, R_sys=R_sys, R_obs=R_obs, delay=int(delay / DT_CTRL))

    self.desired = DiscreteController()

    N = 4.0
    num = np.array([[ N*DT_CTRL, N*DT_CTRL]])
    den = np.array([ N*DT_CTRL + 2, N*DT_CTRL - 2])
    self.desired.set_dlti((num, den, DT_CTRL))

    self.read_params()
    self.running = False

  def reset(self):
    super().reset()
    self.pid.reset()
    self.desired.reset()

  def save_params(self):
    write_param('ke', np.array(self.pid.ke).tolist())

  def read_params(self):
    ke = read_param('ke')
    if ke[1]:
      self.pid.ke = np.atleast_2d(ke[0])
      self.pid.set_pfd()

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
      if self.running:
        self.save_params()
        self.running = False
    else:
      if not self.running:
        self.read_params()
        self.running = True

      desired = self.desired.update(desired_curvature)
      if CS.steeringPressed:
        self.pid.kf_hold()
      output_steer = self.pid.update(-desired * CS.vEgo ** 2, -actual_curvature * CS.vEgo ** 2, actuators.torque)
      output_steer = np.clip(output_steer, -self.steer_max, self.steer_max)

      pid_log.active = True
      pid_log.i = float(self.pid.theta[0][0])
      pid_log.p = float(self.pid.theta[1][0])
      pid_log.f = float(self.pid.theta[2][0]) # d-term

      pid_log.angleError = float(self.pid.e[0][1])

      pid_log.output = float(output_steer)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_steer) < 1e-3, CS, steer_limited_by_controls, curvature_limited))

    return output_steer, angle_steers_des, pid_log
