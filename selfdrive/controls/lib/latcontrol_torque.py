import math
from selfdrive.controls.lib.latcontrol import LatControl, MIN_LATERAL_CONTROL_SPEED
from selfdrive.controls.lib.discrete import DiscreteController
from common.numpy_fast import clip
from common.realtime import DT_CTRL
from common.opedit_mini import read_param, write_param
from cereal import log

class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)

    i = 1.0
    p = 4.0
    d = 0.2
    gains = [g / CP.lateralTuning.torque.latAccelFactor for g in [i, p, d]]

    N = 10 # Filter coefficient. corner frequency in rad/s. 20 = ~3.18hz
    Z = [[[1, 1], [1, -1]], [[1], [1]], [[1, -1], [1-1j, 1+1j    ]]]
    T = [[[1, 0], [    2]], [[1], [1]], [[2    ], [1   , (1/N)*2j]]]
    self.pid = DiscreteController(gains, Z, T, rate=(1 / DT_CTRL))

    self.torque_params = CP.lateralTuning.torque

    write_param('gains', gains)

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def reset(self):
    super().reset()
    self.pid.reset()

    gains = read_param('gains')
    if gains[1]:
      gains = gains[0]
      self.pid.update_gains(gains)

  def update(self, active, CS, VM, params, last_actuators, steer_limited, desired_curvature, desired_curvature_rate, llk):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    if CS.vEgo < MIN_LATERAL_CONTROL_SPEED or not active:
      output_torque = 0.0
      self.reset()
    else:
      actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)

      error = -(desired_curvature - actual_curvature) * CS.vEgo ** 2
      output_torque = self.pid.update(error, last_actuators.steer)
      output_torque = clip(output_torque, -self.steer_max, self.steer_max)

      pid_log.active = True
      pid_log.error = error
      pid_log.i = float(self.pid.gains[0]*self.pid.d[0][1])
      pid_log.p = float(self.pid.gains[1]*self.pid.d[1][1])
      pid_log.d = float(self.pid.gains[2]*self.pid.d[2][1])
      pid_log.output = output_torque
      pid_log.saturated = self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited)
      pid_log.actualLateralAccel = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll) * (CS.vEgo**2)
      pid_log.desiredLateralAccel = desired_curvature * (CS.vEgo**2)

    return output_torque, 0.0, pid_log

