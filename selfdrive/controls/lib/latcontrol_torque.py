import math
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from selfdrive.controls.lib.discrete import DiscreteController
from common.realtime import DT_CTRL
from common.numpy_fast import clip
from cereal import log

class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    
    i = 1.50 / CP.lateralTuning.torque.maxLatAccel**2
    p = 6.75 / CP.lateralTuning.torque.maxLatAccel**2
    d = 5.25 / CP.lateralTuning.torque.maxLatAccel**2
    gains = [i, p, d]
    
    Z = [[[1, 2, 1], [1, -2, 1]], [[1, 1], [1, -1]], [[1], [1]]]
    T = [[[1, 0, 0], [       4]], [[1, 0], [    2]], [[1], [1]]]
    self.pid = DiscreteController(gains, Z, T, rate=(1 / DT_CTRL))
    
    self.steer_max = 1.0

  def reset(self):
    super().reset()
    self.pid.reset()

  def update(self, active, CS, VM, params, last_actuators, desired_curvature, desired_curvature_rate, llk):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    if CS.vEgo < MIN_STEER_SPEED or not active:
      output_torque = 0.0
      self.reset()
    else:
      error = -(desired_curvature_rate) * (CS.vEgo**2)
      output_torque = self.pid.update(error, last_actuators.steer)
      output_torque = clip(output_torque, -self.steer_max, self.steer_max)
      
      pid_log.active = True
      pid_log.error = error
      pid_log.i = float(self.pid.gains[0]*self.pid.d[0][1])
      pid_log.p = float(self.pid.gains[1]*self.pid.d[1][1])
      pid_log.d = float(self.pid.gains[2]*self.pid.d[2][1])
      pid_log.output = output_torque
      pid_log.saturated = self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS)
      pid_log.actualLateralAccel = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll) * (CS.vEgo**2)
      pid_log.desiredLateralAccel = desired_curvature * (CS.vEgo**2)

    return output_torque, 0.0, pid_log
