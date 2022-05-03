import math
from selfdrive.controls.lib.discrete import DiscreteController
from common.numpy_fast import clip
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from common.realtime import DT_CTRL
from cereal import log

LOW_SPEED_FACTOR = 200

class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.pid = DiscreteController([CP.lateralTuning.torque.ki, CP.lateralTuning.torque.kp, CP.lateralTuning.torque.friction],  rate=(1 / DT_CTRL))
    self.steer_max = 1.0
    self.use_steering_angle = CP.lateralTuning.torque.useSteeringAngle

  def reset(self):
    super().reset()
    self.pid.reset()

  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate, llk):
    pid_log = log.ControlsState.LateralTorqueState.new_message()

    if CS.vEgo < MIN_STEER_SPEED or not active:
      output_torque = 0.0
      pid_log.active = False
      self.reset()
    else:
      if self.use_steering_angle:
        actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
      else:
        actual_curvature = llk.angularVelocityCalibrated.value[2] / CS.vEgo
        
      error = -(desired_curvature - actual_curvature) *(CS.vEgo**2 + LOW_SPEED_FACTOR)
      output_torque = self.pid.update(error, last_actuators.steer)
      
      output_torque = clip(output_torque, -self.steer_max, self.steer_max)
      
      pid_log.error = error
      pid_log.active = True
      pid_log.i = float(self.pid.gains[0]*self.pid.d[0][1])
      pid_log.p = float(self.pid.gains[1]*self.pid.d[1][1])
      pid_log.d = float(self.pid.gains[2]*self.pid.d[2][1])
      pid_log.f = 0
      pid_log.output = -output_torque
      pid_log.saturated = self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS)

    return output_torque, 0.0, pid_log
