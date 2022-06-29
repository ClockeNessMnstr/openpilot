from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from cereal import log

class LatControlTorque(LatControl):
  def update(self, active, CS, VM, params, last_actuators, desired_curvature, desired_curvature_rate, llk):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    if CS.vEgo < MIN_STEER_SPEED or not active:
      output_torque = 0.0
      self.reset()
    else:
      output_torque = -desired_curvature
      pid_log.saturated = self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS)
    return output_torque, 0.0, pid_log
