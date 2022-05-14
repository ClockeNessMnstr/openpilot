import numpy as np

from common.realtime import DT_CTRL
from selfdrive.car.body import bodycan
from opendbc.can.packer import CANPacker
from selfdrive.car.body.values import SPEED_FROM_RPM
from selfdrive.controls.lib.discrete import DiscreteController

MAX_TORQUE = 500
MAX_TORQUE_RATE = 50
MAX_ANGLE_ERROR = np.radians(7)
RATE_FACTOR = 2

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.frame = 0
    self.packer = CANPacker(dbc_name)

    # Speed, balance and turn PIDs
    Z = [[[1, 1], [2, -2]], [[1], [1]]] # Trapezoidal IP controller
    T = [[[1, 0], [1    ]], [[1], [1]]] # Trapezoidal IP controller
    
    balance_gains = [1300, 280] # I, P applied to rate I->P, P->D
    speed_gains = [0.23, 0.115] # I, P
    turn_gains = [11.5, 110] # I, P
    
    self.balance_pid = DiscreteController(balance_gains, Z, T, rate=1/DT_CTRL)
    self.speed_pid = DiscreteController(speed_gains, Z, T, rate=1/DT_CTRL)
    self.turn_pid = DiscreteController(turn_gains, Z, T, rate=1/DT_CTRL)

    self.torque_r = 0.
    self.torque_l = 0.
    self.angle_desired = 0.
    self.angle_measured = 0.
    self.torque_diff = 0.
    self.torque = 0
    self.angle_rate_measured = 0

  @staticmethod
  def deadband_filter(torque, deadband):
    if torque > 0:
      torque += deadband
    else:
      torque -= deadband
    return torque

  def update(self, CC, CS):

    torque_l = 0
    torque_r = 0

    llk_valid = len(CC.orientationNED) > 0 and len(CC.angularVelocity) > 0
    if CC.enabled and llk_valid:
      # Read these from the joystick
      # TODO: this isn't acceleration, okay?
      speed_desired = CC.actuators.accel / 5.
      # TODO: make this positive turning right
      speed_diff_desired = -CC.actuators.steer

      speed_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr) / 2.
      angle_measured = CC.orientationNED[1]
      angle_rate_measured = np.clip(CC.angularVelocity[1], -1., 1.)
      speed_diff_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl - CS.out.wheelSpeeds.fr)

      last_angle_rate_desired = self.balance_pid.recalculate(self.torque) + self.angle_rate_measured
      last_angle_desired = (last_angle_rate_desired + self.angle_measured) / RATE_FACTOR
      
      angle_desired = self.speed_pid.update(speed_desired - speed_measured, last_angle_desired)
      
      angle_rate_desired = angle_desired - angle_measured
      angle_rate_desired = np.clip(angle_desired - angle_measured, -MAX_ANGLE_ERROR, MAX_ANGLE_ERROR)
      angle_rate_desired *= RATE_FACTOR
      
      torque = self.balance_pid.update(angle_rate_desired - angle_rate_measured, self.torque)

      turn_error = speed_diff_measured - speed_diff_desired
      torque_diff = self.turn_pid.update(turn_error, self.torque_diff)

      # Combine 2 PIDs outputs
      torque_r = torque + torque_diff
      torque_l = torque - torque_diff

      # Torque rate limits
      torque_r = np.clip(self.deadband_filter(torque_r, 10),
                                       self.torque_r - MAX_TORQUE_RATE,
                                       self.torque_r + MAX_TORQUE_RATE)
      torque_l = np.clip(self.deadband_filter(torque_l, 10),
                                       self.torque_l - MAX_TORQUE_RATE,
                                       self.torque_l + MAX_TORQUE_RATE)
      torque_r = int(np.clip(torque_r, -MAX_TORQUE, MAX_TORQUE))
      torque_l = int(np.clip(torque_l, -MAX_TORQUE, MAX_TORQUE))
    
    self.angle_desired = angle_desired
    self.angle_measured = angle_measured
    self.angle_rate_measured = angle_rate_measured
    self.torque_diff = torque_r - torque_l
    self.torque = (torque_r + torque_r) / 2
    self.torque_r = torque_r
    self.torque_l = torque_l

    can_sends = []
    can_sends.append(bodycan.create_control(self.packer, self.torque_l, self.torque_r, self.frame // 2))

    new_actuators = CC.actuators.copy()
    new_actuators.accel = self.torque_l
    new_actuators.steer = self.torque_r

    self.frame += 1
    return new_actuators, can_sends
