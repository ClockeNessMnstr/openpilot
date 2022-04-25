#!/usr/bin/env python3
from cereal import car
from selfdrive.car.subaru.values import CAR, PREGLOBAL_CARS
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase

class CarInterface(CarInterfaceBase):

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)

    ret.carName = "subaru"
    ret.radarOffCan = True

    if candidate in PREGLOBAL_CARS:
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.subaruLegacy)]
    elif candidate == CAR.OUTBACK:
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.subaruGen2)]
    elif candidate == CAR.CROSSTREK_2020H:
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.subaruHybrid)]
    else:
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.subaru)]

    if candidate in PREGLOBAL_CARS:
      ret.enableBsm = 0x25c in fingerprint[0]
    else:
      ret.enableBsm = 0x228 in fingerprint[0]

    #ret.dashcamOnly = candidate in PREGLOBAL_CARS

    ret.steerRateCost = 2.0
    ret.steerLimitTimer = 0.4
    stiffness_front = 1.0
    stiffness_rear = 1.0

    if candidate == CAR.ASCENT:
      ret.mass = 2031. + STD_CARGO_KG
      ret.wheelbase = 2.89
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 13.5
      ret.steerActuatorDelay = 0.3   # end-to-end angle controller
      ret.lateralTuning.pid.kf = 0.00003
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.0025, 0.1], [0.00025, 0.01]]

    if candidate == CAR.IMPREZA or candidate == CAR.CROSSTREK_2020H:
      ret.safetyConfigs[0].safetyParam = 2 # increase limit on some crosstrek / impreza
    if candidate == CAR.IMPREZA or candidate == CAR.CROSSTREK_2020H or candidate == CAR.IMPREZA_2020:
      ret.mass = 1560. + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 13.5
      stiffness_front = 0.500
      stiffness_rear = 0.600
      ret.steerActuatorDelay = 0.18  # torque apply delay (.2 + .2 offset)
      ret.lateralTuning.init('discrete')
      p = 0.160
      i = 0.240
      d = 0.008
      ret.lateralTuning.discrete.gains = [i, p, d]

    if candidate == CAR.FORESTER:
      ret.mass = 1620. + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 13.5
      stiffness_front = 0.650
      stiffness_rear = 0.650
      ret.steerActuatorDelay = 0.1
      ret.lateralTuning.init('discrete')
      p = 0.080
      i = 0.160
      d = 0.006
      ret.lateralTuning.discrete.gains = [i, p, d]

    if candidate == CAR.OUTBACK:
      ret.mass = 1568. + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 17           # learned, 14 stock
      ret.steerActuatorDelay = 0.1
      ret.lateralTuning.pid.kf = 0.000038
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 14., 23.], [0., 14., 23.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.01, 0.065, 0.2], [0.001, 0.015, 0.025]]

    if candidate in (CAR.FORESTER_PREGLOBAL, CAR.OUTBACK_PREGLOBAL_2018):
      ret.safetyConfigs[0].safetyParam = 1  # Outback 2018-2019 and Forester have reversed driver torque signal
      ret.mass = 1568 + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 20           # learned, 14 stock
      ret.steerActuatorDelay = 0.1
      ret.lateralTuning.pid.kf = 0.000039
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 10., 20.], [0., 10., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.01, 0.05, 0.2], [0.003, 0.018, 0.025]]

    if candidate == CAR.WRX_PREGLOBAL:
      ret.safetyConfigs[0].safetyParam = 1  # WRX has reversed driver torque signal
      ret.mass = 1568 + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 12.5   # 14.5 stock
      ret.steerActuatorDelay = 0.15
      ret.lateralTuning.pid.kf = 0.00005
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.1, 0.2], [0.01, 0.02]]

    if candidate == CAR.LEGACY_PREGLOBAL:
      ret.mass = 1568 + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 12.5   # 14.5 stock
      ret.steerActuatorDelay = 0.15
      ret.lateralTuning.pid.kf = 0.00005
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.1, 0.2], [0.01, 0.02]]

    if candidate == CAR.LEGACY_PREGLOBAL_2018:
      ret.safetyConfigs[0].safetyParam = 1  # Legacy 2018-2019 has reversed driver torque signal
      ret.mass = 1568 + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 12.5   # 14.5 stock
      ret.steerActuatorDelay = 0.15
      ret.lateralTuning.pid.kf = 0.00005
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.1, 0.2], [0.01, 0.02]]

    if candidate == CAR.LEVORG_PREGLOBAL:
       ret.safetyConfigs[0].safetyParam = 1  # Levorg has reversed driver torque signal
       ret.mass = 1568 + STD_CARGO_KG
       ret.wheelbase = 2.67
       ret.centerToFront = ret.wheelbase * 0.5
       ret.steerRatio = 12.5   # 14.5 stock
       ret.steerActuatorDelay = 0.15
       ret.lateralTuning.pid.kf = 0.00005
       ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
       ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.1, 0.2], [0.01, 0.02]]

    if candidate == CAR.OUTBACK_PREGLOBAL:
      ret.mass = 1568 + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 20           # learned, 14 stock
      ret.steerActuatorDelay = 0.1
      ret.lateralTuning.pid.kf = 0.000039
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 10., 20.], [0., 10., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.01, 0.05, 0.2], [0.003, 0.018, 0.025]]

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)
    ret.tireStiffnessFront *= stiffness_front
    ret.tireStiffnessRear *= stiffness_rear

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)
    if self.cp_body:
      self.cp_body.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_cam, self.cp_body)

    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid and (self.cp_body is None or self.cp_body.can_valid)
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    ret.events = self.create_common_events(ret).to_msg()

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    hud_control = c.hudControl
    ret = self.CC.update(c, c.enabled, self.CS, self.frame, c.actuators,
                         c.cruiseControl.cancel, hud_control.visualAlert,
                         hud_control.leftLaneVisible, hud_control.rightLaneVisible, hud_control.leftLaneDepart, hud_control.rightLaneDepart)
    self.frame += 1
    return ret
