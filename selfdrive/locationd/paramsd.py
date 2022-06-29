#!/usr/bin/env python3
import math
import json
import numpy as np

import cereal.messaging as messaging
from cereal import car
from common.params import Params, put_nonblocking
from common.realtime import config_realtime_process, DT_MDL
from common.numpy_fast import clip
from selfdrive.controls.lib.latcontrol import MIN_STEER_SPEED
from selfdrive.locationd.models.car_kf import CarKalman, ObservationKind, States
from selfdrive.locationd.models.constants import GENERATED_DIR
from selfdrive.swaglog import cloudlog


MAX_ANGLE_OFFSET_DELTA = 20 * DT_MDL  # Max 20 deg/s
ROLL_MAX_DELTA = np.radians(20.0) * DT_MDL  # 20deg in 1 second is well within curvature limits
SMALL_ANGLE = 20

class ParamsLearner:
  def __init__(self, CP, stiffness_front, stiffness_rear, angle_offset, k_actuator, k_rest, k_damp, P_initial=None):
    self.kf = CarKalman(GENERATED_DIR, stiffness_front, stiffness_rear, angle_offset, k_actuator, k_rest, k_damp, P_initial)

    self.kf.filter.set_global("mass", CP.mass)
    self.kf.filter.set_global("rotational_inertia", CP.rotationalInertia)
    self.kf.filter.set_global("center_to_front", CP.centerToFront)
    self.kf.filter.set_global("center_to_rear", CP.wheelbase - CP.centerToFront)
    self.kf.filter.set_global("stiffness_front", CP.tireStiffnessFront)
    self.kf.filter.set_global("stiffness_rear", CP.tireStiffnessRear)
    self.kf.filter.set_global("steer_ratio", CP.steerRatio)

    self.small_angle = SMALL_ANGLE*CP.steerRatio
    #TODO add to CP and do actuator as well
    self.steer_std = 0.15 / CP.steerRatio # CP.steerResolution
    self.steering_pressed = False
    self.roll = 0.0

    self.valid = True

  def handle_log(self, t, which, msg):
    if which == 'liveLocationKalman':
      roll = msg.orientationNED.value[0]
      roll_std = np.radians(10) if np.isnan(msg.orientationNED.std[0]) else msg.orientationNED.std[0]
      roll_valid = msg.orientationNED.valid and (-math.radians(SMALL_ANGLE) < roll < math.radians(SMALL_ANGLE))
      self.roll = clip(roll, self.roll - ROLL_MAX_DELTA, self.roll + ROLL_MAX_DELTA)

      yaw_rate = msg.angularVelocityCalibrated.value[2]
      yaw_rate_std = msg.angularVelocityCalibrated.std[2]
      yaw_rate_valid = msg.angularVelocityCalibrated.valid
      yaw_rate_valid = yaw_rate_valid and 0 < yaw_rate_std < 10  # rad/s
      yaw_rate_valid = yaw_rate_valid and abs(yaw_rate) < 1  # rad/s

      if msg.posenetOK:
        if yaw_rate_valid:
          self.kf.predict_and_observe(t,
                                      ObservationKind.ROAD_FRAME_YAW_RATE,
                                      np.array([[yaw_rate]]),
                                      np.array([np.atleast_2d(yaw_rate_std**2)]))
        if roll_valid:
          self.kf.predict_and_observe(t,
                                      ObservationKind.ROAD_ROLL,
                                      np.array([[self.roll]]),
                                      np.array([np.atleast_2d(roll_std**2)]))
    elif which == 'carState':
      steering_angle = -msg.steeringAngleDeg
      self.steering_pressed = msg.steeringPressed
      speed = msg.vEgo
      
      if speed > MIN_STEER_SPEED:
        self.kf.predict_and_observe(t, ObservationKind.ROAD_FRAME_X_SPEED, np.array([[speed]]))
        
        steer_std = self.steer_std * abs(steering_angle) * max(1, 1 + abs(steering_angle) - self.small_angle)
        self.kf.predict_and_observe(t, 
                                    ObservationKind.STEER_ANGLE, 
                                    np.array([[math.radians(steering_angle)]]), 
                                    np.array([np.atleast_2d(math.radians(steer_std**2))]))
    elif which == 'carControl':
      actuator = -msg.actuatorsOutput.steer
      if not self.steering_pressed and 1 >= abs(actuator) > 0:
        self.kf.predict_and_observe(t, ObservationKind.STEER_ACTUATOR, np.array([[actuator]]))

def main(sm=None, pm=None):
  config_realtime_process([0, 1, 2, 3], 5)

  if sm is None:
    sm = messaging.SubMaster(['liveLocationKalman', 'carState', 'carControl'], poll=['liveLocationKalman'])
  if pm is None:
    pm = messaging.PubMaster(['liveParameters'])

  params_reader = Params()
  # wait for stats about the car to come in from controls
  cloudlog.info("paramsd is waiting for CarParams")
  CP = car.CarParams.from_bytes(params_reader.get("CarParams", block=True))
  cloudlog.info("paramsd got CarParams")

  params = params_reader.get("LiveParameters")

  # Check if car model matches
  if params is not None:
    params = json.loads(params)
    if params.get('carFingerprint', None) != CP.carFingerprint:
      cloudlog.info("Parameter learner found parameters for wrong car.")
      params = None
      
  P = None
  # Check if starting values are sane
  if params is not None:
    try:
      P = np.diag(params.get('P'))
      actuator_gain_sane = all((params.get('kActuator') > 0, params.get('kRest') > 0, params.get('kDamp') > 0,))
      stiffness_sane = (0.2 <= params.get('stiffnessFront') <= 5.0) and (0.2 <= params.get('stiffnessRear') <= 5.0)
      angle_offset_sane = abs(params.get('angleOffsetDeg')) < 10.0
      params_sane = all((angle_offset_sane, stiffness_sane, actuator_gain_sane,))
      if not params_sane:
        cloudlog.info(f"Invalid starting values found {params}")
        params = None
    except Exception as e:
      cloudlog.info(f"Error reading params {params}: {str(e)}")
      params = None

  # TODO: cache the params with the capnp struct
  if params is None:
    params = {
      'carFingerprint': CP.carFingerprint,
      'stiffnessFront': 1.0,
      'stiffnessRear': 1.0,
      'angleOffsetDeg': 0.0,
      'kActuator': 1.0,
      'kRest': 1.0,
      'kDamp': 1.0,
    }
    cloudlog.info("Parameter learner resetting to default values")

  learner = ParamsLearner(CP, params['stiffnessFront'],
                          params['stiffnessRear'],
                          math.radians(params['angleOffsetDeg']),
                          params['kActuator'] * CP.kActuator,
                          params['kRest'] * CP.kRest,
                          params['kDamp'] * CP.kDamp, 
                          P_initial=P,)
  angle_offset = params['angleOffsetDeg']

  while True:
    sm.update()
    if sm.all_checks():
      for which in sorted(sm.updated.keys(), key=lambda x: sm.logMonoTime[x]):
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          learner.handle_log(t, which, sm[which])

    if sm.updated['liveLocationKalman']:
      x = learner.kf.x
      P = np.sqrt(learner.kf.P.diagonal())
      if not all(map(math.isfinite, x)):
        cloudlog.error("NaN in liveParameters estimate. Resetting to default values")
        learner = ParamsLearner(CP, 1.0, 1.0, 0.0, CP.kActuator, CP.kRest, CP.kDamp)
        x = learner.kf.x

      angle_offset = clip(math.degrees(x[States.ANGLE_OFFSET]), angle_offset - MAX_ANGLE_OFFSET_DELTA, angle_offset + MAX_ANGLE_OFFSET_DELTA)

      msg = messaging.new_message('liveParameters')

      liveParameters = msg.liveParameters
      liveParameters.posenetValid = True
      liveParameters.sensorValid = True
      liveParameters.stiffnessFront = float(x[States.STIFFNESS_FRONT])
      liveParameters.stiffnessRear = float(x[States.STIFFNESS_REAR])
      liveParameters.roll = float(x[States.ROAD_ROLL])
      liveParameters.angleOffsetDeg = angle_offset
      liveParameters.yawRate = float(x[States.YAW_RATE])
      liveParameters.latVel = float(x[States.VELOCITY][1])
      liveParameters.tireAngle = float(x[States.STEER_ANGLE])
      liveParameters.tireAngleRate = float(x[States.STEER_RATE])
      liveParameters.cF = CP.tireStiffnessFront * liveParameters.stiffnessFront
      liveParameters.cR = CP.tireStiffnessRear * liveParameters.stiffnessRear
      liveParameters.aF = CP.centerToFront
      liveParameters.aR = (CP.wheelbase - CP.centerToFront)
      liveParameters.m = CP.mass
      liveParameters.j = CP.rotationalInertia
      liveParameters.actuator = float(x[States.ACTUATOR])
      liveParameters.kActuator = float(x[States.K_ACTUATOR])
      liveParameters.kRest = float(x[States.K_REST])
      liveParameters.kDamp = float(x[States.K_DAMP])
      
      liveParameters.valDump = [float(v) for v in x]
      
      liveParameters.valid = all((
        abs(liveParameters.angleOffsetDeg) < 10.0,
        0.2 <= liveParameters.stiffnessFront <= 5.0,
        0.2 <= liveParameters.stiffnessRear <= 5.0,
        liveParameters.kActuator > 0,
        liveParameters.kRest > 0,
        liveParameters.kDamp > 0,
      ))
      liveParameters.stiffnessFrontStd = float(P[States.STIFFNESS_FRONT])
      liveParameters.stiffnessRearStd = float(P[States.STIFFNESS_REAR])
      liveParameters.angleOffsetFastStd = float(P[States.ANGLE_OFFSET])
      liveParameters.stdDump = [float(p) for p in P]

      msg.valid = sm.all_checks()

      if sm.frame % 1200 == 0:  # once a minute
        params = {
          'carFingerprint': CP.carFingerprint,
          'stiffnessFront': liveParameters.stiffnessFront,
          'stiffnessRear': liveParameters.stiffnessRear,
          'angleOffsetDeg': liveParameters.angleOffsetDeg,
          'kActuator': liveParameters.kActuator / CP.kActuator,
          'kRest': liveParameters.kRest / CP.kRest,
          'kDamp': liveParameters.kDamp / CP.kDamp,
          'P' : [float(p) for p in learner.kf.P.diagonal()],
        }
        put_nonblocking("LiveParameters", json.dumps(params))

      pm.send('liveParameters', msg)


if __name__ == "__main__":
  main()
