#!/usr/bin/env python3
import gc
import math
import json

import numpy as np
import cereal.messaging as messaging

from selfdrive.locationd.models.car_kf import CarKalman, ObservationKind, States
from selfdrive.locationd.models.constants import GENERATED_DIR
from common.realtime import set_realtime_priority, DT_MDL
from common.params import Params, put_nonblocking
from selfdrive.swaglog import cloudlog
from common.numpy_fast import clip
from cereal import car

MAX_ANGLE_OFFSET_DELTA = 20 * DT_MDL  # Max 20 deg/s
ROLL_MAX_DELTA = np.radians(20.0) * DT_MDL  # 20deg in 1 second is well within curvature limits
ROLL_MIN, ROLL_MAX = math.radians(-10), math.radians(10)

class ParamsLearner:
  def __init__(self, CP, stiffness_front, stiffness_rear, front_ratio, mass_ratio, moment_ratio, angle_offset, P_initial=None):
    self.kf = CarKalman(GENERATED_DIR, stiffness_front, stiffness_rear, front_ratio, mass_ratio, moment_ratio, angle_offset, P_initial)

    self.kf.filter.set_global("mass", CP.mass)
    self.kf.filter.set_global("rotational_inertia", CP.rotationalInertia)
    self.kf.filter.set_global("center_to_front", CP.centerToFront)
    self.kf.filter.set_global("wheelbase", CP.wheelbase)
    self.kf.filter.set_global("stiffness_front", CP.tireStiffnessFront)
    self.kf.filter.set_global("stiffness_rear", CP.tireStiffnessRear)
    self.kf.filter.set_global("steer_ratio", CP.steerRatio)

    self.steering_ratio = CP.steerRatio
    self.active = False
    self.roll = 0.0

  def handle_log(self, t, which, msg):
    if which == 'liveLocationKalman':
      yaw_rate = msg.angularVelocityCalibrated.value[2]
      yaw_rate_std = msg.angularVelocityCalibrated.std[2]
      yaw_rate_valid = msg.angularVelocityCalibrated.valid
      yaw_rate_valid = yaw_rate_valid and 0 < yaw_rate_std < 10  # rad/s
      yaw_rate_valid = yaw_rate_valid and abs(yaw_rate) < 1  # rad/s

      roll = msg.orientationNED.value[0]
      roll_std = np.radians(1) if np.isnan(msg.orientationNED.std[0]) else msg.orientationNED.std[0]
      if not msg.orientationNED.valid and ROLL_MIN < roll < ROLL_MAX:
        # This is done to bound the road roll estimate when localizer values are invalid
        roll = 0.0
        roll_std = np.radians(10.0)
      self.roll = clip(roll, self.roll - ROLL_MAX_DELTA, self.roll + ROLL_MAX_DELTA)

      if self.active:
        if msg.posenetOK:
          if yaw_rate_valid:
            self.kf.predict_and_observe(t, ObservationKind.ROAD_FRAME_YAW_RATE,
                                        np.array([[yaw_rate]]), np.array([np.atleast_2d(yaw_rate_std**2)]))
          self.kf.predict_and_observe(t, ObservationKind.ROAD_ROLL,
                                      np.array([[self.roll]]), np.array([np.atleast_2d(roll_std**2)]))
    elif which == 'carState':
      speed, steering_angle = msg.vEgo, msg.steeringAngleDeg
      in_linear_region = abs(steering_angle) < 3*self.steering_ratio
      self.active = speed > 5

      if self.active:
        if in_linear_region:
          self.kf.predict_and_observe(t, ObservationKind.STEER_ANGLE, np.array([[math.radians(steering_angle)]]))
        self.kf.predict_and_observe(t, ObservationKind.ROAD_FRAME_X_SPEED, np.array([[speed]]))

def main(sm=None, pm=None):
  gc.disable()
  set_realtime_priority(5)

  if sm is None:
    sm = messaging.SubMaster(['liveLocationKalman', 'carState'], poll=['liveLocationKalman'])
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

  # Check if starting values are sane
  if params is not None:
    try:
      stiffness_sane = (0.2 <= params.get('stiffnessFront') <= 5.0) and (0.2 <= params.get('stiffnessRear') <= 5.0)
      angle_offset_sane = abs(params.get('angleOffsetAverageDeg')) < 10.0
      front_to_rear_sane = 0.5 <= params.get('frontRatio') <= 2
      mass_sane = 0.5 <= params.get('massRatio') <= 3.0
      moment_sane = 0.2 <= params.get('momentRatio') <= 5
      params_sane = all((angle_offset_sane, front_to_rear_sane, stiffness_sane, mass_sane, moment_sane,))
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
      'frontRatio': 1.0,
      'massRatio': 1.0,
      'momentRatio': 1.0,
      'angleOffsetAverageDeg': 0.0,
    }
    cloudlog.info("Parameter learner resetting to default values")

  learner = ParamsLearner(CP, params['stiffnessFront'], params['stiffnessRear'], params['frontRatio'], params['massRatio'], params['momentRatio'], math.radians(params['angleOffsetAverageDeg']))
  angle_offset = params['angleOffsetAverageDeg']

  while True:
    sm.update()
    if sm.all_alive_and_valid():
      for which in sorted(sm.updated.keys(), key=lambda x: sm.logMonoTime[x]):
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          learner.handle_log(t, which, sm[which])

    if sm.updated['liveLocationKalman']:
      x = learner.kf.x
      P = np.sqrt(learner.kf.P.diagonal())
      if not all(map(math.isfinite, x)):
        cloudlog.error("NaN in liveParameters estimate. Resetting to default values")
        learner = ParamsLearner(CP, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        x = learner.kf.x

      angle_offset = clip(math.degrees(x[States.ANGLE_OFFSET]), angle_offset - MAX_ANGLE_OFFSET_DELTA, angle_offset + MAX_ANGLE_OFFSET_DELTA)

      msg = messaging.new_message('liveParameters')
      msg.logMonoTime = sm.logMonoTime['carState']

      msg.valid = sm.all_alive_and_valid()
      liveParameters = msg.liveParameters
      liveParameters.posenetValid = True
      liveParameters.sensorValid = True
      liveParameters.stiffnessFront = float(x[States.STIFFNESS_FRONT])
      liveParameters.stiffnessRear = float(x[States.STIFFNESS_REAR])
      liveParameters.frontRatio = float(x[States.FRONT_RATIO])
      liveParameters.massRatio = float(x[States.MASS_RATIO])
      liveParameters.momentRatio = float(x[States.MOMENT_RATIO])
      liveParameters.roll = float(x[States.ROAD_ROLL])
      liveParameters.angleOffsetAverageDeg = angle_offset
      liveParameters.angleOffsetDeg = angle_offset
      liveParameters.yawRate = float(x[States.YAW_RATE])
      liveParameters.valid = all((
        abs(liveParameters.angleOffsetAverageDeg) < 10.0,
        abs(liveParameters.angleOffsetDeg) < 10.0,
        0.2 <= liveParameters.stiffnessFront <= 5.0,
        0.2 <= liveParameters.stiffnessRear <= 5.0,
        0.5 <= liveParameters.frontRatio <= 2.0,
        0.5 <= liveParameters.massRatio <= 3.0,
        0.2 <= liveParameters.momentRatio <= 5.0,
      ))
      liveParameters.stiffnessFrontStd = float(P[States.STIFFNESS_FRONT])
      liveParameters.stiffnessRearStd = float(P[States.STIFFNESS_REAR])
      liveParameters.frontRatioStd = float(P[States.FRONT_RATIO])
      liveParameters.massStd = float(P[States.MASS_RATIO])
      liveParameters.momentStd = float(P[States.MOMENT_RATIO])
      liveParameters.angleOffsetAverageStd = float(P[States.ANGLE_OFFSET])

      if sm.frame % 1200 == 0:  # once a minute
        params = {
          'carFingerprint': CP.carFingerprint,
          'stiffnessFront': liveParameters.stiffnessFront,
          'stiffnessRear': liveParameters.stiffnessRear,
          'frontRatio': liveParameters.frontRatio,
          'massRatio': liveParameters.massRatio,
          'momentRatio': liveParameters.momentRatio,
          'angleOffsetAverageDeg': liveParameters.angleOffsetAverageDeg,
        }
        put_nonblocking("LiveParameters", json.dumps(params))

      pm.send('liveParameters', msg)


if __name__ == "__main__":
  main()
