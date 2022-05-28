import numpy as np
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc, X_DIM
from selfdrive.controls.lib.drive_helpers import CONTROL_N, MPC_COST_LAT, LAT_MPC_N
from selfdrive.controls.lib.lane_planner import LanePlanner, TRAJECTORY_SIZE
from selfdrive.controls.lib.desire_helper import DesireHelper
import cereal.messaging as messaging
from cereal import log


class LateralPlanner:
  def __init__(self, CP, use_lanelines=True, wide_camera=False, use_model_speed=True):
    self.use_lanelines = use_lanelines
    self.LP = LanePlanner(wide_camera)
    self.DH = DesireHelper()

    self.last_cloudlog_t = 0
    self.solution_invalid_cnt = 0

    self.path_xyz = np.zeros((TRAJECTORY_SIZE, 3))
    self.plan_yaw = np.zeros((TRAJECTORY_SIZE,))
    self.t_idxs = np.arange(TRAJECTORY_SIZE)
    self.y_pts = np.zeros(TRAJECTORY_SIZE)
    
    self.use_model_speed = use_model_speed
    self.rotation_radius = CP.centerToFront # - 0.5 # 0.5 ~= commaToFront

    self.lat_mpc = LateralMpc()
    self.reset_mpc(np.zeros(X_DIM))
    self.lat_mpc.set_weights(MPC_COST_LAT.PATH, MPC_COST_LAT.HEADING, MPC_COST_LAT.CURV, MPC_COST_LAT.CURV_RATE)

  def reset_mpc(self, x0=np.zeros(X_DIM)):
    self.x0 = x0
    self.lat_mpc.reset(x0=self.x0)

  def update(self, sm):
    v_ego = sm['carState'].vEgo
    measured_curvature = sm['controlsState'].curvature

    # Parse model predictions
    md = sm['modelV2']
    self.LP.parse_model(md)
    if len(md.position.x) == TRAJECTORY_SIZE and len(md.orientation.x) == TRAJECTORY_SIZE:
      self.path_xyz = np.column_stack([md.position.x, md.position.y, md.position.z])
      self.t_idxs = np.array(md.position.t)
      self.plan_yaw = list(md.orientation.z)
      plan_yaw_rate = list(md.orientationRate.z)
      
      self.speed_forward = np.linalg.norm(np.column_stack([md.velocity.x, md.velocity.y, md.velocity.z]), axis=1)
      self.plan_curvature = plan_yaw_rate / self.speed_forward
      
      self.plan_distance_forward = ((self.speed_forward[1:] + self.speed_forward[:-1]) / 2) * (self.t_idxs[1:] - self.t_idxs[:-1])
      self.plan_distance_forward = np.cumsum(np.insert(self.plan_distance_forward, 0, 0))
      if not self.use_model_speed:
        self.speed_forward = np.repeat(v_ego, len(self.speed_forward))
      self.distance_forward = ((self.speed_forward[1:] + self.speed_forward[:-1]) / 2) * (self.t_idxs[1:] - self.t_idxs[:-1])
      self.distance_forward = np.cumsum(np.insert(self.distance_forward, 0, 0))
      
      self.distance_forward = self.distance_forward[:LAT_MPC_N + 1]
      self.speed_forward = self.speed_forward[:LAT_MPC_N + 1]

    # Lane change logic
    lane_change_prob = self.LP.l_lane_change_prob + self.LP.r_lane_change_prob
    self.DH.update(sm['carState'], sm['controlsState'].active, lane_change_prob)

    # Turn off lanes during lane change
    if self.DH.desire == log.LateralPlan.Desire.laneChangeRight or self.DH.desire == log.LateralPlan.Desire.laneChangeLeft:
      self.LP.lll_prob *= self.DH.lane_change_ll_prob
      self.LP.rll_prob *= self.DH.lane_change_ll_prob

    # Calculate final driving path and set MPC costs
    d_path_xyz = self.path_xyz if not self.use_lanelines else self.LP.get_d_path(self.speed_forward[0], self.t_idxs, self.path_xyz)
    
    low_speed = 5 # hold costs at this speed when stopping/starting
    rotation_radius = np.interp(self.speed_forward, [0, low_speed], [0, self.rotation_radius])
    low_speed_factor = np.interp(self.speed_forward, [0, low_speed], [low_speed, 0])
    
    # TODO save path history over length of rotation radius and prepend to plan before 0
    self.y_pts = np.interp(self.distance_forward, self.plan_distance_forward, d_path_xyz[:, 1])
    heading_pts = np.interp(self.distance_forward - rotation_radius, self.plan_distance_forward, self.plan_yaw)
    curv_pts = np.interp(self.distance_forward - rotation_radius, self.plan_distance_forward, self.plan_curvature)

    assert len(self.y_pts) == LAT_MPC_N + 1
    assert len(heading_pts) == LAT_MPC_N + 1
    assert len(curv_pts) == LAT_MPC_N + 1
    self.x0[1] = rotation_radius[0]*np.sin(heading_pts[0])
    self.x0[3] = measured_curvature
    p = np.column_stack([self.speed_forward, rotation_radius, low_speed_factor])
    self.lat_mpc.run(self.x0, p, self.y_pts, heading_pts, curv_pts, None)

    #  Check for infeasible MPC solution
    mpc_nans = np.isnan(self.lat_mpc.x_sol[:, 3]).any()
    t = sec_since_boot()
    if mpc_nans or self.lat_mpc.solution_status != 0:
      self.reset_mpc()
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Lateral mpc - nan: True")

    if self.lat_mpc.cost > 20000. or mpc_nans:
      self.solution_invalid_cnt += 1
    else:
      self.solution_invalid_cnt = 0

  def publish(self, sm, pm):
    plan_solution_valid = self.solution_invalid_cnt < 2
    plan_send = messaging.new_message('lateralPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'modelV2'])

    lateralPlan = plan_send.lateralPlan
    lateralPlan.modelMonoTime = sm.logMonoTime['modelV2']
    lateralPlan.laneWidth = float(self.LP.lane_width)
    lateralPlan.dPathPoints = self.y_pts.tolist()
    lateralPlan.psis = self.lat_mpc.x_sol[0:CONTROL_N, 2].tolist()
    lateralPlan.curvatures = self.lat_mpc.x_sol[0:CONTROL_N, 3].tolist()
    lateralPlan.curvatureRates = [float(x) for x in self.lat_mpc.u_sol[0:CONTROL_N - 1]] + [0.0]
    lateralPlan.lProb = float(self.LP.lll_prob)
    lateralPlan.rProb = float(self.LP.rll_prob)
    lateralPlan.dProb = float(self.LP.d_prob)

    lateralPlan.mpcSolutionValid = bool(plan_solution_valid)
    lateralPlan.solverExecutionTime = self.lat_mpc.solve_time

    lateralPlan.desire = self.DH.desire
    lateralPlan.useLaneLines = self.use_lanelines
    lateralPlan.laneChangeState = self.DH.lane_change_state
    lateralPlan.laneChangeDirection = self.DH.lane_change_direction

    pm.send('lateralPlan', plan_send)
