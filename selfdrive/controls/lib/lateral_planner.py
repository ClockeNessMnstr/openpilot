import numpy as np
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc, X_DIM
from selfdrive.controls.lib.drive_helpers import CONTROL_N, MPC_COST_LAT, LAT_MPC_N
from selfdrive.controls.lib.lane_planner import LanePlanner, TRAJECTORY_SIZE, LANE_WIDTH
from common.opedit_mini import read_param, write_param
import cereal.messaging as messaging


class LateralPlanner:
  def __init__(self, CP, use_lanelines=True, wide_camera=False):
    self.use_lanelines = use_lanelines
    self.LP = LanePlanner(wide_camera)

    self.lat_mpc = LateralMpc()
    self.lat_mpc.set_weights(MPC_COST_LAT.PATH, MPC_COST_LAT.HEADING, MPC_COST_LAT.CURV, MPC_COST_LAT.CURV_RATE)
    self.reset_mpc(np.zeros(X_DIM))

    self.y_pts = np.zeros(TRAJECTORY_SIZE)
    assert(TRAJECTORY_SIZE > LAT_MPC_N)
    
    self.rotation_radius = CP.centerToFront - 0.7 # 0.5 ~= commaToFront
    self.last_cloudlog_t = 0
    self.solution_invalid_cnt = 0

    write_param('weights', [MPC_COST_LAT.PATH, MPC_COST_LAT.HEADING, MPC_COST_LAT.CURV, MPC_COST_LAT.CURV_RATE])
    

  def reset_mpc(self, x0=np.zeros(X_DIM)):
    self.x0 = x0
    self.lat_mpc.reset(x0=self.x0)

  def update(self, sm):
    # Parse model predictions
    md = sm['modelV2']
    self.LP.parse_model(md, sm) # updates lane change states
    assert all(len(x) == TRAJECTORY_SIZE for x in [md.position.y, md.orientation.z, md.orientationRate.z])

    weights = read_param('weights')[0]
    self.lat_mpc.set_weights(weights[0], weights[1], weights[2], weights[3])
      
    speed_forward = np.linalg.norm(np.column_stack([md.velocity.x, md.velocity.y, md.velocity.z]), axis=1)[:LAT_MPC_N + 1]
    x_pts = list(md.position.x)[:LAT_MPC_N + 1]
    self.y_pts = list(md.position.y)[:LAT_MPC_N + 1]
    heading_pts = list(md.orientation.z)[:LAT_MPC_N + 1]
    curv_pts = list(md.orientationRate.z)[:LAT_MPC_N + 1] / speed_forward

    self.x0[0] = self.rotation_radius*np.cos(heading_pts[0])
    self.x0[1] = self.rotation_radius*np.sin(heading_pts[0])
    self.x0[2] = self.rotation_radius*curv_pts[0]
    self.x0[3] = sm['controlsState'].curvature

    rotation_radii = np.repeat(self.rotation_radius, LAT_MPC_N + 1)
    p = np.column_stack([speed_forward, rotation_radii])
    self.lat_mpc.run(self.x0, p, x_pts, self.y_pts, heading_pts, curv_pts, None)

    #  Check for infeasible MPC solution
    mpc_nans = np.isnan(self.lat_mpc.x_sol[:, 3]).any() or self.lat_mpc.solution_status != 0
    if mpc_nans:
      t = sec_since_boot()
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Lateral mpc - nan: True")

    if self.lat_mpc.cost > 20000. or mpc_nans:
      self.reset_mpc()
      self.solution_invalid_cnt += 1
    else:
      self.solution_invalid_cnt = 0

  def publish(self, sm, pm):
    plan_solution_valid = self.solution_invalid_cnt < 2
    plan_send = messaging.new_message('lateralPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'modelV2'])

    lateralPlan = plan_send.lateralPlan
    lateralPlan.modelMonoTime = sm.logMonoTime['modelV2']
    lateralPlan.dPathPoints = self.y_pts
    lateralPlan.psis = self.lat_mpc.x_sol[0:CONTROL_N, 2].tolist()
    lateralPlan.curvatures = self.lat_mpc.x_sol[0:CONTROL_N, 3].tolist()
    lateralPlan.curvatureRates = [float(x) for x in self.lat_mpc.u_sol[0:CONTROL_N - 1]] + [0.0]

    lateralPlan.mpcSolutionValid = bool(plan_solution_valid)
    lateralPlan.solverExecutionTime = self.lat_mpc.solve_time

    lateralPlan.laneWidth = float(LANE_WIDTH)
    lateralPlan.lProb = float(self.LP.lll_prob)
    lateralPlan.rProb = float(self.LP.rll_prob)
    lateralPlan.dProb = float(self.LP.d_prob)
    lateralPlan.desire = self.LP.DH.desire
    lateralPlan.laneChangeState = self.LP.DH.lane_change_state
    lateralPlan.laneChangeDirection = self.LP.DH.lane_change_direction

    pm.send('lateralPlan', plan_send)
