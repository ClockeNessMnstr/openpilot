from cereal import log
from selfdrive.controls.lib.desire_helper import DesireHelper

TRAJECTORY_SIZE = 33
# camera offset is meters from center car to camera
# model path is in the frame of the camera
PATH_OFFSET = 0.00
CAMERA_OFFSET = 0.04
LANE_WIDTH = 3.7

class LanePlanner:
  def __init__(self, wide_camera=False):
    self.DH = DesireHelper()

    self.l_lane_change_prob = 0.
    self.r_lane_change_prob = 0.
    self.lll_prob = 0.
    self.rll_prob = 0.
    self.d_prob = 0.

  def parse_model(self, md, sm):
    lane_lines = md.laneLines
    if len(lane_lines) != 4 or len(lane_lines[0].t) != TRAJECTORY_SIZE:
      return
    
    self.lll_prob = md.laneLineProbs[1]
    self.rll_prob = md.laneLineProbs[2]

    desire_state = md.meta.desireState
    if len(desire_state):
      l_lane_change_prob = desire_state[log.LateralPlan.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.LateralPlan.Desire.laneChangeRight]
      
    # Lane change logic
    lane_change_prob = l_lane_change_prob + r_lane_change_prob
    self.DH.update(sm['carState'], sm['controlsState'].active, lane_change_prob)

    # Turn off lanes during lane change
    if self.DH.desire == log.LateralPlan.Desire.laneChangeRight or self.DH.desire == log.LateralPlan.Desire.laneChangeLeft:
      self.lll_prob *= self.DH.lane_change_ll_prob
      self.rll_prob *= self.DH.lane_change_ll_prob