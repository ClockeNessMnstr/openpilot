#!/usr/bin/env python3
import os
import numpy as np

from casadi import SX, vertcat, sin, cos

from common.realtime import sec_since_boot
from selfdrive.controls.lib.drive_helpers import LAT_MPC_N as N
from selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from selfdrive.modeld.constants import T_IDXS

if __name__ == '__main__':  # generating code
  from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from selfdrive.controls.lib.lateral_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython  # pylint: disable=no-name-in-module, import-error

LAT_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LAT_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LAT_MPC_DIR, "acados_ocp_lat.json")
X_DIM = 8
P_DIM = 12
MODEL_NAME = 'lat'
ACADOS_SOLVER_TYPE = 'SQP_RTI'
COST_DIM = 4

def gen_lat_model():
  model = AcadosModel()
  model.name = MODEL_NAME
  
  #constants
  g = ACCELERATION_DUE_TO_GRAVITY

  # parameters
  u_ego = SX.sym('u_ego')
  u_inv = SX.sym('u_ego')
  roll = SX.sym('roll')
  
  cF = SX.sym('stiffness_front')
  cR = SX.sym('stiffness_rear')
  aF = SX.sym('center_to_front')
  aR = SX.sym('center_to_rear')
  m_inv = SX.sym('mass')
  j_inv = SX.sym('rotational_inertia')
  k_actuator = SX.sym('actuator_gain')
  k_rest = SX.sym('restorative_torque_gain')
  k_damp = SX.sym('steer_damping_gain')
  model.p = vertcat(u_ego, u_inv, roll, cF, cR, aF, aR, m_inv, j_inv, k_actuator, k_rest, k_damp)

  # set up states & controls
  x_ego = SX.sym('x_ego')
  y_ego = SX.sym('y_ego')
  psi_ego = SX.sym('psi_ego')
  r_ego = SX.sym('r_ego')
  v_ego = SX.sym('v_ego')
  d_ego = SX.sym('d_ego')
  dr_ego = SX.sym('dr_ego')
  actuator_ego = SX.sym('da_ego')
  model.x = vertcat(x_ego, y_ego, psi_ego, r_ego, v_ego, d_ego, dr_ego, actuator_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  y_ego_dot = SX.sym('y_ego_dot')
  psi_ego_dot = SX.sym('psi_ego_dot')
  r_ego_dot = SX.sym('r_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  d_ego_dot = SX.sym('d_ego_dot')
  dr_ego_dot = SX.sym('dr_ego_dot')
  actuator_ego_dot = SX.sym('da_ego_dot')
  
  # controls
  d_actuator = SX.sym('tire_angle_acc')
  model.u = vertcat(d_actuator)

  model.xdot = vertcat(x_ego_dot, y_ego_dot, psi_ego_dot, r_ego_dot, v_ego_dot, d_ego_dot, dr_ego_dot, actuator_ego_dot)

  # dynamics model
  f_expl = vertcat(u_ego * cos(psi_ego) - v_ego * sin(psi_ego),
                   u_ego * sin(psi_ego) + v_ego * cos(psi_ego),
                   r_ego,
                   (((-(cF*aF - cR*aR))*v_ego + (-(cF*aF**2 + cR*aR**2))*r_ego)*(u_inv) + (cF*aF)*d_ego)*j_inv,
                   (((-(cF    + cR   ))*v_ego + (-(cF*aF    - cR*aR   ))*r_ego)*(u_inv) + (cF   )*d_ego)*m_inv - u_ego*r_ego + g*roll,
                   dr_ego,
                   actuator_ego*k_actuator -k_rest*r_ego*u_ego,
                   d_actuator)
  model.f_expl_expr = f_expl
  return model


def gen_lat_ocp():
  ocp = AcadosOcp()
  ocp.model = gen_lat_model()

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  Q = np.diag(np.zeros(COST_DIM - 1))
  QR = np.diag(np.zeros(COST_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, y_ego, psi_ego, r_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2], ocp.model.x[3]
  u_ego_cost = ocp.model.p[0]

  ocp.parameter_values = np.zeros((P_DIM, ))

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_DIM - 1, ))
  costs = [
    x_ego,
    y_ego,
    psi_ego * (u_ego_cost),
    r_ego   * (u_ego_cost)
  ]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # set constraints
  ocp.constraints.constr_type = 'BGH'
  ocp.constraints.idxbx = np.array([2, 5, 6, 7])
  ocp.constraints.ubx = np.array([np.radians(90), np.radians(12), np.radians(3), 1])
  ocp.constraints.lbx = np.array([-np.radians(90), -np.radians(12), -np.radians(3), -1])
  x0 = np.zeros((X_DIM,))
  ocp.constraints.x0 = x0
  
  ocp.constraints.idxbu = np.array([0])
  ocp.constraints.ubu = np.array([0.7])
  ocp.constraints.lbu = np.array([-0.7])
  
  # set prediction horizon
  ocp.solver_options.tf = np.array(T_IDXS)[N]
  ocp.solver_options.shooting_nodes = np.array(T_IDXS)[:N+1]
  ocp.solver_options.sim_method_num_steps = (np.array(range(N))+1)*8
  ocp.solver_options.sim_method_jac_reuse = True
  ocp.solver_options.qp_solver_cond_N = 1

  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LateralMpc():
  def __init__(self, x0=np.zeros(X_DIM)):
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.reset(x0)

  def reset(self, x0=np.zeros(X_DIM), p0=np.zeros(P_DIM)):
    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.yref = np.zeros((N+1, COST_DIM))
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_DIM - 1])

    # Somehow needed for stable init
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))
      self.solver.set(i, 'p', np.zeros(P_DIM))
    self.solver.constraints_set(0, "lbx", x0)
    self.solver.constraints_set(0, "ubx", x0)
    self.solver.solve()
    self.solution_status = 0
    self.solve_time = 0.0
    self.cost = 0

  def set_weights(self, path_weight, heading_weight, curv_weight, curv_rate_weight):
    W = np.asfortranarray(np.diag([path_weight, path_weight, heading_weight, curv_weight]))
    for i in range(N):
      self.solver.cost_set(i, 'W', W)
    #TODO hacky weights to keep behavior the same
    self.solver.cost_set(N, 'W', (3/20.)*W[:COST_DIM - 1,:COST_DIM - 1])

  def run(self, x0, p, x_pts, y_pts, heading_pts, curv_pts):
    x0_cp = np.copy(x0)
    p_cp = np.copy(p)
    self.solver.constraints_set(0, "lbx", x0_cp)
    self.solver.constraints_set(0, "ubx", x0_cp)
    u_ego_cost = p_cp[:, 0]
    self.yref[:, 0] = x_pts
    self.yref[:, 1] = y_pts
    self.yref[:, 2] = heading_pts * (u_ego_cost)
    self.yref[:, 3] = curv_pts * (u_ego_cost)**2
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
      self.solver.set(i, "p", p_cp[i])
    self.solver.set(N, "p", p_cp[N])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_DIM - 1])

    t = sec_since_boot()
    self.solution_status = self.solver.solve()
    self.solve_time = sec_since_boot() - t

    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')
    self.cost = self.solver.get_cost()


if __name__ == "__main__":
  ocp = gen_lat_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
  #AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
