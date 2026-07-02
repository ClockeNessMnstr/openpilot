import numpy as np
import numpy.polynomial.polynomial as P
import numpy.linalg as LA

class DiscreteController:
  def __init__(self):
    self.ref_set = False

  def reset(self):
    self.e = mat(np.zeros(len(self.ke)))
    self.u = mat(np.zeros(len(self.ku)))

    self.identity = np.identity(len(self.ke))

    if self.ref_set:
      self.r = mat(np.zeros(len(self.kr)))
      self.y_bar = mat(np.zeros(len(self.ky)))

      self.Rho = self.identity * self.P0

      self.d = np.zeros_like(self.identity)

  def kf_hold(self):
    self.Rho = self.identity * self.P0

  def dRho(self):
    dRho = np.ones_like(self.identity)
    dRho = self.theta @ self.theta.T
    return dRho

  #recalculate the last error from corrected u0
  def recalculate(self, last_output):
    self.u[0][0] = last_output
    self.e[0][0] += (self.u @ self.ku - self.e @ self.ke) / self.ke[0][0]

  def update(self, r, y=0, last_output=None):
    if last_output is not None:
      self.recalculate(last_output)

    if self.ref_set:
      self.predict_and_observe(y)

    #next timestep
    self.e = roll(self.e)
    self.e[0][0] = r - y

    #calculate next step desired
    self.u = roll(self.u)
    self.u[0][0] += (self.e @ self.ke - self.u @ self.ku) / self.ku[0][0] # TODO: NORMALIZE = 1

    return float(self.u[0][0])

  def predict_and_observe(self, y):
    self.d = roll(self.d)
    self.d[:, 0] += (self.e @ self.a - (self.d @ self.ku).T)[0] / self.ku[0][0]
    self.u = (self.d.T @ self.theta).T

    self.predict_Rho()
    self.predict_theta(y)
    self.set_gains()

    self.r = roll(self.r)
    self.r[0][0] = y + self.e[0][0]

    self.y_bar = roll(self.y_bar)
    self.y_bar[0][0] += (self.r @ self.kr - self.y_bar @ self.ky) / self.ky[0][0]

  # Model of gains is that they are constant
  def predict_theta(self, y):
    self.theta = self.filter_theta(y)

  def predict_Rho(self):
    self.Rho = self.filter_Rho() + self.dRho() * self.R_sys
    self.Rho = np.diag(np.diagonal(self.Rho))

  def filter_g(self):
    num = (self.Rho @ self.get_psi())
    den = self.get_psi().T @ self.Rho @ self.get_psi() + self.R_obs
    return (num / den)

  def filter_theta(self, y):
    return self.theta + (self.filter_g() @ self.filter_epsilon(y))

  def filter_Rho(self):
    return ((self.identity - (self.filter_g() @ self.get_psi().T)) @ self.Rho)

  def filter_epsilon(self, y):
    return mat(self.y_bar[0][0] - y)

  def get_psi(self):
    return mat(self.d[:, 0]).T

  def set_gains(self):
    self.ke = self.a @ self.theta

  def get_dlti(self):
    return (self.ke.T, self.ku.T.flatten() , self.dt)

  def set_dlti(self, d_system):
    # TODO
    # NORMALIZED! : ku[0] = 1 , Division by k[0][0] can be skipped
    self.ku = mat(d_system[1] / d_system[1][0]).T
    self.ke = mat(d_system[0][0] / d_system[1][0]).T
    self.dt = d_system[2]
    self.ref_set = False
    self.reset()
    self.set_pfd()

  def set_ref(self, sigma, delta=0.2, P0=1, R_sys=1/100000, R_obs=1, delay=0):
    # TODO: Generalize and/or condense this.
    self.P0 = P0
    self.R_sys = R_sys
    self.R_obs = R_obs

    rho = self.dt / sigma
    mu = 0.25 + (0.26 * delta)
    p1 = -2*np.exp(-rho/(2*mu))*np.cos(rho*np.sqrt(4*mu-1)/(2*mu))
    p2 = np.exp(-rho/mu)

    num = np.append(np.zeros(delay), 1+p1+p2)
    den = [1, p1, p2]

    self.kr = mat(num).T
    self.ky = mat(den).T

    self.ref_set = True
    self.reset()

  def set_pfd(self):
    c = (self.ku[::-1]).flatten()
    denroots = P.polyroots(c)

    # TODO: multiplicity

    a = [self.ku.flatten()]
    for i in range(len(denroots)):
      root = denroots[i]
      poly = P.polydiv(c, P.polyfromroots(np.atleast_1d(root)))[0]
      poly = np.pad(poly, (0,1)).tolist()
      a.append(poly[::-1])

    self.a = mat(a).T
    x = LA.solve(self.a, self.ke.flatten())
    self.theta = mat(x).T

def roll(a):
  return np.roll(a, 1, axis=1)

def mat(a):
  return np.atleast_2d(a)
