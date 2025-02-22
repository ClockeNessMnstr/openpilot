import numpy as np
import numpy.polynomial.polynomial as P

class DiscreteController():
  def __init__(self, gains, Z, T, rate=100):
    self.update_controller(gains, Z, T, rate=rate)
    self.reset()

  def reset(self):
    self.e = np.zeros(len(self.ke))
    self.u = np.zeros(len(self.ku))
    self.d = np.zeros((len(self.c), len(self.u)))

  def recalculate(self, last_output):
    self.u[0] = last_output
    self.e[0] += np.divide(np.sum(np.multiply(self.ku, self.u)) - np.sum(np.multiply(self.ke, self.e)), self.ke[0])

  def update(self, error, last_output):
    #recalculate the last error from corrected u0
    self.recalculate(last_output)

    #logging
    for i in range(len(self.c)):
      self.d[i][0] += np.divide(np.sum(np.multiply(self.c[i]/self.a[0], self.e)) - np.sum(np.multiply(self.ku, self.d[i])), self.ku[0])
      self.d[i] = np.roll(self.d[i], 1)

    #next timestep
    self.e = np.roll(self.e, 1)
    self.u = np.roll(self.u, 1)

    #calculate next step desired
    self.e[0] = error
    self.u[0] += np.divide(np.sum(np.multiply(self.ke, self.e)) - np.sum(np.multiply(self.ku, self.u)), self.ku[0])

    return float(self.u[0])

  def update_gains(self, gains):
    self.gains = gains
    b = [0]
    for i in range(len(self.c)):
      b = P.polyadd(b, self.gains[i]*self.c[i])
    self.ke = np.array([i / self.a[0] for i in b])
    self.b = b

  def update_controller(self, gains, Z, T, rate):
    G = [[P.polymul(Z[i][j][::-1], P.polyval(1/rate, T[i][j][::-1]))[::-1].real.tolist() for j in range(len(Z[i]))] for i in range(len(Z))]
    self.a = [1]
    c = np.array(np.zeros(len(G)), dtype=object)
    for g in G:
      self.a = P.polymul(self.a[::-1], g[1][::-1])[::-1]
    for i in range(len(G)):
      c[i] = G[i][0]
      for j in range(len(G)):
        if i != j:
          c[i] = P.polymul(c[i][::-1], G[j][1][::-1])[::-1]
    maxlen = 0
    for i in range(len(c)):
      maxlen = maxlen if maxlen >= len(c[i]) else len(c[i])
    self.c = np.zeros((len(c), maxlen))
    for i in range(len(c)):
      self.c[i] = np.append(np.zeros(maxlen-len(c[i])), c[i])
    self.ku = np.array([i / self.a[0] for i in self.a])
    self.update_gains(gains)

    if self.a[0] == 0 or len(self.a) < len(self.b):
      raise ValueError("Controller is non-causal. Output depends on future value. e.g. Forward Euler PID")
    if np.linalg.matrix_rank(self.c)<len(self.c):
      raise ValueError("Controller gains are not linearly independent")

