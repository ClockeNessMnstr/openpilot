#!/usr/bin/env python3
import random
import sys

from common.numpy_fast import clip, interp
from common.filter_simple import FirstOrderFilter
from selfdrive.controls.lib.pid import PIController

#################################################################################################################

gain_range = 10 # (1 / range -> 1 ->  range   )
ratio = 4 #   I = (P / ratio -> P -> P * ratio)

num_divs = 4# (2*n + 1)**2 graphs
_p = [[0, num_divs, 2*num_divs],[1/gain_range, 1, gain_range]]

limit = 100 # max output 
step_size = 0.5 # percent of output to request during tests. 
rate = 100 # number of steps per second
ttm = 1 # seconds to max rate actuator
rate_limit = limit / (ttm * rate) # max change in request per cycle
sec = 15 # duration of data in sec
start_offset = 0.1 #sec
override = 5 # seconds
overshoot_allowance_percent = 1
overshoot_allowance = overshoot_allowance_percent / 100
noise_intensity = 10 # noise = rand() * (noise_intensity * rate_limit) / kp

####################################################################################################################

#define the constructor call and name for controllers to be tested
def init_pids(kp, ki, kd, kf):
  pid = []
  name = []

  if simple:
    pid.append(PI_Simple(kp, ki, k_f=kf, pos_limit=limit, neg_limit=-limit))
    name.append("PI_Simple")
  
  if classic:
    pid.append(PI_Classic(kp, ki, k_f=kf, pos_limit=limit, neg_limit=-limit))
    name.append("PI_Classic")

  if backfeed:
    pid.append(PI_Backfeed(kp, ki, k_f=kf, pos_limit=limit, neg_limit=-limit))
    name.append("PI Integrator Backfeed")

  if discrete:
    pid.append(PIDTrapazoid(kp, ki, kd, pos_limit=limit, neg_limit=-limit))
    name.append("PID Error Recalculate")
  
  if current:
    pid.append(PIController(([0, 1], [kp, kp]), ([0, 1], [ki, ki]), k_f=kf, pos_limit=limit, neg_limit=-limit))
    name.append("pid.py")

  return pid, name
#####################################################################################################################

# define how the update values are sent to the controller classes
def update(pid, setpoint, measure, last_output):
  if isinstance(pid, (PI_Simple, PI_Classic)):
        return pid.update(setpoint, measure)
  if isinstance(pid, (PIDTrapazoid, PI_Backfeed)):
    return pid.update(setpoint, measure, last_output=last_output)
  if isinstance(pid, PIController):
    return pid.update(setpoint, measure, last_output=last_output, feedforward=setpoint)

######################################################################################################################

# No need to go past this line to use this program. But you're not here because you "need" to be are you?

# add values here to use them on the command line
if __name__ == "__main__":
  plot = False
  debug = False
  test_noise, test_override, test_ramp = False, False, False
  simple, classic, backfeed, current, discrete = False, False, False, False, False

  while len(sys.argv) > 1:
    if "noise" in sys.argv:
      test_noise = True
    if "override" in sys.argv:
      test_override = True
    if "ramp" in sys.argv: # run ramp test
      test_ramp = True

    if "plot" in sys.argv: # enable plots
      plot = True
    if "debug" in sys.argv: # only plot failures
      debug = True

    if "simple" in sys.argv: # run "myfirstpid.derp"
      simple = True
    if "classic" in sys.argv: # run classic OP PI implementation
      classic = True
    if "backfeed" in sys.argv: # run integrator backfeed PI implementation
      backfeed = True
    if "discrete"  in sys.argv: # run Trapazoidal discretization PID implementation with Error Recalculation
      discrete = True
    if "current"  in sys.argv: # run pid.py PIController
      current = True

    sys.argv.pop()
  
  if not (test_noise or test_override or test_ramp): # default to "all"
    test_noise, test_override, test_ramp = True, True, True
  if not (current or discrete or backfeed or classic or simple): # default to all
    current, discrete, classic, backfeed, simple = True, True, True, True, True

if plot:
  import matplotlib.pyplot as plt
  
##########################################################################################################################

# Add your own Controller Classes

class PI_Simple():
  def __init__(self, k_p, k_i, k_f=0.0, pos_limit=None, neg_limit=None, rate=100):
    self.kp = k_p
    self.ki = k_i
    self.kf = k_f

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit
    self.i_rate = 1.0 / rate

    self.reset()

  def reset(self):
    self.i = 0.0

  def update(self, setpoint, measurement):
    error = float(setpoint - measurement)
    self.p = error * self.kp
    self.i = self.i + error * self.ki * self.i_rate
    self.f = setpoint * self.kf
    control = self.p + self.i + self.f

    control = clip(control, self.neg_limit, self.pos_limit)
    return control


class PI_Classic():
  def __init__(self, k_p, k_i, k_f=0.0, pos_limit=None, neg_limit=None, rate=100, sat_limit=0.8):
    self.kp = k_p
    self.ki = k_i
    self.kf = k_f

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.sat_count_rate = 1.0 / rate
    self.i_unwind_rate = 0.3 / rate
    self.i_rate = 1.0 / rate
    self.sat_limit = sat_limit

    self.reset()

  def reset(self):
    self.p = 0.0
    self.i = 0.0

  def update(self, setpoint, measurement, override=False, freeze_integrator=False):

    error = float(setpoint - measurement)
    self.p = error * self.kp
    self.f = setpoint * self.kf
    if override:
      self.i *= (1 - self.i_unwind_rate)
    else:
      i = self.i + error * self.ki * self.i_rate
      control = self.p + i

      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if ((error >= 0 and (control <= self.pos_limit or i < 0.0)) or
          (error <= 0 and (control >= self.neg_limit or i > 0.0))) and \
         not freeze_integrator:
        self.i = i

    control = self.p + self.i + self.f

    control = clip(control, self.neg_limit, self.pos_limit)
    return control

class PI_Backfeed():
  def __init__(self, k_p, k_i, k_f=0.0, pos_limit=None, neg_limit=None, rate=100):
    self.k_p = k_p  # proportional gain
    self.k_i = k_i  # integral gain
    self.k_f = k_f   # feedforward gain

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.i_backfeed_rate = 2.0 / rate
    self.i_rate = 1.0 / rate

    self.reset()

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.f = 0.0
    self.control = 0
    self.control_last = 0

  def update(self, setpoint, measurement, last_output=None):
    error = float(setpoint - measurement)
    self.p = error * self.k_p
    self.f = setpoint * self.k_f

    control_clip = self.control_last - last_output
    self.i = self.i +  self.k_i * (error * self.i_rate - control_clip * self.i_backfeed_rate)

    control = self.p + self.i + self.f

    self.control_last = control
    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control

class PIDTrapazoid():
  def __init__(self, k_p, k_i, k_d, pos_limit=None, neg_limit=None, rate=100):
    self.k_p = k_p  # proportional gain
    self.k_i = k_i  # integral gain
    self.k_d = k_d  # derivative gain

    self.rate = rate

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.reset()

  def reset(self):
    self.e0, self.e1, self.e2 = 0.0, 0.0, 0.0
    self.u0, self.u1, self.u2 = 0.0, 0.0, 0.0

    self.p, self.p1, self.p2 = 0.0, 0.0, 0.0
    self.i, self.i1, self.i2 = 0.0, 0.0, 0.0
    self.d, self.d1, self.d2 = 0.0, 0.0, 0.0
    self.speed = 0.0
    self.sat_count = 0.0
    self.saturated = False

  def update(self, setpoint, measurement, last_output):
    _Ts = 1 / self.rate
    
    # calculate coefficients
    Kp, Ki, Kd = self.k_p, self.k_i, self.k_d
    a0, a1, a2 = (2*_Ts), 0, -(2*_Ts)
    b0, b1, b2 = (2*Kp*_Ts + Ki*_Ts*_Ts + 4*Kd), (2*Ki*_Ts*_Ts - 8*Kd), (Ki*_Ts*_Ts - 2*Kp*_Ts + 4*Kd)
    ke0, ke1, ke2 = b0/a0, b1/a0, b2/a0
    ku1, ku2 = a1/a0, a2/a0

    #recalculate the last error from corrected u0
    self.u0 = last_output
    self.e0 = (self.u0 + ku1*self.u1 + ku2*self.u2 - ke1*self.e1 - ke2*self.e2) / ke0 

    #calculate the last logging partials (self.u0 is already = last_output)
    self.p2, self.p1 = self.p1, self.p
    self.i2, self.i1 = self.i1, self.i
    self.d2, self.d1 = self.d1, self.d
    self.i = (_Ts*_Ts*Ki*(1*self.e0 + 2*self.e1 + 1*self.e2) / a0) - ku1*self.i1 - ku2*self.i2 
    self.p = (    _Ts*Kp*(2*self.e0 + 0*self.e1 - 2*self.e2) / a0) - ku1*self.p1 - ku2*self.p2
    self.d = (        Kd*(4*self.e0 - 8*self.e1 + 4*self.e2) / a0) - ku1*self.d1 - ku2*self.d2

    #calculate next step desired
    self.e2, self.e1, self.e0 = self.e1, self.e0, float(setpoint - measurement)
    self.u2, self.u1, self.u0 = self.u1, self.u0, (ke0*self.e0 + ke1*self.e1 + ke2*self.e2 - ku1*self.u1 - ku2*self.u2)

    self.u0 = clip(self.u0, self.neg_limit, self.pos_limit)
    return self.u0

######################################################################################################################

# Below are the tests: 

# NOTE here 1.0 * setpoint is perfect FF, FF should not impact stability
# Using 0 or 1 for FF should not change these tests.
# TODO test that 0 or 1 for FF pass override and Ramp up the same

def ratelimit(new, last):
   return clip(new, last - rate_limit, last + rate_limit)

def test_error_noise():
  num_pids = len(init_pids(0,0,0,0)[0])
  print(f"Running Noise Test: {2 * num_pids * (2*num_divs + 1) * (2*num_divs + 1)} x {noise_intensity*sec}s runs initiated...")

  for i in range(0, 2*num_divs + 1):
    for j in range (0, 2*num_divs + 1):
      kp = interp(i, _p[0], _p[1])
      _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
      ki = interp(j, _i[0], _i[1])
      
      pid, name = init_pids(kp, ki, 0.0, 0.0)
      pid_control = []
      output = []

      pid_nn, name = init_pids(kp, ki, 0.0, 0.0)
      pid_control_nn = []
      output_nn = []
      
      ss_error = []
      error = False
      for q in range(0, len(pid)):
        output.append(FirstOrderFilter(0, 1, 1/rate))
        pid_control.append(0)
        output_nn.append(FirstOrderFilter(0, 1, 1/rate))
        pid_control_nn.append(0)
        ss_error.append(0)

      if plot:
        x = []
        y = []
        for q in range(0, len(pid)):
          y.append([])
      for t in range(0, int(noise_intensity*sec*100)):
        if plot:
          x.append(t)
          for q in range(0, len(pid)):
            y[q].append(output[q].x)

        noise = 0
        target = 0
        if (t > start_offset):
          target = limit*step_size

        if target != 0:
          noise = noise_intensity * (rate_limit / kp) * ((random.randint(0, 200) - 100) / 100)
        
        for q in range(0, len(pid)):
          pid_control[q] = ratelimit(update(pid[q], target, output[q].x + noise, pid_control[q]), pid_control[q])
          output[q].update(pid_control[q])
          pid_control_nn[q] = ratelimit(update(pid_nn[q], target, output_nn[q].x, pid_control_nn[q]), pid_control_nn[q])
          output_nn[q].update(pid_control_nn[q])
          ss_error[q] += ((output_nn[q].x - output[q].x) - ss_error[q]) / (t+1)
      
      if not plot:
        print(f"Noise Test: P = {kp:5.3}, I = {ki:5.3}")
      for q in range(0, len(pid)):
        if (abs(ss_error[q]) > limit / rate):
          error = True
          print(f"Detected possible SS_error in: {name[q]}")
      if plot and ((not debug) or error):
        for q in range(0, len(pid)):
          plt.plot(x, y[q], label=name[q])
        plt.title(f"Noise Test: P = {kp:5.3}, I = {ki:5.3}")
        plt.legend()
        plt.show()

def test_override_overshoot():
  num_pids = len(init_pids(0,0,0,0)[0])
  print(f"Running Override Test: {2 * num_pids * (2*num_divs + 1) * (2*num_divs + 1)} x {sec}s runs initiated...")

  for i in range(0, 2*num_divs + 1):
    for j in range (0, 2*num_divs + 1):
      kp = interp(i, _p[0], _p[1])
      _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
      ki = interp(j, _i[0], _i[1])
      
      pid, name = init_pids(kp, ki, 0.0, 0.0)
      pid_control = []
      output = []

      pid_n, name = init_pids(kp, ki, 0.0, 0.0)
      pid_control_n = []
      output_n = []
      
      sum_overshoot_error = []
      error = False
      for q in range(0, len(pid)):
        output.append(FirstOrderFilter(0, 1, 1/rate))
        pid_control.append(0)
        output_n.append(FirstOrderFilter(0, 1, 1/rate))
        pid_control_n.append(0)
        sum_overshoot_error.append(0)

      if plot:
        x = []
        y = []
        for q in range(0, len(pid)):
          y.append([])
      for t in range(0, int(noise_intensity*sec*100)):
        if plot:
          x.append(t)
          for q in range(0, len(pid)):
            y[q].append(output[q].x)

        noise = 0
        target = 0
        if (t > start_offset):
          target = limit*step_size

        for q in range(0, len(pid)):
          pid_control[q] = ratelimit(update(pid[q], target, output[q].x + noise, pid_control[q]), pid_control[q])
          # override / hold output of controller at 0 
          if t > override*rate:
            output[q].update(pid_control[q])
            pid_control_n[q] = ratelimit(update(pid_n[q], target, output_n[q].x, pid_control_n[q]), pid_control_n[q])
            output_n[q].update(pid_control_n[q])
          else:
            pid_control[q] = 0

          # only sum up overshoot in excess of the allowance of both the unclamped signal and the target. 
          if output_n[q].x > target or output[q].x > target:
            sum_overshoot_error[q] += min(max(abs(output_n[q].x - output[q].x)-output_n[q].x*(overshoot_allowance), 0), max(abs(target - output[q].x)-target*(overshoot_allowance), 0)) / rate
      
      if not plot:
        print(f"Override Test: P = {kp:5.3}, I = {ki:5.3}")
      for q in range(0, len(pid)):
        if (sum_overshoot_error[q]/target > overshoot_allowance):
          error = True
          print(f"Exessive Overshoot measured after override. Exeeded {100*overshoot_allowance}% : {100*sum_overshoot_error[q]/target}% in: {name[q]}")

      if plot and ((not debug) or error):
        for q in range(0, len(pid)):
          plt.plot(x, y[q], label=name[q])
        plt.title(f"Override Test: P = {kp:5.3}, I = {ki:5.3}")
        plt.legend()
        plt.show()

def test_ramp_up():
  num_pids = len(init_pids(0,0,0,0)[0])
  print(f"Running Limit Compensator Test: {2 * num_pids * (2*num_divs + 1) * (2*num_divs + 1)} x {sec}s runs initiated...")
  print("This test disables actuator feedback to test rate limit compensations. Controllers without compensation cannot be evaluated")

  for i in range(0, 2*num_divs + 1):
    for j in range (0, 2*num_divs + 1):
      kp = interp(i, _p[0], _p[1])
      _i = [[0, num_divs, 2*num_divs], [kp/ratio, kp, ratio*kp]]
      ki = interp(j, _i[0], _i[1])
      
      pid, name = init_pids(kp, ki, 0.0, 0.0)
      pid_control = []
      output = []

      pid_n, name = init_pids(kp, ki, 0.0, 0.0)
      pid_control_n = []
      pid_control_n_last = []
      output_n = []
      
      sum_overshoot_error = []
      error = False
      for q in range(0, len(pid)):
        output.append(FirstOrderFilter(0, 1, 1/rate))
        pid_control.append(0)
        output_n.append(FirstOrderFilter(0, 1, 1/rate))
        pid_control_n.append(0)
        pid_control_n_last.append(0)
        sum_overshoot_error.append(0)

      if plot:
        x = []
        y = []
        y_n = []
        for q in range(0, len(pid)):
          y.append([])
          y_n.append([])
      for t in range(0, int(noise_intensity*sec*100)):
        if plot:
          x.append(t)
          for q in range(0, len(pid)):
            y[q].append(output[q].x)
            y_n[q].append(output_n[q].x)

        noise = 0
        target = 0
        if (t > start_offset):
          target = limit*step_size

        for q in range(0, len(pid)):
          pid_control[q] = ratelimit(update(pid[q], target, output[q].x + noise, pid_control[q]), pid_control[q])
          output[q].update(pid_control[q])

          pid_control_n[q] = update(pid_n[q], target, output_n[q].x, pid_control_n_last[q])
          pid_control_n_last[q] = ratelimit(pid_control_n[q], pid_control_n_last[q])
          output_n[q].update(pid_control_n_last[q])

          # sum up the (-)overshoot as good, sum up the undershoot as bad
          if output_n[q].x > target or output[q].x > target:
            sum_overshoot_error[q] += ((output[q].x - output_n[q].x) + max(target - output[q].x, 0)) / rate

      if not plot:
        print(f"Slew Rate Limit Test: P = {kp:5.3}, I = {ki:5.3}")
      for q in range(0, len(pid)):
        if (sum_overshoot_error[q]/target > overshoot_allowance):
          error = True
          print(f"Rate limit compensation caused excessive damping or instability. {100*sum_overshoot_error[q]/target}% additional error from target in: {name[q]}")

      if plot and ((not debug) or error):
        for q in range(0, len(pid)):
          plt.plot(x, y[q], label=name[q])
          plt.plot(x, y_n[q], label=f"{name[q]} disabled")
        plt.title(f"Slew Rate Limit Test: P = {kp:5.3}, I = {ki:5.3}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
  if test_noise:
    test_error_noise()
  if test_override:
    test_override_overshoot()
  if test_ramp:
    test_ramp_up()
