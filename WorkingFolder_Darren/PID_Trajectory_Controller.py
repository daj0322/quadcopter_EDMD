import numpy as np

class PID_trajectory_controller:
    def __init__(self, kp, ki, kd, integral_limit=np.inf):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self.integral = 0.0
        self.prev_error = 0.0

    def fct_control(self, mea, ref, dt):
        error = ref - mea
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        PID = self.kp*error + self.ki*self.integral + self.kd*derivative
        return PID

    def fct_reset(self):
        self.integral = 0.0
        self.prev_error = 0.0