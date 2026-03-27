import numpy as np

class iPID_trajectory_controller:
    def __init__(self, Kp, Ki, Kd, integral_limit=np.inf, alpha = 1.0):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.alpha = alpha
        self.integral_limit = float(integral_limit)
        self.integral = 0
        self.prev_error = 0
        self.prev_meas = 0
        self.prev_ref = 0
 
    def fct_control(self, mea, ref, dt):
        error = ref - mea
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        PID = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
 
        d_mea = (mea - self.prev_meas) / dt
        self.prev_meas = mea
        d_ref = (ref - self.prev_ref) / dt
        self.prev_ref = ref
        u = PID * self.alpha

        info = -d_mea + u + d_ref
        iPID = PID + info
        # print(f"d_mea: {d_mea}")
        # print(f"d_ref: {d_ref}")
        # print(f"PID: {PID}")
        return iPID
    
    def fct_reset(self):
        self.integral = 0
        self.prev_error = 0
        self.prev_meas = 0
        self.prev_ref = 0