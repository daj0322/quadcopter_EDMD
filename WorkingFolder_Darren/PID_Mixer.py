import numpy as np

class pid_mixer:

    def fct_mixer(u, kT, kD, l, min_omega=0, max_omega=2000):
        """
        Parameters:
        u : list or array [u1, u2, u3, u4]
            u1: Total Thrust (N)
            u2: Roll Torque (N*m)  (phi)
            u3: Pitch Torque (N*m) (theta)
            u4: Yaw Torque (N*m)   (psi)
        kT : float
            Thrust coefficient
        kD : float
            Drag torque coefficient
        l : float
            Arm length (meters)
            
        Returns:
        omega : np.array
            Array of 4 motor speeds [w0, w1, w2, w3] (rad/s)
        """
        
        total_thrust = u[0]
        roll_torque  = u[1]  # u2
        pitch_torque = u[2]  # u3
        yaw_torque   = u[3]  # u4

        arm = l/np.sqrt(2)

        # 1. Determine Force required per motor
        # Thrust contribution
        t_lift = total_thrust / 4.0
        
        # Term for pitch
        t_pitch = pitch_torque / (4.0 * arm)
        
        # Term for roll
        t_roll = roll_torque / (4.0 * arm)
        
        # Term for yaw
        # The ratio between Torque and Force for a prop is kD/kT. 
        # Convert the Yaw Torque back into a Force differential.
        # The scalar is derived from: u4 = (kD/kT) * (Terms)
        # Force contribution is: u4 * (kT/kD) / 4
        t_yaw = (yaw_torque * kT) / (4.0 * kD)

        # 2. Mix the forces
        # Motor 0 (Front-Right):
        T0 = t_lift - t_roll + t_pitch + t_yaw
        
        # Motor 1 (Rear-Right):
        T1 = t_lift - t_roll - t_pitch - t_yaw
        
        # Motor 2 (Rear-Left):
        T2 = t_lift + t_roll - t_pitch + t_yaw
        
        # Motor 3 (Front-Left):
        T3 = t_lift + t_roll + t_pitch - t_yaw

        # 3. Convert Force (T) to Speed (Omega)
        # T = kT * omega^2  =>  omega = sqrt(T / kT)
        forces = np.array([T0, T1, T2, T3])
        
        # Clip negative forces to 0
        forces = np.maximum(forces, 0)
        
        omega = np.sqrt(forces / kT)
        
        # 4. Apply Motor Limits
        omega = np.clip(omega, min_omega, max_omega)
        
        return omega