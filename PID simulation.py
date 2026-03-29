import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    """A discrete PID controller implementation."""
    def __init__(self, Kp, Ki, Kd, dt, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.min_out, self.max_out = output_limits
        
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, setpoint, measured_value):
        # 1. Calculate Error
        error = setpoint - measured_value
        
        # 2. Proportional Term
        P = self.Kp * error
        
        # 3. Integral Term (Accumulates error over time)
        self.integral += error * self.dt
        I = self.Ki * self.integral
        
        # 4. Derivative Term (Rate of change of error)
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative
        
        # 5. Total Output
        output = P + I + D
        
        # 6. Apply Anti-windup / Output Limits (Motor thrust limits)
        if self.min_out is not None:
            output = max(self.min_out, output)
        if self.max_out is not None:
            output = min(self.max_out, output)
            
        self.prev_error = error
        return output

# ==========================================
# SIMULATION SETUP: UAV Altitude Control
# ==========================================

# 1. Define the Physics Parameters (The Plant)
m = 1.0       # Mass of the vehicle (kg)
g = 9.81      # Acceleration due to gravity (m/s^2)
c = 0.5       # Aerodynamic drag coefficient (N*s/m)

# 2. Define Time Parameters
dt = 0.05               # Controller time step (50 ms)
total_time = 15.0       # Total simulation time in seconds
time = np.arange(0, total_time, dt)
n_steps = len(time)

# 3. Initialize State Arrays (for plotting)
altitude = np.zeros(n_steps)
velocity = np.zeros(n_steps)
thrust_cmd = np.zeros(n_steps)

# 4. Initial Conditions & Target
altitude[0] = 0.0       # Starting at ground level
velocity[0] = 0.0
setpoint = 10.0         # Target altitude is 10 meters

# 5. Initialize the Controller
# Tuning these values changes the system's behavior (Overdamped, Underdamped, etc.)
# Max thrust is capped at 25 Newtons, Min thrust is 0 (motors can't pull down)
pid = PIDController(Kp=8.0, Ki=2.5, Kd=4.5, dt=dt, output_limits=(0, 25))

# ==========================================
# MAIN SIMULATION LOOP
# ==========================================

for i in range(1, n_steps):
    # Current states
    current_alt = altitude[i-1]
    current_vel = velocity[i-1]
    
    # --- Control Step ---
    # The PID controller calculates required thrust to reach the setpoint
    thrust = pid.update(setpoint, current_alt)
    thrust_cmd[i] = thrust
    
    # --- Physics Step (F = ma) ---
    # Net Force = Thrust - Gravity - Aerodynamic Drag
    net_force = thrust - (m * g) - (c * current_vel)
    
    # a = F/m
    acceleration = net_force / m
    
    # Euler Integration to find new velocity and position
    velocity[i] = current_vel + (acceleration * dt)
    altitude[i] = current_alt + (velocity[i] * dt)

    # Floor constraint (the vehicle cannot fall through the ground)
    if altitude[i] < 0:
        altitude[i] = 0.0
        velocity[i] = 0.0

# ==========================================
# PLOTTING RESULTS
# ==========================================

plt.figure(figsize=(12, 6))

# Plot 1: Altitude Response
plt.subplot(2, 1, 1)
plt.plot(time, altitude, label='Actual Altitude (PV)', color='blue', linewidth=2)
plt.axhline(setpoint, color='red', linestyle='--', label='Target Altitude (SP)')
plt.title('UAV Altitude PID Control Response')
plt.ylabel('Altitude (meters)')
plt.grid(True)
plt.legend()

# Plot 2: Control Effort (Thrust)
plt.subplot(2, 1, 2)
plt.plot(time, thrust_cmd, label='Motor Thrust Command (OP)', color='orange')
plt.axhline(m*g, color='green', linestyle=':', label='Hover Thrust ($m \cdot g$)')
plt.xlabel('Time (seconds)')
plt.ylabel('Thrust (Newtons)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()