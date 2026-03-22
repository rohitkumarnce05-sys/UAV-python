import numpy as np
import matplotlib.pyplot as plt

def simulate_uav_altitude():
    """
    Simulates UAV altitude based on thrust inputs for three scenarios:
    Insufficient Thrust, Excessive Thrust, and Stabilized Flight (using PID).
    Generates a graph matching the Task 1 requirements.
    """

    # --- Simulation Parameters & Constants ---
    t_max = 20           # Maximum simulation time (seconds)
    dt = 0.01            # Time step (seconds)
    num_steps = int(t_max / dt)
    time_array = np.linspace(0, t_max, num_steps)

    g = 9.81             # Acceleration due to gravity (m/s^2)
    m = 1.5              # Drone mass (kg) from user context
    W = m * g            # Drone weight (N), ~14.715 N
    C_drag = 0.025       # Simplified quadratic drag coefficient (N/(m/s)^2)
    target_altitude = 10 # Desired altitude (m) for stabilized flight

    # --- Data Arrays for All Scenarios ---
    altitudes_red = np.zeros(num_steps)
    altitudes_cyan = np.zeros(num_steps)
    altitudes_green = np.zeros(num_steps)

    # --- 1. Scenario: Insufficient Thrust (Red, Nominal 10N Trial) ---
    # Model: Initial burst for takeoff, followed by constant insuffient thrust (10N).
    # This matches the curve (rise, peak, crash).
    v_r = 0.0
    y_r = 0.0
    for i in range(1, num_steps):
        t = i * dt
        
        # Simulated thrust profile for this unstable trial
        if t <= 1.2:     # Initial high-thrust pulse to simulate takeoff
            T = 22.0
        else:            # Drops to nominal "insufficient" value
            T = 10.0

        # Physics simulation: a = (T - W - Drag) / m
        # Drag acts opposing velocity: F_drag = C_drag * v^2
        a = (T - W - (C_drag * np.sign(v_r) * v_r**2)) / m
        v_r += a * dt
        y_r += v_r * dt
        
        # Ground clamping: Position cannot be less than 0
        altitudes_red[i] = max(0.0, y_r)
        if y_r < 0.0: # If it crashes, stop motion
            v_r = 0.0

    # --- 2. Scenario: Excessive Thrust (Cyan, 25N Constant) ---
    # Model: Simple constant thrust greater than weight. Continuous acceleration.
    v_c = 0.0
    y_c = 0.0
    T_excessive = 25.0
    for i in range(1, num_steps):
        t = i * dt
        T = T_excessive
        
        a = (T - W - (C_drag * np.sign(v_c) * v_c**2)) / m
        v_c += a * dt
        y_c += v_c * dt
        altitudes_cyan[i] = max(0.0, y_c)

    # --- 3. Scenario: Stabilized Flight (Green, Optimized Thrust) ---
    # Model: This is the critical case. Simple constant thrust can't settle.
    # We simulate a PID (Proportional-Integral-Derivative) controller to regulate altitude.
    v_g = 0.0
    y_g = 0.0
    
    # PID gains (tuned to match the visual step-response overshoot and damping)
    Kp = 15.0     # Proportional gain (rise time)
    Ki = 0.6      # Integral gain (eliminates steady-state error)
    Kd = 11.0     # Derivative gain (damps oscillations)
    
    integral_error = 0.0
    last_error = 0.0
    T_max_uav = 40.0 # Maximum possible motor thrust saturation (N)

    for i in range(1, num_steps):
        t = i * dt
        
        # Calculate controller values based on altitude error
        error = target_altitude - y_g
        integral_error += error * dt
        derivative_error = (error - last_error) / dt
        last_error = error
        
        # Generate control signal (thrust modulation above weight)
        control_signal = Kp * error + Ki * integral_error + Kd * derivative_error
        
        # Actual thrust is weight compensation (FeedForward) + control signal
        T = W + control_signal
        # Saturation: Clamp thrust within physical limits [0, max]
        T = max(0.0, min(T_max_uav, T))
        
        # Physics simulation
        a = (T - W - (C_drag * np.sign(v_g) * v_g**2)) / m
        v_g += a * dt
        y_g += v_g * dt
        altitudes_green[i] = max(0.0, y_g)

    # --- Create the Graph (Matplotlib Visualization) ---
    # Create high-resolution figure
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    # Plot lines matching colors and styles (all solid, green thick)
    plt.plot(time_array, altitudes_red, color='red', label='Initial Trial: Thrust=10N', linewidth=1.5)
    plt.plot(time_array, altitudes_cyan, color='cyan', label='Excessive Thrust: Thrust=25N', linewidth=1.5)
    plt.plot(time_array, altitudes_green, color='darkgreen', label='Stabilized Flight: Optimized Thrust', linewidth=3) # THICK green line

    # --- Styling the Graph (Labels, Grid, Limits) ---
    # Set bold title and accurate axis labels
    plt.title('Drone Altitude Response vs. Thrust', fontsize=18, fontweight='bold', family='sans-serif')
    plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
    plt.ylabel('Altitude (m)', fontsize=14, fontweight='bold')

    # Enable grid with faint, dashed grey lines
    plt.grid(True, linestyle='--', alpha=0.5, color='grey')

    # Set exact limits and tick marks from image
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 22.5, 2.5), fontsize=12) # Major ticks every 2.5
    plt.ylim(0, 15)
    plt.yticks(np.arange(0, 16, 2.5), fontsize=12)   # Major ticks every 2.5

    # Create the legend, exactly matching content and style
    # Positioning: upper right (loc=1) looks closest to original
    plt.legend(loc='upper right', fontsize=11, frameon=True, fancybox=False, edgecolor='grey')

    # Ensure everything fits
    plt.tight_layout()

    # Show the plot
    plt.show()

# Run the simulation
if __name__ == "__main__":
    simulate_uav_altitude()