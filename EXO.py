import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt
import time

xml_path = 'elbow_exoskeleton.xml'  # xml file (assumes this is in the same folder as this file)
simend = 60  # simulation time (60 seconds for demonstration)
print_camera_config = 0  # set to 1 to print camera config

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# AAN Control Parameters
class AANController:
    def __init__(self):
        # Control parameters
        self.target_elbow_angle = 90.0  # Target angle in degrees
        self.initial_motor_assistance = 0.8  # Start with 80% motor assistance
        self.final_motor_assistance = 0.1   # End with 10% motor assistance
        self.current_motor_assistance = self.initial_motor_assistance
        
        # Adaptation parameters
        self.cycles_per_adaptation = 15  # Reduce assistance every 15 cycles
        self.adaptation_rate = 0.05      # Reduce assistance by 5% each time
        self.cycle_count = 0
        self.direction = 1  # 1 for extension, -1 for flexion
        
        # Human effort simulation (simulating patient's limited capability)
        self.human_strength_factor = 0.3  # Patient can only provide 30% of required torque
        self.fatigue_factor = 1.0  # Decreases over time to simulate fatigue
        
        # Data logging
        self.time_log = []
        self.elbow_angle_log = []
        self.motor_torque_log = []
        self.human_effort_log = []
        self.assistance_level_log = []
        self.cycle_completion_log = []
        
        # PID Controller parameters
        self.kp = 50.0
        self.ki = 5.0
        self.kd = 10.0
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        # Cycle detection
        self.last_angle = 0.0
        self.movement_threshold = 5.0  # degrees
        self.cycle_start_time = 0.0
        
    def simulate_human_effort(self, desired_torque, current_time):
        """Simulate human effort with fatigue and limited strength"""
        # Simulate fatigue (strength decreases over time)
        self.fatigue_factor = max(0.5, 1.0 - current_time * 0.01)
        
        # Human can only provide limited torque
        max_human_torque = desired_torque * self.human_strength_factor * self.fatigue_factor
        
        # Add some noise to simulate real human behavior
        noise = np.random.normal(0, 0.1)
        human_torque = max_human_torque + noise
        
        return human_torque
    
    def update_assistance_level(self):
        """Update motor assistance level based on cycle count"""
        if self.cycle_count > 0 and self.cycle_count % self.cycles_per_adaptation == 0:
            if self.current_motor_assistance > self.final_motor_assistance:
                self.current_motor_assistance = max(
                    self.final_motor_assistance,
                    self.current_motor_assistance - self.adaptation_rate
                )
                print(f"Cycle {self.cycle_count}: Motor assistance reduced to {self.current_motor_assistance:.2f}")
    
    def control_step(self, model, data):
        """Main control step for AAN algorithm"""
        current_time = data.time
        
        # Get current joint angles (convert to degrees)
        elbow_angle = np.rad2deg(data.sensordata[2])  # elbow position sensor
        elbow_velocity = np.rad2deg(data.sensordata[3])  # elbow velocity sensor
        
        # Cycle detection and target setting
        if self.direction == 1:  # Extension phase
            target_angle = self.target_elbow_angle
            if elbow_angle >= self.target_elbow_angle - 5:  # Near target
                self.direction = -1
                self.cycle_count += 0.5  # Half cycle completed
        else:  # Flexion phase
            target_angle = 0.0
            if elbow_angle <= 5:  # Near zero
                self.direction = 1
                self.cycle_count += 0.5  # Full cycle completed
                self.update_assistance_level()
        
        # PID Control for desired torque calculation
        error = target_angle - elbow_angle
        self.integral_error += error * model.opt.timestep
        derivative_error = (error - self.previous_error) / model.opt.timestep
        
        desired_torque = (self.kp * error + 
                         self.ki * self.integral_error + 
                         self.kd * derivative_error)
        
        self.previous_error = error
        
        # Simulate human effort
        human_torque = self.simulate_human_effort(desired_torque, current_time)
        
        # Calculate motor assistance torque
        motor_torque = desired_torque * self.current_motor_assistance
        
        # Apply torques to the simulation
        data.ctrl[1] = motor_torque  # Elbow motor control
        
        # Data logging
        if len(self.time_log) == 0 or current_time - self.time_log[-1] >= 0.1:  # Log every 0.1 seconds
            self.time_log.append(current_time)
            self.elbow_angle_log.append(elbow_angle)
            self.motor_torque_log.append(motor_torque)
            self.human_effort_log.append(human_torque)
            self.assistance_level_log.append(self.current_motor_assistance)
            self.cycle_completion_log.append(self.cycle_count)
    
    def plot_results(self):
        """Generate plots for analysis"""
        if len(self.time_log) < 2:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Elbow Angle vs Time
        ax1.plot(self.time_log, self.elbow_angle_log, 'b-', linewidth=2)
        ax1.axhline(y=90, color='r', linestyle='--', label='Target Angle')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Elbow Angle (degrees)')
        ax1.set_title('Elbow Joint Angle During Rehabilitation')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Motor Torque vs Time
        ax2.plot(self.time_log, self.motor_torque_log, 'g-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Motor Torque (Nm)')
        ax2.set_title('Motor Assistance Torque')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Assistance Level vs Cycles
        cycle_times = []
        assistance_at_cycles = []
        for i, cycle in enumerate(self.cycle_completion_log):
            if i == 0 or cycle != self.cycle_completion_log[i-1]:
                cycle_times.append(cycle)
                assistance_at_cycles.append(self.assistance_level_log[i])
        
        ax3.plot(cycle_times, assistance_at_cycles, 'ro-', linewidth=2, markersize=6)
        ax3.set_xlabel('Rehabilitation Cycles')
        ax3.set_ylabel('Motor Assistance Level')
        ax3.set_title('AAN Control: Reducing Motor Assistance')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Human vs Motor Effort
        ax4.plot(self.time_log, self.human_effort_log, 'orange', linewidth=2, label='Human Effort')
        ax4.plot(self.time_log, self.motor_torque_log, 'blue', linewidth=2, label='Motor Assistance')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Torque (Nm)')
        ax4.set_title('Human Effort vs Motor Assistance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('aan_control_results.png', dpi=300, bbox_inches='tight')
        plt.show()

# Initialize AAN Controller
aan_controller = AANController()

def init_controller(model, data):
    """Initialize the controller here. This function is called once, in the beginning"""
    print("AAN Control System Initialized")
    print(f"Initial motor assistance: {aan_controller.initial_motor_assistance*100:.1f}%")
    print(f"Target reduction to: {aan_controller.final_motor_assistance*100:.1f}%")
    print(f"Adaptation every {aan_controller.cycles_per_adaptation} cycles")

def controller(model, data):
    """Put the controller here. This function is called inside the simulation."""
    aan_controller.control_step(model, data)

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
    elif act == glfw.PRESS and key == glfw.KEY_P:
        # Plot results when 'P' is pressed
        aan_controller.plot_results()

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

# Get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "AAN Control - Elbow Rehabilitation", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Set camera configuration for better view
cam.azimuth = 45
cam.elevation = -30
cam.distance = 2.0
cam.lookat = np.array([0.0, 0.0, 0.8])

# Initialize the controller
init_controller(model, data)

# Set the controller
mj.set_mjcb_control(controller)

print("Starting AAN Control Simulation...")
print("Press 'P' to plot results during simulation")
print("Press 'Backspace' to reset simulation")

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time >= simend):
        print("Simulation completed!")
        print(f"Total cycles completed: {aan_controller.cycle_count:.1f}")
        print(f"Final motor assistance level: {aan_controller.current_motor_assistance*100:.1f}%")
        aan_controller.plot_results()
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # print camera configuration (help to initialize the view)
    if (print_camera_config == 1):
        print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =', cam.elevation, ';', 'cam.distance = ', cam.distance)
        print('cam.lookat =np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()