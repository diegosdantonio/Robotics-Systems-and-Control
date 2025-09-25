# ============================================================
# MuJoCo + KUKA LBR iiwa14 Forward Kinematics Demo
# ============================================================
# This script:
#   1. Implements forward kinematics (FK) using homogeneous
#      transformations (rotation + translation).
#   2. Compares FK results against MuJoCo’s reported end-effector pose.
#   3. Prints numerical errors for position (and orientation, if extended).
#      You can plot them and compare
#
# ------------------------------------------------------------
# PRO TIPS / HINTS FOR STUDENTS:
#   - FK is a chain product of joint transforms: ^0T7 = T1(R,p)*T2(R,p)*...*T7(R,p).
#   - Read joint states from data.qpos (radians).
#   - Be mindful of axis choice and link offsets; simplify first, refine later.
#   - Compare both position and orientation errors.
# ============================================================
from audioop import error

import mujoco
import mujoco.viewer
import numpy as np
import time

# ------------------------------------------------------------
# Load the KUKA iiwa14 model
# ------------------------------------------------------------
model = mujoco.MjModel.from_xml_path("./car/car.xml")
data = mujoco.MjData(model)
opt = mujoco.MjvOption()

# Visualization options (optional)
opt.frame = mujoco.mjtFrame.mjFRAME_BODY
opt.label = mujoco.mjtLabel.mjLABEL_BODY

# ------------------------------------------------------------
# Actuator setup (for demonstration control inputs)
# ------------------------------------------------------------
act_names = ["forward", "turn"]
act_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names}

def set_target_values(u):

    data.ctrl[act_ids["forward"]] = u[0]
    data.ctrl[act_ids["turn"]] = u[1]

##
##
##

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi



def controller(x, xd):
    kp1 = 1.
    kp2 = 0.1

    e  =  np.linalg.norm(xd - x)
    e_theta = wrap_pi(np.arctan2(xd[1] - x[1], xd[0] - x[0]) - x[2])

    print(e)

    u = np.array([kp1*e,
                  kp2*e_theta])  # Example constant disturbance
    return u


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================
print("Started")
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation started. Press ESC to exit.")
    t_start = time.time()

    viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

    while viewer.is_running():
        # Time since start
        t = time.time() - t_start
        print("TIME:", t)



        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
        x = data.xpos[body_id]

        # controller

        x_des = np.array([1, 1, 0.0])
        u = controller(x, x_des)

        # Update joint targets

        set_target_values(u)

        print(x)
        # Advance simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
