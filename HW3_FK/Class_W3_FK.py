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

import mujoco
import mujoco.viewer
import numpy as np
import time

# ------------------------------------------------------------
# Load the KUKA iiwa14 model
# ------------------------------------------------------------
model = mujoco.MjModel.from_xml_path("./kuka_iiwa_14/scene.xml")
data = mujoco.MjData(model)
opt = mujoco.MjvOption()

# Visualization options (optional)
opt.frame = mujoco.mjtFrame.mjFRAME_BODY
opt.label = mujoco.mjtLabel.mjLABEL_BODY

# ------------------------------------------------------------
# Actuator setup (for demonstration control inputs)
# ------------------------------------------------------------
act_names = [f"actuator{i}" for i in range(1,8)]
act_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names}

def set_target_angles(t):
    """
    Simple test pattern: sinusoidal target positions.
    In reality, you may want to send fixed poses or trajectories.
    """
    data.ctrl[act_ids["actuator1"]] = np.sin(t)
    data.ctrl[act_ids["actuator2"]] = np.cos(t)
    data.ctrl[act_ids["actuator3"]] = np.sin(t)
    data.ctrl[act_ids["actuator4"]] = np.cos(t)
    data.ctrl[act_ids["actuator5"]] = np.cos(t)
    data.ctrl[act_ids["actuator6"]] = np.sin(t)

# ------------------------------------------------------------
# Helper functions: basic rotations + homogeneous transform
# ------------------------------------------------------------
def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])

def Ry(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])

def Rz(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,-sa,0],[sa, ca,0],[0,0,1]])

def T_of(R, p):
    """Return homogeneous transform given rotation R and translation p."""
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = p
    return T

# ------------------------------------------------------------
# Forward Kinematics Function
# ------------------------------------------------------------
def forward_kinematics(q):
    """
    Compute simplified FK for KUKA iiwa14.
    Args:
        q : array of 7 joint angles [rad]
    Returns:
        Pos_efector : 3D position of end-effector in base frame
        R_efector   : 3x3 rotation matrix of end-effector
    """
    # Assumed link offsets (toy model; approximate)
    link_lengths = [1., 1., 1., 1., 1., 1., 1.]

    # NOTE: Axes chosen arbitrarily here (not actual KUKA convention).
    # Students: refine with true axes from the XML for accuracy or inspecting from the Figure 4.
    T1 = T_of(Rz(q[0]), [0, 0, link_lengths[0]])
    T2 = T_of(Ry(q[1]), [0, 0, link_lengths[1]])
    T3 = T_of(Ry(q[2]), [0, 0, link_lengths[2]])
    T4 = T_of(Rx(q[3]), [0, 0, link_lengths[3]])
    T5 = T_of(Rz(q[4]), [0, 0, link_lengths[4]])
    T6 = T_of(Rx(q[5]), [0, 0, link_lengths[5]])
    T7 = T_of(Rx(q[6]), [0, 0, link_lengths[6]])



    # Final FK (base to joint 7)
    T_fk = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7

    Pos_efector = T_fk[:3,3]
    R_efector = T_fk[:3,:3]
    return Pos_efector, R_efector

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

        # Update joint targets
        set_target_angles(t)

        # ------------------------------------------------------------
        # Forward Kinematics: Compare FK vs MuJoCo end-effector pose
        # ------------------------------------------------------------
        # Get MuJoCo’s pose of the end-effector body
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "iiwa_link_7")
        pos_mj = data.xpos[body_id]
        R_mj = data.xmat[body_id].reshape(3, 3)

        # Read actual joint states
        q = data.qpos.copy()[:7]

        # Compute FK (student model)
        # (TODO) define your FK and plot the error to compare simulator with the theoretical model
        Pos_fk, R_fk = forward_kinematics(q)

        # Position error
        error_pos = np.linalg.norm(Pos_fk - pos_mj)
        print("FK position:", Pos_fk,
              " MuJoCo position:", pos_mj,
              " error:", error_pos)



        # Advance simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
