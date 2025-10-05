import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Symbols
theta1, theta2, theta3 = sp.symbols("theta1 theta2 theta3")
l1, l2, l3 = sp.symbols("l1 l2 l3")
theta1_dot, theta2_dot, theta3_dot = sp.symbols("theta1_dot theta2_dot theta3_dot")

# Forward kinematics (position)
x = (
    l1 * sp.cos(theta1)
    + l2 * sp.cos(theta1 + theta2)
    + l3 * sp.cos(theta1 + theta2 + theta3)
)
y = (
    l1 * sp.sin(theta1)
    + l2 * sp.sin(theta1 + theta2)
    + l3 * sp.sin(theta1 + theta2 + theta3)
)
phi = theta1 + theta2 + theta3
pos = sp.Matrix([x, y, phi])

# Jacobian (velocity kinematics)
q = sp.Matrix([theta1, theta2, theta3])
q_dot = sp.Matrix([theta1_dot, theta2_dot, theta3_dot])
J = pos.jacobian(q)
vel = J * q_dot

# Inverse velocity kinematics
xdot, ydot, phidot = sp.symbols("xdot ydot phidot")
end_effector_vel = sp.Matrix([xdot, ydot, phidot])
qdot_from_task = J.inv() * end_effector_vel

# Display results
print("Forward Kinematics (Position):")
sp.pprint(pos)

print("\nVelocity Kinematics:")
sp.pprint(vel)

print("\nJacobian Matrix:")
sp.pprint(J)

print("\nInverse Velocity Kinematics (Joint velocities):")
sp.pprint(qdot_from_task)


# Determinant of Jacobian
det_J = J.det()
print("\nDeterminant of Jacobian:")
sp.pprint(det_J)

# Check if the Jacobian is invertible
if det_J != 0:
    print("\nJacobian is invertible.")
else:
    print("\nJacobian is singular, cannot compute inverse.")

# Simplify symbolic outputs
print("\nSimplified Forward Kinematics (Position):")
sp.pprint(sp.simplify(pos))

print("\nSimplified Velocity Kinematics:")
sp.pprint(sp.simplify(vel))

print("\nSimplified Inverse Velocity Kinematics (Joint velocities):")
sp.pprint(sp.simplify(qdot_from_task))


# Define specific numerical values
values = {
    theta1: sp.pi / 2,
    theta2: sp.pi / 2,
    theta3: sp.pi / 2,
    # theta1 : sp.pi / 2,
    # theta2: sp.pi ,
    # theta3: sp.pi *1.5,
    l1: 1.0,
    l2: 1.0,
    l3: 0.5,
    xdot: 0.1,
    ydot: 0.1,
    phidot: 0.05,
}

# Evaluate forward kinematics numerically
pos_numeric = pos.evalf(subs=values)
print("\nNumerical Forward Kinematics (Position):")
sp.pprint(pos_numeric)

# Evaluate Jacobian numerically
J_numeric = J.evalf(subs=values)
print("\nNumerical Jacobian Matrix:")
sp.pprint(J_numeric)

det_val = J_numeric.det()
print("\nDeterminant of numeric Jacobian:", det_val)

try:
    J_inv_numeric = J_numeric.inv()
    end_effector_vel_numeric = end_effector_vel.evalf(subs=values)
    qdot_numeric = J_inv_numeric * end_effector_vel_numeric
    print("\nNumerical Joint Velocities from Inverse Velocity Kinematics:")
    sp.pprint(qdot_numeric)
except:
    print("Jacobian is singular at this configuration. Skipping inverse kinematics.")


theta1_val = float(values[theta1])
theta2_val = float(values[theta2])
theta3_val = float(values[theta3])

l1_val, l2_val, l3_val = values[l1], values[l2], values[l3]

# Joint coordinates
x0, y0 = 0, 0
x1, y1 = l1_val * np.cos(theta1_val), l1_val * np.sin(theta1_val)
x2, y2 = x1 + l2_val * np.cos(theta1_val + theta2_val), y1 + l2_val * np.sin(
    theta1_val + theta2_val
)
x3, y3 = x2 + l3_val * np.cos(
    theta1_val + theta2_val + theta3_val
), y2 + l3_val * np.sin(theta1_val + theta2_val + theta3_val)

plt.figure()
plt.plot([x0, x1], [y0, y1], "r-", marker="o", linewidth=2, label="Link 1")
plt.plot([x1, x2], [y1, y2], "g-", marker="o", linewidth=2, label="Link 2")
plt.plot([x2, x3], [y2, y3], "b-", marker="o", linewidth=2, label="Link 3")
plt.title("Manipulator Configuration")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


# ------------------------------
# Plot 2: End-effector trajectory
# ------------------------------
theta1_vals = np.linspace(0, np.pi / 2, 50)
traj_x, traj_y = [], []

for th1 in theta1_vals:
    x_val = (
        l1_val * np.cos(th1)
        + l2_val * np.cos(th1 + theta2_val)
        + l3_val * np.cos(th1 + theta2_val + theta3_val)
    )
    y_val = (
        l1_val * np.sin(th1)
        + l2_val * np.sin(th1 + theta2_val)
        + l3_val * np.sin(th1 + theta2_val + theta3_val)
    )
    traj_x.append(x_val)
    traj_y.append(y_val)

plt.figure()
plt.plot(traj_x, traj_y, "r-", linewidth=2)
plt.title("End-Effector Trajectory (theta1 sweep)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.show()

# Sweep theta2
theta2_vals = np.linspace(0, np.pi / 2, 50)
traj_x, traj_y = [], []

for th2 in theta2_vals:
    x_val = (
        l1_val * np.cos(theta1_val)
        + l2_val * np.cos(theta1_val + th2)
        + l3_val * np.cos(theta1_val + th2 + theta3_val)
    )
    y_val = (
        l1_val * np.sin(theta1_val)
        + l2_val * np.sin(theta1_val + th2)
        + l3_val * np.sin(theta1_val + th2 + theta3_val)
    )
    traj_x.append(x_val)
    traj_y.append(y_val)

plt.figure()
plt.plot(traj_x, traj_y, "g-", linewidth=2)
plt.title("End-Effector Trajectory (theta2 sweep)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.show()


# Sweep theta3
theta3_vals = np.linspace(0, np.pi / 2, 50)
traj_x, traj_y = [], []

for th3 in theta3_vals:
    x_val = (
        l1_val * np.cos(theta1_val)
        + l2_val * np.cos(theta1_val + theta2_val)
        + l3_val * np.cos(theta1_val + theta2_val + th3)
    )
    y_val = (
        l1_val * np.sin(theta1_val)
        + l2_val * np.sin(theta1_val + theta2_val)
        + l3_val * np.sin(theta1_val + theta2_val + th3)
    )
    traj_x.append(x_val)
    traj_y.append(y_val)

plt.figure()
plt.plot(traj_x, traj_y, "b-", linewidth=2)
plt.title("End-Effector Trajectory (theta3 sweep)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.show()
