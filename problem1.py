import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix, cos, sin, tan, symbols, lambdify


class Derivation:
    def __init__(self):
        self.WRAD = 25
        self.WSEP = 150
        self.WBASE = 400
        self.x, self.y, self.phi = symbols("x y phi")
        self.omega, self.alpha, self.L, self.r = symbols("omega alpha L r")
        self.state = None
        self.x0, self.y0, self.phi0 = None, None, None

    def start(self):
        vals = input("Enter Initial Vals")
        x0, y0, phi0 = map(float, vals.split(","))
        self.x0, self.y0, self.phi0 = x0, y0, phi0
        T = 100
        dt = 0.1


    def eqn(self):
        v = self.r * self.omega
        dx = v * cos(self.phi)
        dy = v * sin(self.phi)
        dphi = (v / self.L) * tan(self.alpha)
        self.state = Matrix([dx, dy, dphi])


class Sim:
    def __init__(self, der):
        self.der = der
        self.f = lambdify(
            (self.der.r, self.der.omega, self.der.phi, self.der.L, self.der.alpha),
            self.der.state,
            "numpy",
        )

    def plots(self, omega_val, alpha_val, T=10, dt=0.1):
        r_val = self.der.WRAD / 100.0
        L_val = self.der.WBASE / 100.0
        phi_val = self.der.phi0
        x_val, y_val = self.der.x0, self.der.y0
        N = int(T / dt) + 1
        traj = np.zeros((N, 3))
        traj[0] = [x_val, y_val, phi_val]
        for k in range(1, N):
            dx, dy, dphi = [
                val.item()
                for val in self.f(r_val, omega_val, phi_val, L_val, alpha_val)
            ]
            x_val += dx * dt
            y_val += dy * dt
            phi_val += dphi * dt
            traj[k] = [x_val, y_val, phi_val]
        plt.plot(traj[:, 0], traj[:, 1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    der = Derivation()
    der.start()
    der.eqn()
    sim = Sim(der)

    choice = input("Use manual values (M) or random values (R)? ").strip().lower()
    if choice == "m":
        omega_val = float(input("Enter omega (wheel angular velocity): "))
        alpha_val = float(input("Enter alpha (steering angle in radians): "))
    else:
        omega_val = np.random.uniform(0, 5.0)  # random omega in [0.5, 5.0]
        alpha_val = np.random.uniform(-5, 5)  # random alpha in [-0.5, 0.5]
        print(
            f"Randomly chosen values -> omega: {omega_val:.2f}, alpha: {alpha_val:.2f}"
        )

    sim.plots(omega_val=omega_val, alpha_val=alpha_val)
