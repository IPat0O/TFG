# Script with:
# - Functionality to generate a periodically forced pendulum dataset
# - Functionality to plot the periodically forced pendulum dataset, to verify it's working

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def periodically_forced_pendulum_ode(t, y, epsilon=0.4):
    """
    ODE for a periodically forced pendulum
    """
    x1, x2 = y
    dx1_dt = x2
    dx2_dt = -np.sin(x1) + epsilon * np.cos(t)
    return [dx1_dt, dx2_dt]

#the poincare map is just the flow map evaluated from t0 to t1=t0+2pi
#so in poincare_map i will just integrate the ode from t0 to t1 and return the last point and the final time
#later, as in the paper they follow an iterative approach, i will just iterate the poincare map function
def poincare_map(y0, epsilon, t0=0.0):
    """
    Poincare map for the periodically forced pendulum ODE
    Takes into account the initial time t0 and returns the final time and the final point.
    """
    t1 = t0 + 2 * np.pi
    t_span = (t0, t1)
    sol = solve_ivp(periodically_forced_pendulum_ode, t_span, y0, args=(epsilon,), rtol=1e-6, atol=1e-6)
    return sol.y[:, -1], t1

def generate_orbit(y0, epsilon, n_iterations):
    """
    Generate an orbit for the periodically forced pendulum ODE Poincaré map,
    keeping track of ongoing time for the non-autonomous system.
    """
    orbit = [y0]
    t = 0.0
    y = y0
    for _ in range(n_iterations):
        y, t = poincare_map(y, epsilon, t)
        orbit.append(y)
    return orbit

# Now plot the orbits of a grid of initial conditions in the periodically forced pendulum phase space
def test_poincare_map():
    """
    Test the poincare map for the periodically forced pendulum ODE, by plotting the orbits
    starting from a grid of initial conditions within the spatial domain:
      -2.6 <= x1 <= -0.3
      -1.2 <= x2 <= 1.2
    """
    x1_min, x1_max = -2.6, -0.3
    x2_min, x2_max = -1.2, 1.2
    epsilon = 0.40
    n_iterations = 800

    n_grid_x1 = 30
    n_grid_x2 = 30
    total_starting_points = n_grid_x1 * n_grid_x2
    generated_points = 0

    x1_values = np.linspace(x1_min, x1_max, n_grid_x1)
    x2_values = np.linspace(x2_min, x2_max, n_grid_x2)

    plt.figure(figsize=(8, 8))
    for x1_0 in x1_values:
        for x2_0 in x2_values:
            y0 = [x1_0, x2_0]
            orbit = generate_orbit(y0, epsilon, n_iterations)
            orbit = np.array(orbit)
            plt.scatter(orbit[:, 0], orbit[:, 1], s=1)
            generated_points += 1
            progress = generated_points / total_starting_points
            print(
                f"Generated starting points: {generated_points}/{total_starting_points} "
                f"({progress:.0%})",
                end="\r",
                flush=True,
            )

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(fr"Orbits in Phase Space from Grid of Initial Conditions for $\epsilon = {epsilon}$")

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    print()  # finish progress line cleanly
    plt.show()

if __name__ == "__main__":
    test_poincare_map()