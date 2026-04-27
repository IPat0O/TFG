# Script with:
# - Functionality to generate a periodically forced pendulum dataset
# - Functionality to plot the periodically forced pendulum dataset, to verify it's working

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

def periodically_forced_pendulum_ode(t, y, epsilon=0.4):
    """
    ODE for a periodically forced pendulum
    """
    x1, x2 = y
    dx1_dt = x2
    dx2_dt = -np.sin(x1) + epsilon * np.cos(t)
    return [dx1_dt, dx2_dt]

def periodically_forced_pendulum_ode_vec(t, y, epsilon=0.4):
    """
    Vectorized ODE for a periodically forced pendulum.
    y has shape (2*N,) where N is the number of initial conditions.
    """
    n = len(y) // 2
    x1 = y[:n]
    x2 = y[n:]
    dx1_dt = x2
    dx2_dt = -np.sin(x1) + epsilon * np.cos(t)
    return np.concatenate([dx1_dt, dx2_dt])

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

def poincare_map_vec(y0, epsilon, t0=0.0):
    """
    Vectorized Poincare map for the periodically forced pendulum ODE.
    y0 has shape (N, 2).
    """
    n = y0.shape[0]
    y0_flat = np.concatenate([y0[:, 0], y0[:, 1]])
    t1 = t0 + 2 * np.pi
    t_span = (t0, t1)
    sol = solve_ivp(periodically_forced_pendulum_ode_vec, t_span, y0_flat, args=(epsilon,), rtol=1e-6, atol=1e-6)
    y_final = sol.y[:, -1]
    return np.column_stack([y_final[:n], y_final[n:]]), t1

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

    n_grid_x1 = 5
    n_grid_x2 = 5
    total_starting_points = n_grid_x1 * n_grid_x2
    generated_points = 0

    x1_values = np.linspace(x1_min, x1_max, n_grid_x1)
    x2_values = np.linspace(x2_min, x2_max, n_grid_x2)

    plt.figure(figsize=(8, 8))
    for x1_0 in x1_values:
        for x2_0 in x2_values:
            y0 = [x1_0, x2_0]
            orbit, time_points = generate_orbit(y0, epsilon, n_iterations)
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
    plt.title(fr"Orbits in Phase Space from Grid of Initial Conditions for perturbation strength $\epsilon = {epsilon}$")

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    print()  # finish progress line cleanly
    plt.show()

def generate_and_save_dataset():
    """
    Generate and save a dataset for the periodically forced pendulum ODE.
    """
    dataset_path = "data/periodically_forced_pendulum.npz"

    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        print("Overwrite? (y/n)")
        overwrite = input()
        if overwrite != "y":
            print("Dataset not overwritten. Exiting.")
            return
        else:
            print("Overwriting dataset...")
            os.remove(dataset_path)

    x1_min, x1_max = -2.6, -0.3
    x2_min, x2_max = -1.2, 1.2
    epsilon = 0.40
    n_iterations = 800

    n_grid_x1 = 50
    n_grid_x2 = 50
    total_starting_points = n_grid_x1 * n_grid_x2
    generated_points = 0

    x1_values = np.linspace(x1_min, x1_max, n_grid_x1)
    x2_values = np.linspace(x2_min, x2_max, n_grid_x2)

    trajectories = []
    t = np.linspace(0, 2 * np.pi * (n_iterations - 1), n_iterations)

    plt.figure(figsize=(8, 8))
    for x1_0 in x1_values:
        for x2_0 in x2_values:
            y0 = [x1_0, x2_0]
            orbit= generate_orbit(y0, epsilon, n_iterations)
            orbit = np.array(orbit)

            trajectories.append(orbit)

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
    plt.title(fr"Orbits in Phase Space from Grid of Initial Conditions for perturbation strength $\epsilon = {epsilon}$")

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    print()  # finish progress line cleanly
    plt.show()

    print(f"Dataset successfully generated. Saving to {dataset_path}")
    np.savez(dataset_path, t=t, trajectories=trajectories)
    print(f"Dataset successfully saved to {dataset_path}")

#TODO: function that plots the dataset

if __name__ == "__main__":
    generate_and_save_dataset()