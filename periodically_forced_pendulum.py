# Script with:
# - Functionality to generate a periodically forced pendulum dataset
# - Functionality to plot the periodically forced pendulum dataset, to verify it's working

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

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

def generate_orbits_vec(y0, epsilon, n_iterations):
    """
    Generate orbits for many initial conditions using the vectorized Poincaré map.
    y0 has shape (N, 2). Returns array of shape (N, n_iterations + 1, 2).
    """
    n = y0.shape[0]
    trajectories = np.empty((n, n_iterations + 1, 2))
    trajectories[:, 0, :] = y0

    y = y0.copy()
    t = 0.0
    for k in range(n_iterations):
        y, t = poincare_map_vec(y, epsilon, t)
        trajectories[:, k + 1, :] = y
        print(f"Poincaré iterate {k + 1}/{n_iterations}", end="\r", flush=True)
    print()
    return trajectories

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
    dataset_path = BASE_DIR / "code" / "data" / "periodically_forced_pendulum.npz"

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
    epsilon = 0.4
    n_iterations = 800

    n_grid_x1 = 50
    n_grid_x2 = 50

    x1_values = np.linspace(x1_min, x1_max, n_grid_x1)
    x2_values = np.linspace(x2_min, x2_max, n_grid_x2)

    X1, X2 = np.meshgrid(x1_values, x2_values, indexing="ij")
    y0 = np.column_stack([X1.ravel(), X2.ravel()])

    t = np.arange(n_iterations + 1) * (2 * np.pi)
    trajectories = generate_orbits_vec(y0, epsilon, n_iterations)

    pts = trajectories.reshape(-1, 2)
    plt.figure(figsize=(8, 8))
    plt.scatter(pts[:, 0], pts[:, 1], s=0.002, color="black")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(fr"Orbits in Phase Space from Grid of Initial Conditions for perturbation strength $\epsilon = {epsilon}$")
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

    print(f"Dataset successfully generated. Saving to {dataset_path}")
    np.savez(dataset_path, t=t, trajectories=trajectories)
    print(f"Dataset successfully saved to {dataset_path}")

def load_pendulum_dataset(dataset_path=BASE_DIR / "code" / "data" / "periodically_forced_pendulum.npz"):
    """
    Load the periodically forced pendulum dataset from the given path.
    Returns a tuple (t, trajectories) if successful, otherwise None.
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return None

    data = np.load(dataset_path, allow_pickle=True)
    if "t" not in data:
        print(f"Invalid dataset: missing 't' in {dataset_path}")
        return None
    if "trajectories" not in data:
        print(f"Invalid dataset: missing 'trajectories' in {dataset_path}")
        return None

    return data["t"], data["trajectories"]

def load_and_plot_dataset(ic_only=False, mask_indices=None, mask_title=None, labels=None, cmap="tab20"):
    """
    Load an existing periodically forced pendulum dataset and plot its trajectories.
    If mask_indices is provided, plot only the trajectories corresponding to the mask indices.
    If mask_title is provided, use it as the title of the plot.
    If labels is provided, color each trajectory according to its label,
    and add a legend for the labels.
    """
    dataset_path = BASE_DIR / "code" / "data" / "periodically_forced_pendulum.npz"
    x1_min, x1_max = -2.6, -0.3
    x2_min, x2_max = -1.2, 1.2

    t, trajectories = load_pendulum_dataset(dataset_path) #dont need t to plot, for now
    if trajectories is None:
        return

    #get dimensions of the dataset
    n, m, d = trajectories.shape
    n_grid = int(np.sqrt(n)) #assuming square grid of initial conditions

    if mask_indices is not None:
        trajectories = trajectories[mask_indices]

    legend_handles = {}
    color_map = None

    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array-like.")
        if len(labels) != len(trajectories):
            raise ValueError(
                "labels length must match the number of plotted trajectories "
                f"({len(labels)} != {len(trajectories)})."
            )
        color_map = plt.get_cmap(cmap)
    else:
        color_map = None

    plt.figure(figsize=(8, 8))

    for idx, orbit in enumerate(trajectories):
        if ic_only:
            orbit = orbit[:1]
            # Dynamically set dot size so that n_grid^2 dots fill the plot limits (with some headroom).
            x_span = x1_max - x1_min
            y_span = x2_max - x2_min
            xy_area = x_span * y_span
            # Area allotted per dot: total area / (number of dots)
            area_per_dot = xy_area / n
            # Set size as a fraction of area_per_dot; tweak factor to avoid overlap/crowding
            #50000 fills up the space, but i'll use lower to have more clarity for now
            size = 45000 * area_per_dot  # matplotlib 's' is in points^2, so needs scaling #TODO: find better ways to make the ic labels fill up the space
            marker = ","
        else:
            #size = 1
            size = .002
            marker = "o"

        if labels is not None:
            label_val = int(labels[idx])
            color = color_map(label_val % color_map.N)
            plot_label = f"Label {label_val}" if label_val not in legend_handles else None
            handle = plt.scatter(orbit[:, 0], orbit[:, 1], s=size, color=color, label=plot_label, marker=marker)
            if plot_label is not None:
                legend_handles[label_val] = handle
        else:
            plt.scatter(orbit[:, 0], orbit[:, 1], s=size, marker=marker, color="black")

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    if mask_title is not None:
        plt.title(f"Orbits in Phase Space from Saved Dataset {mask_title}") #TODO: correct nomenclature of the title. applies to other plots as well.
    #else:
    #    plt.title("Orbits in Phase Space from Saved Dataset")
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()

    # Add legend if labels are given
    if labels is not None and legend_handles:
        # Sort legend by label
        handles_sorted = [legend_handles[key] for key in sorted(legend_handles)]
        labels_sorted = [f"Label {key}" for key in sorted(legend_handles)]
        #plt.legend(handles_sorted, labels_sorted, title="Labels", loc="best", markerscale=2, fontsize=10)

    plt.show()


if __name__ == "__main__":
    generate_and_save_dataset()
    load_and_plot_dataset()