import numpy as np
from netCDF4 import Dataset
from scipy.linalg import svd

import matplotlib.pyplot as plt


def read_exodus_field(filename, field_name):
    """Read velocity field from Exodus file using netCDF4."""
    ds = Dataset(filename)
    # Exodus stores variables as [time, nodes, components]
    vel = ds.variables[field_name][:]
    ds.close()
    return vel


def read_exodus_coords(filename):
    """Read coords from Exodus file using netCDF4."""
    ds = Dataset(filename)
    coords = np.column_stack([ds.variables["coordx"][:], ds.variables["coordy"][:]])
    ds.close()
    return coords


def dynamic_mode_decomposition(X, r=None):
    """Perform Dynamic Mode Decomposition (DMD) on data matrix X."""
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    U, S, Vh = svd(X1, full_matrices=False)
    if r is not None:
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]
    A_tilde = U.T @ X2 @ Vh.T @ np.diag(1 / S)
    eigvals, W = np.linalg.eig(A_tilde)
    Phi = U @ W
    return eigvals, Phi


def plot_field(coords, field, title="Velocity field"):
    plt.figure()
    plt.tricontourf(coords[:, 0], coords[:, 1], field, levels=50)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()


def main():
    filename = "test_data_1.e"
    coords = read_exodus_coords(filename)
    velx = read_exodus_field(filename, "vals_nod_var1")
    vely = read_exodus_field(filename, "vals_nod_var2")

    time_step = -1

    # Use only last time step if multiple present
    if velx.ndim == 2:
        velx_t = velx[time_step]
    if vely.ndim == 2:
        vely_t = vely[time_step]
    vel = np.column_stack((velx_t, vely_t))

    # Visualize the magnitude of the final velocity field
    magnitude = np.linalg.norm(vel, axis=1)
    plot_field(coords, velx_t)
    plot_field(coords, vely_t)
    plot_field(coords, magnitude)

    # Perform DMD

    eigvalsx, modesx = dynamic_mode_decomposition(velx.data.T, r=10)
    eigvalsy, modesy = dynamic_mode_decomposition(vely.data.T, r=10)

    # Plot realm part of few DMD modes
    mode_indices = [0, 1, 2]
    for mode in mode_indices:
        plot_field(coords, np.real(modesx[:, mode]), title="modes of vx")
        plot_field(coords, np.real(modesy[:, mode]), title="modes of vy")


if __name__ == "__main__":
    main()
