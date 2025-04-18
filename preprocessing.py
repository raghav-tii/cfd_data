import numpy as np
from dmd_test import read_exodus_field, dynamic_mode_decomposition
import argparse


def read_exodus_file(exodus_file):
    times = read_exodus_field(exodus_file, "time_whole")
    velx = read_exodus_field(exodus_file, "vals_nod_var1")
    vely = read_exodus_field(exodus_file, "vals_nod_var2")
    # pressure = read_exodus_field(exodus_file, "vals_nod_var2")
    return times, np.stack((velx.data, vely.data), axis=2)


def compute_dmd(data_matrix, rank):
    return dynamic_mode_decomposition(data_matrix, r=rank)


def preprocess_exodus_data(exodus_file, output_file):
    # Step 1: Read simulation data (assumes velocities are returned as a 3D array: [time, nodes, components])
    times, velocities = read_exodus_file(exodus_file)
    time_steps = (
        times[:-1] - times[1:]
    )  # Right now this is just all ones due to the dataset

    print("Loaded data from the files successfully!")
    print("Shape of times array: ", times.shape)
    print("Shape of velocities array: ", velocities.shape)

    # Step 2: Reshape velocities for DMD (flatten spatial dimensions)
    num_times, num_nodes, num_components = velocities.shape
    
    data_matrix = velocities.reshape(num_times, -1).T
    # shape: (num_nodes*num_components, num_times)

    # Step 3: Compute DMD
    rank = 10
    dmd_eigs, dmd_modes = compute_dmd(data_matrix, rank)
    print("Dynamic mode decomposition performed successfully!")
    print("Shape of dmd_modes array: ", dmd_modes.shape)

    # Step 4: Transform timeseries using DMD modes
    transformed_timeseries = data_matrix.T @ dmd_modes
    print("Shape of transformed_timeseries array: ", transformed_timeseries.shape)

    # Step 5: Compute time derivatives
    time_derivatives = time_steps.reshape(-1, 1) * (
        transformed_timeseries[:-1, :] - transformed_timeseries[1:, :]
    )
    print("Shape of time_derivatives array: ", time_derivatives.shape)

    # Step 6: Save transformed timeseries
    np.savez_compressed(
        output_file + f"_rank_{rank}",
        times=times,
        transformed_timeseries=transformed_timeseries[:, :-1],
        time_derivatives=time_derivatives,
        dmd_modes=dmd_modes,
        dmd_eigs=dmd_eigs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CFD data from Exodus file using DMD."
    )
    parser.add_argument("exodus_file", type=str, help="Path to the input Exodus file")
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the transformed timeseries (npz file)",
    )
    args = parser.parse_args()

    preprocess_exodus_data(args.exodus_file, args.output_file)
