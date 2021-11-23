""" Diagnose crashes in computation of covariance matrix. Local, offline version.

"""
import os
import numpy as np
import random
import dask
from math import radians
from sklearn.metrics.pairwise import haversine_distances
from climate.utils import load_zarr_dataset
from climate.kalman_filter import DryRunEnsembleKalmanFilter


def main():
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()
    client = Client(cluster)
    
    base_folder = "/home/cedric/PHD/Dev/Climate/Data/"
    TOT_ENSEMBLES_NUMBER = 2
    
    (dataset_mean, dataset_members,
            dataset_instrumental, dataset_reference,
            dataset_mean_zarr, dataset_members_zarr) = load_zarr_dataset(
                    base_folder, TOT_ENSEMBLES_NUMBER)

    # Choose random couples.
    n_samples = 6000
    inds_i = random.choices(list(range(dataset_members_zarr.anomaly.shape[1])), k=n_samples)
    inds_j = random.choices(list(range(dataset_members_zarr.anomaly.shape[1])), k=n_samples)

    covs = []
    dists = []
    cov = dask.array.cov(
            dataset_members_zarr.anomaly, dataset_members_zarr.anomaly, rowvar=False)
    for i, j in zip(inds_i, inds_j):
        covs.append(cov[i, j].compute())
        coords_i = [radians(dataset_members_zarr.latitude.values[i]),
                radians(dataset_members_zarr.longitude.values[i])]
        coords_j = [radians(dataset_members_zarr.latitude.values[j]),
                radians(dataset_members_zarr.longitude.values[j])]

        dists.append(6371*haversine_distances([coords_i, coords_j])[0, 1])

    np.save("./variogram_covs.npy", covs)
    np.save("./variogram_dists.npy", dists)


if __name__ == "__main__":
    main()
