""" Run EnKF on ubelix cluster.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import dask
from climate.utils import load_dataset
from climate.data_wrapper import DatasetWrapper
from climate.utils import build_base_forward
from climate.kalman_filter import EnsembleKalmanFilter


def main():
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    cluster = LocalCUDACluster()
    client = Client(cluster)
    
    # Loading is done using the *load_dataset* function from the *utils* submodule.
    # The user has to specify the path to the root of the data folder (use the
    # *download_and_preprocess* script to create the folder structure).
    #
    # When working on a laptop, it is useful to only consider part of the dataset.
    # The *download_and_preprocess_subset_subset* script only download the first 9
    # ensembles members out of 30. The user should specify this number to the
    # loading function (this will be automated in the future).
    base_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
    TOT_ENSEMBLES_NUMBER = 30
    
    # The loading function returns 4 datasets: the ensemble members, the ensemble
    # mean, the instrumental data and the reference dataset.
    dataset_mean, dataset_members, dataset_instrumental, dataset_reference = load_dataset(
            base_folder, TOT_ENSEMBLES_NUMBER)
    
    # Select a sub-region (Europe) to make computations easier.
    # Note that there is a bug in xarray that forces the selection of a subset of
    # the coordinates to happen in a sorted manner.
    # Hence for the latitudes (that go in decreasing order) we have to invert the
    # range.
    subset_members = dataset_members.sel(longitude=slice(0, 115), latitude=slice(90, 0))
    subset_mean = dataset_mean.sel(longitude=slice(0, 115), latitude=slice(90, 0))
    subset_instrumental = dataset_instrumental.sel(longitude=slice(0, 115), latitude=slice(0, 90))
    subset_reference = dataset_reference.sel(longitude=slice(0, 115), latitude=slice(90, 0))
    
    
    # Test Kalman filter.
    my_filter = EnsembleKalmanFilter(subset_mean, subset_members,
            subset_instrumental)
    mean_updated, members_updated = my_filter.update_mean_window('1961-01-01', '1961-06-28', 6, data_var=0.9)
    """
    
    # Plot updated mean vs non updated vs reference.
    updated = mean_updated.unstack('stacked_dim').sel(time='1961-01-16')
    updated_member = member_updated.unstack('stacked_dim').sel(time='1961-01-16')
    non_updated = subset_mean.anomaly.sel(time='1961-01-16')
    non_updated_member = subset_members.sel(member_nr=9).anomaly.sel(time='1961-01-16')
    reference = subset_reference.anomaly.sel(time='1961-01-16')
    data = subset_instrumental.anomaly.sel(time='1961-01-16')
    
    # Clip datasets to common extent.
    updated = updated.where(
                xr.ufuncs.logical_not(xr.ufuncs.isnan(reference)))
    updated_member = updated_member.where(
                xr.ufuncs.logical_not(xr.ufuncs.isnan(reference)))
    non_updated = non_updated.where(
                xr.ufuncs.logical_not(xr.ufuncs.isnan(reference)))
    non_updated_member = non_updated_member.where(
                xr.ufuncs.logical_not(xr.ufuncs.isnan(reference)))
    
    
    # Plot all with same colorbar.
    xr.set_options(cmap_sequential='RdYlBu_r')
    levels = np.arange(-5, 5, 1)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6),
            subplot_kw={'projection': ccrs.PlateCarree()})
    
    non_updated_member.plot(ax=ax1,
            cbar_kwargs={'ticks': levels, 'spacing':
            'proportional', 'label': 'anomaly (non updated)'},
            vmin=-5, vmax=5)
    updated.plot(ax=ax2, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
            'label': 'anomaly (updated)'},
            vmin=-5, vmax=5)
    reference.plot(ax=ax3, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
            'label': 'anomaly (reference)'},
            vmin=-5, vmax=5)
    updated_member.plot(ax=ax4, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
            'label': 'anomaly (updated)'},
            vmin=-5, vmax=5)
    data.plot(ax=ax4, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
            'label': 'anomaly (instrumental)'},
            vmin=-5, vmax=5)
    ax1.coastlines()
    ax2.coastlines()
    ax3.coastlines()
    ax4.coastlines()
    
    plt.savefig("first_run_kalman.png")
    plt.show()
    """
    return mean_updated, members_updated


if __name__ == "__main__":
    main()
