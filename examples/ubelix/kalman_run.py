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


import time


def main():
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    cluster = SLURMCluster(
        cores=1,
        memory="16 GB",
        death_timeout=6000,
        walltime="02:00:00",
        job_extra=['--qos="job_epyc2"', '--partition="epyc2"']
    )
    client = Client(cluster)

    # Manually define the size of the cluster.
    cluster.scale(20)
    
    # Loading is done using the *load_dataset* function from the *utils* submodule.
    # The user has to specify the path to the root of the data folder (use the
    # *download_and_preprocess* script to create the folder structure).
    #
    # When working on a laptop, it is useful to only consider part of the dataset.
    # The *download_and_preprocess_subset_subset* script only download the first 9
    # ensembles members out of 30. The user should specify this number to the
    # loading function (this will be automated in the future).
    base_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
    TOT_ENSEMBLES_NUMBER = 5
    
    # The loading function returns 4 datasets: the ensemble members, the ensemble
    # mean, the instrumental data and the reference dataset.
    dataset_mean, dataset_members, dataset_instrumental, dataset_reference = load_dataset(
            base_folder, TOT_ENSEMBLES_NUMBER)
    print("Loading done.")

    # Load in distributed memory.
    # First have to chunk the non-dask datasets.
    dataset_instrumental = dataset_instrumental.chunk()
    dataset_reference = dataset_reference.chunk()

    dataset_mean = client.persist(dataset_mean)
    dataset_members = client.persist(dataset_members)
    dataset_instrumental = client.persist(dataset_instrumental)
    dataset_reference = client.persist(dataset_reference)
    print("Loading dataset in distributed memory done.")
    client.profile(filename="dask-profile.html") # Collect usage information.
    
    # Select a sub-region (Europe) to make computations easier.
    # Note that there is a bug in xarray that forces the selection of a subset of
    # the coordinates to happen in a sorted manner.
    # Hence for the latitudes (that go in decreasing order) we have to invert the
    # range.
    MAX_LAT = 90.0
    MIN_LAT = -90.0
    MAX_LON = 180.0
    subset_members = dataset_members.sel(longitude=slice(0, MAX_LON), latitude=slice(MAX_LAT, MIN_LAT))
    subset_mean = dataset_mean.sel(longitude=slice(0, MAX_LON), latitude=slice(MAX_LAT, MIN_LAT))
    subset_instrumental = dataset_instrumental.sel(
            longitude=slice(0, MAX_LON), latitude=slice(MIN_LAT, MAX_LAT))
    subset_reference = dataset_reference.sel(
            longitude=slice(0, MAX_LON), latitude=slice(MAX_LAT, MIN_LAT))
    
    
    # Test Kalman filter.
    my_filter = EnsembleKalmanFilter(subset_mean, subset_members,
            subset_instrumental)
    print("Filter built.")
    """
    mean_updated, members_updated = my_filter.update_mean_window(
            '1961-01-01', '1961-06-28', 6, data_var=0.9)
    """

    tmp = my_filter.update_mean_window(
            '1961-01-01', '1961-06-28', 6, data_var=0.9)
    tmp2 = client.compute(tmp)
    print(tmp2.result())

    """

    # Plot updated mean vs non updated vs reference.
    date_plot = '1961-05-16'
    updated_mean_slice_future = mean_updated.sel(time=date_plot)
    updated_mean_slice_future = client.compute(updated_mean_slice_future)

    non_updated_mean = subset_mean.anomaly.sel(time=date_plot)
    reference = subset_reference.anomaly.sel(time=date_plot)
    data = subset_instrumental.anomaly.sel(time=date_plot)

    # Clip datasets to common extent.

    # First wait for finish.
    updated_mean = updated_mean_slice_future.result()
    print("Finished updating the mean.")

    updated_mean = updated_mean.where(
                    xr.ufuncs.logical_not(xr.ufuncs.isnan(reference)))
    non_updated_mean = non_updated_mean.where(
                xr.ufuncs.logical_not(xr.ufuncs.isnan(reference)))

    # Plot all with same colorbar.
    xr.set_options(cmap_sequential='RdYlBu_r')
    levels = np.arange(-6, 6, 1)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6),
            subplot_kw={'projection': ccrs.PlateCarree()})
    
    reference.plot(ax=ax1, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
        'label': 'anomaly (reference)'},
            vmin=-6, vmax=6)
    data.plot(ax=ax2, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
            'label': 'anomaly (instrumental)'},
            vmin=-6, vmax=6)
    non_updated_mean.plot(ax=ax3,
            cbar_kwargs={'ticks': levels, 'spacing':
            'proportional', 'label': 'anomaly (non updated)'},
            vmin=-6, vmax=6)
    updated_mean.plot(ax=ax4, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
            'label': 'anomaly (updated)'},
                vmin=-6, vmax=6)
    ax1.coastlines()
    ax2.coastlines()
    ax3.coastlines()
    ax4.coastlines()
        
    plt.savefig("updating_plot_mean.png")

    for i in range(1, 4):
        print("Evalutating Member {}.".format(i))
        start_time = time.time()
        updated_member = members_updated.sel(member_nr=i).sel(time=date_plot)
        updated_member = client.compute(updated_member)

        # First wait for finish.
        updated_member = updated_member.result()
        end_time = time.time()
        print(f"Updating run in {end_time - start_time}")
        non_updated_member = subset_members.sel(member_nr=i).anomaly.sel(time=date_plot)

        # Clip datasets to common extent.
        updated_member = updated_member.where(
                    xr.ufuncs.logical_not(xr.ufuncs.isnan(reference)))
        non_updated_member = non_updated_member.where(
                    xr.ufuncs.logical_not(xr.ufuncs.isnan(reference)))
    
        # Plot all with same colorbar.
        xr.set_options(cmap_sequential='RdYlBu_r')
        levels = np.arange(-6, 6, 1)
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6),
                subplot_kw={'projection': ccrs.PlateCarree()})
        
        reference.plot(ax=ax1, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
                'label': 'anomaly (reference)'},
                vmin=-6, vmax=6)
        data.plot(ax=ax2, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
                'label': 'anomaly (instrumental)'},
                vmin=-6, vmax=6)
        non_updated_member.plot(ax=ax3,
                cbar_kwargs={'ticks': levels, 'spacing':
                'proportional', 'label': 'anomaly (non updated)'},
                vmin=-6, vmax=6)
        updated_member.plot(ax=ax4, cbar_kwargs={'ticks': levels, 'spacing': 'proportional',
                'label': 'anomaly (updated)'},
                vmin=-6, vmax=6)
        ax1.coastlines()
        ax2.coastlines()
        ax3.coastlines()
        ax4.coastlines()
        
        plt.savefig("updating_plot_{}.png".format(i))
        """


if __name__ == "__main__":
    main()
