""" Compute update of the mean over the period 1902-2002 without any 
covariance estimation (bare).

"""
import os
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from climate.utils import load_dataset, generate_6month_windows
from climate.kalman_filter import EnsembleKalmanFilterScatter


import time


# base_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
# results_folder = "/storage/homefs/ct19x463/Dev/Climate/reporting/climate_comparisons/results/mean_bare_estimation/"
base_folder = "/home/cedric/PHD/Dev/Climate/Data/"
results_folder = "/home/cedric/PHD/Dev/Climate/reporting/climate_comparisons/results/mean_bare_estimation/"

if __name__ == "__main__":
    """
    cluster = SLURMCluster(
        cores=4,                                                                
        memory="24 GB",                                                         
        death_timeout=6000,                                                                              
        walltime="06:00:00",                                                    
        job_extra=['--qos="job_bdw"', '--partition="bdw"']
    )
    """
    cluster = LocalCluster()
    
    client = Client(cluster)
    # cluster.scale(12)
        
    # The loading function returns 4 datasets: the ensemble members, the ensemble
    # mean, the instrumental data and the reference dataset.
    # Note that ignore_members=True still loads the members as ZARR files, 
    # so we are fine.
    TOT_ENSEMBLES_NUMBER = 30
    (dataset_mean, _,
            dataset_instrumental, dataset_reference,
            dataset_members_zarr)= load_dataset(
            base_folder, TOT_ENSEMBLES_NUMBER, ignore_members=True)
    print("Loading done.")
    
    my_filter = EnsembleKalmanFilterScatter(dataset_mean, dataset_members_zarr,
            dataset_instrumental, client)
    print("Filter built.")

    # Try stacking and unstacking.
    stacked = my_filter.dataset_members.get_window_vector('1961-01-16', '1961-01-16')
    unstacked = my_filter.dataset_members.unstack_window_vector(
            stacked.sel(member_nr=3))

    # Try plotting to see if stacking / unstacking went correctly.
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.set_global()
    unstacked.isel(time=0).plot.contourf(levels=30, ax=ax, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.show()

    # Select flat indices for Iceland.
    flat_inds_Iceland = my_filter.dataset_members.get_region_flat_inds(
            lat_min=60, lat_max=70, lon_min=-27, lon_max=-8)
