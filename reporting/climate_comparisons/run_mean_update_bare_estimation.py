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


base_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
results_folder = "/storage/homefs/ct19x463/Dev/Climate/reporting/climate_comparisons/results/mean_bare_estimation/"

if __name__ == "__main__":
    cluster = SLURMCluster(
        cores=4,                                                                
        memory="24 GB",                                                         
        death_timeout=6000,                                                                              
        walltime="06:00:00",                                                    
        job_extra=['--qos="job_bdw"', '--partition="bdw"']
    )
    
    client = Client(cluster)
    cluster.scale(12)
        
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
    
    windows = generate_6month_windows(1959, 2003)
    for window in windows:
        print("Processing window {}.".format(window))
        start = time.time()
        mean_updated = my_filter.update_window(
                window[0], window[1], n_months=6, data_var=0.9,
                update_members=False)
        mean_updated = my_filter.dataset_mean.unstack_window_vector(
                mean_updated.result())
        mean_updated.to_netcdf(
                os.path.join(results_folder, "mean_updated_begin_{}.nc".format(window[0])))
        end = time.time()
        print("Processed window in {} s.".format((end - start) / 60))
