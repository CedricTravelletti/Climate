import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from climate.utils import load_dataset, generate_6month_windows
from climate.kalman_filter import EnsembleKalmanFilterScatter


import time


cluster = SLURMCluster(
    cores=4,                                                                
    memory="32 GB",                                                         
    death_timeout=6000,                                                                              
    walltime="02:00:00",                                                    
    job_extra=['--qos="job_epyc2"', '--partition="epyc2"']
)
# cluster = LocalCluster()

client = Client(cluster)
cluster.scale(30)

windows = generate_6month_windows(1950, 1954)
print(windows)


base_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
TOT_ENSEMBLES_NUMBER = 30
    
    # The loading function returns 4 datasets: the ensemble members, the ensemble
    # mean, the instrumental data and the reference dataset.
(dataset_mean, dataset_members,
        dataset_instrumental, dataset_reference,
        dataset_members_zarr)= load_dataset(
        base_folder, TOT_ENSEMBLES_NUMBER, ignore_members=False)
print("Loading done.")

dataset_instrumental = dataset_instrumental.chunk()


my_filter = EnsembleKalmanFilterScatter(dataset_mean, dataset_members_zarr,
        dataset_instrumental, client)
print("Filter built.")

windows = generate_6month_windows(year_begin, year_end)
for window in windows:
    mean_updated, members_updated = my_filter.update_mean_window(
            '1961-01-16', '1961-06-16', 6, data_var=0.9)
print("Function returned.")
# print(mean_updated.result())
# result = client.gather(mean_updated)
print("Finished waiting on mean_updated.")

mean_updated = my_filter.dataset_mean.unstack_window_vector(
        mean_updated.result())
mean_updated.to_netcdf("mean_updated.nc")

for i, x in enumerate(members_updated):
    tmp = x.result()
    print(tmp)
    tmp = my_filter.dataset_members.unstack_window_vector(tmp)
    tmp.to_netcdf("member_updated_{}.nc".format(i))
