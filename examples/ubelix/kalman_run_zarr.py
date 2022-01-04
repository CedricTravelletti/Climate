import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from climate.utils import load_zarr_dataset
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
cluster.scale(20)


base_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
TOT_ENSEMBLES_NUMBER = 30
    
    # The loading function returns 4 datasets: the ensemble members, the ensemble
    # mean, the instrumental data and the reference dataset.
(dataset_mean, dataset_members,
        dataset_instrumental, dataset_reference,
        dataset_members_zarr)= load_zarr_dataset(
        base_folder, TOT_ENSEMBLES_NUMBER, ignore_members=True)
print("Loading done.")

dataset_instrumental = dataset_instrumental.chunk()


my_filter = EnsembleKalmanFilterScatter(dataset_mean, dataset_members_zarr,
        dataset_instrumental, client)
print("Filter built.")
mean_updated, members_updated = my_filter.update_mean_window(
        '1961-01-16', '1961-06-16', 6, data_var=0.9)
print("Function returned.")
print(mean_updated.result())
# result = client.gather(mean_updated)
print("Finished waiting on mean_updated.")
mean_updated.result().to_netcdf("mean_updated.nc")
for i, x in enumerate(members_updated):
    tmp = x.result()
    tmp = my_filter.dataset_member.unstack_window_vector(tmp)
    tmp.to_netcdf("member_updated_{}.nc".format(i))
