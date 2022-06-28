""" Merge the netcdf GLSD files into zarr files. 
Each zarr file will group 100 of the original netcdf files. 

Problematic file that fail merging because of 'compat' should be moved 
from netcdf_merged_cleaned to netcdf_merged_problematic and will be handled manually.

"""
import os
import glob
import time
import xarray as xr
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster


# merged_glsd_path = "/home/cedric/PHD/Dev/Climate/Data/Instrumental/GLSD/zarr/"
glsd_path = "/storage/homefs/ct19x463/Dev/Climate/Data/Instrumental/GLSD/zarr/"
out_path = "/storage/homefs/ct19x463/Dev/Climate/Data/Instrumental/GLSD/merged_stage2/"


if __name__ == "__main__":
    cluster = SLURMCluster(
        cores=4,                                                                
        memory="32 GB",                                                         
        death_timeout=6000,
        walltime="02:00:00",
        job_extra=['--qos="job_epyc2"', '--partition="epyc2"']
        )
    client = Client(cluster)
    cluster.scale(25)

    path_list = glob.glob(glsd_path + "*.zarr")
    # path_list = path_list[:5]
    datasets = []
    for i, path in enumerate(path_list):
        print(i)
        datasets.append(xr.open_zarr(path, chunks="auto"))

    computed_datasets = []
    for ds in datasets:
        computed_datasets.append(client.persist(ds))

    # Loop in chunks of 5.
    chunk_size = 5
    for i in range(175, len(computed_datasets), chunk_size):
        print(i)
        chunk = computed_datasets[i:i+chunk_size]
        ds_merged = chunk.pop(0)
        for ds in chunk:
            ds_merged = ds_merged.combine_first(ds)
        ds_merged = client.compute(ds_merged)
        ds_merged.result().to_zarr(os.path.join(out_path, "merged_{}.zarr".format(i)))
