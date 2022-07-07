""" Merge the GLSD dataset once it is already in ZARR format.

"""
import os
import glob
import matplotlib.pyplot as plt
import xarray as xr
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster


# merged_glsd_path = "/home/cedric/PHD/Dev/Climate/Data/Instrumental/GLSD/zarr/"
merged_glsd_path = "/storage/homefs/ct19x463/Dev/Climate/Data/Instrumental/GLSD/zarr/"
out_path = "/storage/homefs/ct19x463/Dev/Climate/Data/Instrumental/GLSD/"


if __name__ == "__main__":
    cluster = SLURMCluster(
        cores=4,                                                                
        memory="32 GB",                                                         
        death_timeout=6000,
        walltime="02:00:00",
        job_extra=['--qos="job_epyc2"', '--partition="epyc2"']
        )
    client = Client(cluster)
    cluster.scale(20)
    
    path_list = glob.glob(merged_glsd_path + "*.zarr")
    merged_ds = xr.open_zarr(path_list[0])
    print(path_list[0])
    for i, f in enumerate(path_list[1:]):
        print(i)
        print(f)
        ds = xr.open_zarr(f)
        merged_ds = xr.merge([merged_ds, ds], compat='no_conflicts')
        ds.close()
        del(ds)
        if i % 5 == 0:
            # Rechunk befor saving.
            merged_ds['surface_average_temperature'] = merged_ds.surface_average_temperature.chunk("auto")
            merged_ds.to_zarr(os.path.join(out_path, "glsd_full_merge.zarr"), mode="w")
    merged_ds.to_zarr(os.path.join(out_path, "glsd_full_merge.zarr"), mode="w")
    merged_ds.close()
