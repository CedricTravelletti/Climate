""" Merge the netcdf GLSD files into zarr files. 
Each zarr file will group 100 of the original netcdf files. 

Problematic file that fail merging because of 'compat' should be moved 
from netcdf_merged_cleaned to netcdf_merged_problematic and will be handled manually.

"""
import os
import glob
import xarray as xr
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster


glsd_path = "/home/cedric/PHD/Dev/Climate/Data/Instrumental/GLSD/netcdf_merged_cleaned/"
out_path = "/home/cedric/PHD/Dev/Climate/Data/Instrumental/GLSD/zarr/"
# glsd_path = "/home/cedric/PHD/Dev/Climate/Data/Instrumental/GLSD/small_subset/"


if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)
    
    path_list = glob.glob(glsd_path + "*.nc")
    
    # Create bare dataset using the first one.
    
    """
    merged_ds = merged_ds.expand_dims(["lat", "lon"])
    merged_ds = merged_ds.assign_coords(lat=("lat", merged_ds.lat.values))
    merged_ds = merged_ds.assign_coords(lon=("lon", merged_ds.lon.values))
    """
    
    # Split in files of size 100.
    for k in range(318, 361):
        print(k)
        current_path_list = path_list[(k-1)*100:k*100]
        first_ds_path = current_path_list.pop(0)
        merged_ds = xr.open_dataset(first_ds_path)
        for i, f in enumerate(current_path_list):
            print(i)
            print(f)
            ds = xr.open_dataset(f)
            merged_ds = xr.merge([merged_ds, ds], compat="no_conflicts")
            ds.close()
            del(ds)
        merged_ds.to_zarr(os.path.join(out_path, "glsd_{}.zarr".format(k)))
        merged_ds.close()
