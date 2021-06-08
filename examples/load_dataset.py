""" Example script for loading the whole dataset.

"""
import os
import xarray as xr


ens_mean_folder = "/home/cedric/PHD/Dev/Climate/Data/Ensembles/Means/"
ens_mem_folder = "/home/cedric/PHD/Dev/Climate/Data/Ensembles/Members/"
TOT_ENSEMBLES_NUMBER = 3


dataset_mean = xr.open_mfdataset(ens_mean_folder + '*.nc', concat_dim="time", combine="nested",
                          data_vars='minimal', coords='minimal',
                          compat='override')

# Loop over members folders and merge.
datasets = []
for i in range(1, TOT_ENSEMBLES_NUMBER + 1):
    current_folder = os.path.join(ens_mem_folder, "member_{}/".format(i))
    print(current_folder)
    datasets.append(xr.open_mfdataset(current_folder + '*.nc', concat_dim="time", combine="nested",
                          data_vars='minimal', coords='minimal',
                          compat='override'))

dataset_members = xr.combine_by_coords(datasets)
