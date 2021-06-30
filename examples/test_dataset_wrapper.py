import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyresample
import cartopy.crs as ccrs
from climate.utils import load_dataset
from climate.covariance import great_circle_distance
from climate.data_wrapper import Dataset


base_folder = "/home/cedric/PHD/Dev/Climate/Data/"
TOT_ENSEMBLES_NUMBER = 30

dataset_members, dataset_mean, dataset_instrumental = load_dataset(base_folder, TOT_ENSEMBLES_NUMBER)

# Wrap the dataset.
dataset_members = Dataset(dataset_members)

# Test by computing distance from points 1500.
# Find index of the point.
ds = dataset_members.dataset
ilon = list(ds.longitude.values).index(ds.sel(longitude=0.0,
        method='nearest').longitude)
ilat = list(ds.latitude.values).index(ds.sel(latitude=0.0,
        method='nearest').latitude)
ind = np.ravel_multi_index((ilat, ilon), dataset_members.dims)


dists_from_0 = great_circle_distance(dataset_members.coords[ind, :],
        dataset_members.coords).reshape(dataset_members.dims)
dists_data = dataset_members.dataset.air_pressure_at_sea_level.isel(time=0,
        member_nr=1).copy(data=dists_from_0)

axs = dists_data.plot(subplot_kws={'projection': ccrs.PlateCarree()})
axs.axes.coastlines()
plt.show()
