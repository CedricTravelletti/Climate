""" Plots different types of data (ensemble mean, ensemble members, reference)
with a common colorscale.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from climate.utils import load_dataset


base_folder = "/home/cedric/PHD/Dev/Climate/Data/"
TOT_ENSEMBLES_NUMBER = 30

dataset_merged, dataset_instrumental = load_dataset(
        base_folder, TOT_ENSEMBLES_NUMBER)

# Plot with same colorbar.
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
ref = dataset_merged.merged.temperature_ref.isel(time=100)
mean = dataset_merged.temperature_ens_mean.isel(time=100)
member1 = dataset_merged.temperature_ens_member.isel(time=100, member_nr=1)
member2 = dataset_merged.temperature_ens_member.isel(time=100, member_nr=2)


levels = np.arange(-30, 40, 5)
ref.plot(ax=ax1, cbar_kwargs={'ticks': levels, 'spacing': 'proportional'},
        vmin=-30, vmax=35)
mean.plot(ax=ax2, cbar_kwargs={'ticks': levels, 'spacing': 'proportional'},
        vmin=-30, vmax=35)
member1.plot(ax=ax3, cbar_kwargs={'ticks': levels, 'spacing': 'proportional'},
        vmin=-30, vmax=35)
member2.plot(ax=ax4, cbar_kwargs={'ticks': levels, 'spacing': 'proportional'},
        vmin=-30, vmax=35)

plt.show()
