""" Utility function for the climate package.

"""
import os
import numpy as np
import xarray as xr


def load_dataset(base_folder, TOT_ENSEMBLES_NUMBER):
    """ Load the climate dataset.

    Parameters
    ----------
    base_folder: string
        Path to the root folder for the data. Should contain the Ensembles/ and
        Instrumental/ folder.
    TOT_ENSEMBLES_NUMBER: int
        Only load the first TOT_ENSEMBLES_NUMBER ensemble members.

    """
    ens_mean_folder = os.path.join(base_folder, "Ensembles/Means/")
    ens_mem_folder = os.path.join(base_folder, "Ensembles/Members/")
    instrumental_path = os.path.join(base_folder, "Instrumental/HadCRUT.4.6.0.0.median.nc")
    reference_path = os.path.join(base_folder, "Reference/cru_ts4.05.1901.2020.tmp.dat.nc")

    dataset_instrumental = xr.open_dataset(instrumental_path)
    dataset_reference = xr.open_dataset(reference_path)

    # Rename so dimensions names agree with the other datasets.
    dataset_reference = dataset_reference.rename({'lat': 'latitude', 'lon': 'longitude'})

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

    # --------------
    # POSTPROCESSING
    # --------------

    # Drop the unused variables for clarity.
    dataset_mean = dataset_mean.drop(['cycfreq', 'blocks',
            'lagrangian_tendency_of_air_pressure',
            'northward_wind',
            'eastward_wind', 'geopotential_height',
            'air_pressure_at_sea_level',
            'total precipitation',
            'pressure_level_wind', 'pressure_level_gph'])
    dataset_members = dataset_members.drop(['cycfreq', 'blocks',
            'lagrangian_tendency_of_air_pressure',
            'northward_wind',
            'eastward_wind', 'geopotential_height',
            'air_pressure_at_sea_level',
            'total precipitation',
            'pressure_level_wind', 'pressure_level_gph'])

    dataset_reference = dataset_reference.drop(['stn'])
    dataset_instrumental = dataset_instrumental.drop(['field_status'])

    # Rename.
    dataset_mean = dataset_mean.rename(
            {'air_temperature': 'temperature'})
    dataset_members = dataset_members.rename(
            {'air_temperature': 'temperature'})
    dataset_reference = dataset_reference.rename(
            {'tmp': 'temperature'})
    dataset_instrumental = dataset_instrumental.rename(
            {'temperature_anomaly': 'temperature_anomaly'})

    # Convert longitudes to the standard [-180, 180] format.
    dataset_mean = dataset_mean.assign_coords(
            longitude=(((dataset_mean.longitude + 180) % 360) - 180)).sortby('longitude')
    dataset_members = dataset_members.assign_coords(
            longitude=(((dataset_members.longitude + 180) % 360) - 180)).sortby('longitude')

    # Match time index: except for the instrumental dataset, all are gridded
    # monthly, with index that falls on the middle of the month. Problem is
    # that February (middle is 15 or 16) is treated differently across the
    # datasets.

    # First convert everything to datetime.
    dataset_mean['time'] = dataset_mean.indexes['time'].to_datetimeindex()
    dataset_members['time'] = dataset_members.indexes['time'].to_datetimeindex()

    # Use the index from the mean dataset, matching to the nearest timestamp.
    dataset_reference = dataset_reference.reindex(
            time=dataset_mean.time.values, method='nearest')
    dataset_instrumental = dataset_instrumental.reindex(
            time=dataset_mean.time.values, method='nearest')

    # Regrid the reference to the simulation grid (lower resolution).
    dataset_reference = dataset_reference.interp(latitude=dataset_mean["latitude"],
        longitude=dataset_mean["longitude"])

    # Cast to common dtype.
    dataset_reference['temperature'] = dataset_reference['temperature'].astype(np.float32)

    # Compute anomalies with respect to the 1961-1990 mean.
    mean_ens_mean = dataset_mean.temperature.sel(
            time=slice('1961-01-01', '1990-12-31')).mean(dim='time')
    mean_ens_members = dataset_members.temperature.sel(
            time=slice('1961-01-01', '1990-12-31')).groupby('member_nr').mean(dim='time')
    mean_reference = dataset_reference.temperature.sel(
            time=slice('1961-01-01', '1990-12-31')).mean(dim='time')

    dataset_mean['anomaly'] = dataset_mean.temperature - mean_ens_mean
    dataset_members['anomaly'] = dataset_members.temperature - mean_ens_members
    dataset_reference['anomaly'] = dataset_reference.temperature - mean_reference

    dataset_mean['mean_temp'] = mean_ens_mean
    dataset_members['mean_temp'] = mean_ens_members
    dataset_reference['mean_temp'] = mean_reference

    """
    # Clip datasets to common extent.
    # The reference dataset is only defined on land, hence, we can cut out
    # the sea from the other dataset to lighten the computations.
    ref = dataset_reference.temperature_ref.isel(time=100) # Select one date as baseline.
    dataset_mean['temperature_ens_mean'] = dataset_mean['temperature_ens_mean'].where(
            xr.ufuncs.logical_not(xr.ufuncs.isnan(ref)))
    dataset_members['temperature_ens_member'] = dataset_members['temperature_ens_member'].where(
            xr.ufuncs.logical_not(xr.ufuncs.isnan(ref)))

    # Merge the dataset.
    dataset_merged = xr.combine_by_coords(
            [dataset_mean, dataset_members, dataset_reference],
        coords=['latitude', 'longitude', 'time'], join="inner",
        combine_attrs='drop_conflicts')

    """

    return dataset_mean, dataset_members, dataset_instrumental, dataset_reference
