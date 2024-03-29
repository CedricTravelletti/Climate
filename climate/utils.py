""" Utility function for the climate package.

"""
import os
import numpy as np
from datetime import datetime
import xarray as xr
from sklearn.neighbors import BallTree


def load_bare_dataset(base_folder, TOT_ENSEMBLES_NUMBER, ignore_members=False):
    """ Just load the bare datasets, as given by the climate group.

    Parameters
    ----------
    base_folder: string
        Path to the root folder for the data. Should contain the Ensembles/ and
        Instrumental/ folder.
    TOT_ENSEMBLES_NUMBER: int
        Only load the first TOT_ENSEMBLES_NUMBER ensemble members.
    ignore_members: bool (default=False)
        If set to yes, to not load the ensemble members. This can decrease the 
        loading time on systems with slow disks. Note that ensemble_members is 
        still returned, but it is just a copy of ensemble_mean.

    Returns
    -------
    xarray.Dataset: dataset_mean, dataset_members, dataset_instrumental, dataset_reference

    """
    ens_mean_folder = os.path.join(base_folder, "Ensembles/Means/")
    ens_mem_folder = os.path.join(base_folder, "Ensembles/Members/")
    instrumental_path = os.path.join(base_folder, "Instrumental/HadCRUT.4.6.0.0.median.nc")
    # glsd_path = os.path.join(base_folder, "Instrumental/GLSD/netcdf_merged/")
    # glsd_path = os.path.join(base_folder, "Instrumental/GLSD/small_subset/")
    reference_path = os.path.join(base_folder, "Reference/cru_ts4.05.1901.2020.tmp.dat.nc")

    dataset_instrumental = xr.open_dataset(instrumental_path)
    # dataset_instrumental_glsd = xr.open_mfdataset(
    #         glsd_path + '*.nc', concat_dim="time")
    dataset_reference = xr.open_dataset(reference_path)

    # Rename so dimensions names agree with the other datasets.
    dataset_reference = dataset_reference.rename({'lat': 'latitude', 'lon': 'longitude'})

    dataset_mean = xr.open_mfdataset(ens_mean_folder + '*.nc', concat_dim="time", combine="nested",
                          data_vars='minimal', coords='minimal',
                          compat='override')

    # Loop over members folders and merge.
    if ignore_members is False:
        datasets = []
        for i in range(1, int(TOT_ENSEMBLES_NUMBER) + 1):
            current_folder = os.path.join(ens_mem_folder, "member_{}/".format(i))
            print(current_folder)
            datasets.append(
                    xr.open_mfdataset(current_folder + '*.nc', concat_dim="time", combine="nested",
                                  data_vars='minimal', coords='minimal',
                                  compat='override',
                                  chunks={'time': 12}))
        
        dataset_members = xr.combine_by_coords(datasets)
    # Otherwise just return a dummy copy.
    else: dataset_members = dataset_mean.copy(deep=True)

    return dataset_mean, dataset_members, dataset_instrumental, dataset_reference


def _load_dataset(base_folder, TOT_ENSEMBLES_NUMBER, ignore_members=False):
    """ Helper function for dataset loading. 

    Parameters
    ----------
    base_folder: string
        Path to the root folder for the data. Should contain the Ensembles/ and
        Instrumental/ folder.
    TOT_ENSEMBLES_NUMBER: int
        Only load the first TOT_ENSEMBLES_NUMBER ensemble members.
    ignore_members: bool (default=False)
        If set to yes, to not load the ensemble members. This can decrease the 
        loading time on systems with slow disks. Note that ensemble_members is 
        still returned, but it is just a copy of ensemble_mean.

    Returns
    -------
    xarray.Dataset: dataset_mean, dataset_members, dataset_instrumental, dataset_reference

    """
    # Start by just loading the bare datasets.
    dataset_mean, dataset_members, dataset_instrumental, dataset_reference = load_bare_dataset(base_folder, TOT_ENSEMBLES_NUMBER, ignore_members)

    # --------------
    # POSTPROCESSING
    # --------------

    # Drop the unused variables for clarity.
    dataset_mean = dataset_mean.drop(['cycfreq', 'blocks',
            'lagrangian_tendency_of_air_pressure',
            'northward_wind',
            'eastward_wind', 'geopotential_height',
            # 'air_pressure_at_sea_level',
            'total precipitation',
            'pressure_level_wind', 'pressure_level_gph'])
    dataset_members = dataset_members.drop(['cycfreq', 'blocks',
            'lagrangian_tendency_of_air_pressure',
            'northward_wind',
            'eastward_wind', 'geopotential_height',
            # 'air_pressure_at_sea_level',
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
            {'temperature_anomaly': 'anomaly'})

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
    # Subset to valid times. This is because np.datetime only allows dates back to 1687,
    # but on the other hand cftime does not support the full set of pandas slicing features.
    dataset_mean = dataset_mean.sel(time=slice("1800-01", "2001-12"))
    dataset_members = dataset_members.sel(time=slice("1800-01", "2001-12"))

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

    # Compute difference wrt mean.
    dataset_members['difference'] = dataset_members.temperature - dataset_mean.temperature

    # Compute 71 year runnig mean anomaly for the reference dataset and 
    # the ensemble mean.
    dataset_mean = rolling_mean(dataset_mean, yr_delta_plus=35, yr_delta_minus=-35)
    dataset_reference = rolling_mean(dataset_reference, yr_delta_plus=35, yr_delta_minus=-35)

    # Rechunk the anomaly to have a big enough chunk size.
    dataset_members['difference'] = dataset_members.difference.chunk({'time': 480})

    # Put the instrumental dataset into Dask format.
    dataset_instrumental['anomaly'] = dataset_instrumental['anomaly'].chunk()

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

    return (dataset_mean, dataset_members, dataset_instrumental,
        # dataset_instrumental_glsd,
        dataset_reference)

def match_vectors_indices(base_vector, vector_to_match):
    """" Given two stacked datasets (vectors), for each element in the dataset_tomatch,
    find the index of the element in the base dataset that is closest.

    Note that the base dataset should contain only one element at each spatial locaiton, 
    so that the matched index is unique.

    Parameters
    ----------
    base_vector: xarray.DataArray
        Stacked dataset.
    vector_to_match: xarray.DataArray
        Stacked dataset.

    Returns
    -------
    Array[int] (vector_to_match.shape[0])
        Indices in the base dataset of closest element for each 
        element of the dataset_tomatch.

    """
    # Convert to radians.
    lat_rad = np.deg2rad(base_vector.latitude.values.astype(np.float32))
    lon_rad = np.deg2rad(base_vector.longitude.values.astype(np.float32))

    # Build a ball tree to make nearest neighbor queries faster.
    ball = BallTree(np.vstack([lat_rad, lon_rad]).T, metric='haversine')

    # Define grid to be matched.
    lon_tomatch = np.deg2rad(vector_to_match.longitude.values.astype(np.float32))
    lat_tomatch = np.deg2rad(vector_to_match.latitude.values.astype(np.float32))
    coarse_grid_list = np.vstack([lat_tomatch.T, lon_tomatch.T]).T

    distances, index_array_1d = ball.query(coarse_grid_list, k=1)
    
    # Convert back to kilometers.
    distances_km = 6371 * distances
    # Sanity check.
    print("Maximal distance to matched point: {} km.".format(np.max(distances_km)))

    return index_array_1d.squeeze()

def match_datasets(base_dataset, dataset_tomatch):
    # Define original lat-lon grid
    # Creates new columns converting coordinate degrees to radians.
    lon_rad = np.deg2rad(base_dataset.longitude.values.astype(np.float32))
    lat_rad = np.deg2rad(base_dataset.latitude.values.astype(np.float32))
    lat_grid, lon_grid = np.meshgrid(lat_rad, lon_rad, indexing='ij')
    
    # Define grid to be matched.
    lon_tomatch = np.deg2rad(dataset_tomatch.longitude.values.astype(np.float32))
    lat_tomatch = np.deg2rad(dataset_tomatch.latitude.values.astype(np.float32))
    lat_tomatch_grid, lon_tomatch_grid = np.meshgrid(lat_tomatch, lon_tomatch,
            indexing='ij')
    
    # Put everything in two stacked lists (format required by BallTree).
    coarse_grid_list = np.vstack([lat_tomatch_grid.ravel().T, lon_tomatch_grid.ravel().T]).T
    
    ball = BallTree(np.vstack([lat_grid.ravel().T, lon_grid.ravel().T]).T, metric='haversine')
    
    distances, index_array_1d = ball.query(coarse_grid_list, k=1)
    
    # Convert back to kilometers.
    distances_km = 6371 * distances

    # Sanity check.
    print("Maximal distance to matched point: {} km.".format(np.max(distances_km)))
    
    # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute
    # the 2D grid indices:
    index_array_2d = np.hstack(np.unravel_index(index_array_1d, lon_grid.shape))

    return index_array_1d, index_array_2d


def build_base_forward(model_dataset, data_dataset):
    """ Build the forward operator mapping a model dataset to a data dataset.
    This builds the generic forward matrix that maps points in the model space
    to points in the data space, considering both as static, uniform grids. 
    When used in practice one should remove lines of the matrix that correspond
    to NaN values in the data space.

    """
    index_array_1d, index_array_2d = match_datasets(model_dataset, data_dataset)

    model_lon_dim = model_dataset.dims['longitude']
    model_lat_dim = model_dataset.dims['latitude']
    data_lon_dim = data_dataset.dims['longitude']
    data_lat_dim = data_dataset.dims['latitude']

    G = np.zeros((data_lon_dim * data_lat_dim, model_lon_dim * model_lat_dim),
            np.float32)

    # Set corresponding cells to 1 (ugly implementation, could be better).
    for data_ind, model_ind in enumerate(index_array_1d):
        G[data_ind, model_ind] = 1.0

    return G

def load_dataset(base_folder, TOT_ENSEMBLES_NUMBER, ignore_members=False):
    """ Load the whole climate dataset. Note the the ensemble members 
    are also returned in ZARR format, since this allows for faster access.

    The dataset consists of ensemble mean, ensemble members, reference data 
    and instrumental data. Those are returned as 4 different xarray datasets.
    Note that anomalies w.r.t. the 1961-01-01, 1990-12-31 base period are returned.

    Parameters
    ----------
    base_folder: string
        Path to the root folder for the data. Should contain the Ensembles/ and
        Instrumental/ folder.
    TOT_ENSEMBLES_NUMBER: int
        Only load the first TOT_ENSEMBLES_NUMBER ensemble members.
    ignore_members: bool (default=False)
        If set to yes, to not load the ensemble members. This can decrease the 
        loading time on systems with slow disks. Note that ensemble_members is 
        still returned, but it is just a copy of ensemble_mean.
        Note also that ensemble_members_zarr is still returned.

    Returns
    -------
    xarray.Dataset: dataset_mean, dataset_members, dataset_instrumental, dataset_reference, 
        dataset_members_zarr

    """
    ens_mean_path = os.path.join(base_folder, "ensemble_mean.zarr")
    ens_members_path = os.path.join(base_folder, "ensemble_members.zarr")

    (dataset_mean, dataset_members, dataset_instrumental,
            # dataset_instrumental_glsd,
            dataset_reference) = _load_dataset(
            base_folder, TOT_ENSEMBLES_NUMBER, ignore_members)
    dataset_members_zarr = xr.open_zarr(ens_members_path)

    return (dataset_mean, dataset_members,
            dataset_instrumental,
            # dataset_instrumental_glsd,
            dataset_reference,
            dataset_members_zarr)

def add_year(dt64, yr_delta):
    """ Given a np.datetime64 add some number of years to it.

    """
    # convert to timestamp:
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    
    # standard utctime from timestamp
    dt = datetime.utcfromtimestamp(ts)
    
    # update year
    dt = dt.replace(year=dt.year+yr_delta)
    
    # convert back to numpy.datetime64:
    dt64 = np.datetime64(dt)

    return dt64

def get_year(dt64):
    """ Get the year from a given numpy.datetime64 object.

    """
    # convert to timestamp:
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    
    # standard utctime from timestamp
    dt = datetime.utcfromtimestamp(ts)

    return dt.year

def get_year_window(dt64):
    """ Given a np.datetime64, return the first and last day of the year 
    (to have a window).

    """
    # convert to timestamp:
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    
    # standard utctime from timestamp
    dt = datetime.utcfromtimestamp(ts)
    
    dt_begin = dt.replace(month=1, day=1)
    dt_end = dt.replace(month=12, day=31)
    
    # convert back to numpy.datetime64:
    dt64_begin = np.datetime64(dt_begin)
    dt64_end = np.datetime64(dt_end)

    return dt64_begin, dt64_end

def rolling_mean(dataset, yr_delta_plus, yr_delta_minus):
    """ Compute rolling mean with some given window.

    """
    # First need to duplicate the variable so it already has a 
    # time component. 
    dataset['anomaly'] = dataset['temperature'].copy()

    # Find the first and last year in the dataset.
    yr_min = get_year(dataset.time.values.min())
    yr_max = get_year(dataset.time.values.max())

    anomalies = []
    # Do computation for each year in the dataset.
    for i in range(yr_min, yr_max+1):
        yr_current = np.datetime64("{}-01-01".format(i))
        # First compute (for each month) the average in a 71 year window 
        # around the current year.
        time_begin = add_year(yr_current, yr_delta_minus)
        time_end = add_year(yr_current, yr_delta_plus)
        monthly_avg_mean = dataset.temperature.sel(
                time=slice(time_begin, time_end)).groupby('time.month').mean(dim='time')
    
        # Now subtract from current year.
        yr_begin, yr_end = get_year_window(yr_current)
        anomalies.append((
                dataset.temperature.sel(time=slice(yr_begin, yr_end)).groupby('time.month')
                - monthly_avg_mean).drop_vars('month'))
    dataset['anomaly'] = xr.concat(anomalies, dim='time')
    return dataset

def generate_6month_windows(year_begin, year_end):
    """ Generate strings defining 6 month window over a given 
    span of years. 
    This is to be used to provide parameters for the kalman filter update method.

    Parameters
    ----------
    year_begin: int
        Year at which to start.
    year_end: int
        Year at which to end (included).

    Returns
    -------
    List[(string, string)]
        List of window begin and end in string format. 
        A window is defined by a tuple of the form (¹961-01-16', '1961-06-16').

    """
    windows = []
    for year in range(year_begin, year_end + 1):
        for month in [('01', '06'), ('07', '12')]:
            window_begin = '{}-{}-16'.format(year, month[0])
            window_end = '{}-{}-16'.format(year, month[1])
            windows.append((window_begin, window_end))
    return windows

def generate_monthly_dates(year_begin, year_end):
    """ Generate strings corresponding to monthly dates (month middle) 
    for a given span of years.

    Parameters
    ----------
    year_begin: int
        Year at which to start.
    year_end: int
        Year at which to end (included).

    Returns
    -------
    List[string]
        List of dates string format. 
        A date is of the form '1961-06-16'.

    """
    dates = []
    for year in range(year_begin, year_end + 1):
        for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            date = '{}-{}-16'.format(year, month)
            dates.append(date)
    return dates


def load_bare_highres_dataset(base_folder, TOT_ENSEMBLES_NUMBER=5, ignore_members=False):
    """ Just load the bare datasets, as given by the climate group.

    Parameters
    ----------
    base_folder: string
        Path to the root folder for the data. Should contain the Ensembles/ and
        Instrumental/ folder.
    TOT_ENSEMBLES_NUMBER: int
        Only load the first TOT_ENSEMBLES_NUMBER ensemble members.
    ignore_members: bool (default=False)
        If set to yes, to not load the ensemble members. This can decrease the 
        loading time on systems with slow disks. Note that ensemble_members is 
        still returned, but it is just a copy of ensemble_mean.

    Returns
    -------
    xarray.Dataset: dataset_mean, dataset_members, dataset_instrumental, dataset_reference

    """
    highres_folder = os.path.join(base_folder, "high_res/simulations/")
    mean_path = os.path.join(highres_folder, "ModE-Simhires_ensmean_temp2_abs_1420-2009.nc")
    dataset_mean_highres = xr.open_dataset(mean_path)

    # Loop over members folders and merge.
    if ignore_members is False:
        datasets = []
        for i in range(1, int(TOT_ENSEMBLES_NUMBER) + 1):
            member_path = os.path.join(highres_folder, "ModE-Simhires_m00{}_temp2_abs_1420-2009.nc".format(i))
            datasets.append(xr.open_dataset(member_path).expand_dims({'member_nr': [i]}))
        
        dataset_members_highres = xr.combine_by_coords(datasets, combine_attrs='drop')
    # Otherwise just return a dummy copy.
    else: dataset_members_highres = dataset_mean_highres.copy(deep=True)

    return dataset_mean_highres, dataset_members_highres

def load_and_preprocess_highres_dataset(base_folder, TOT_ENSEMBLES_NUMBER=5, ignore_members=False):
    # First load.
    dataset_mean_highres, dataset_members_highres = load_bare_highres_dataset(base_folder, TOT_ENSEMBLES_NUMBER, ignore_members)

    # Rename so dimensions names agree with the other datasets.
    dataset_mean_highres = dataset_mean_highres.rename({'lat': 'latitude', 'lon': 'longitude'})
    dataset_members_highres = dataset_members_highres.rename({'lat': 'latitude', 'lon': 'longitude'})
    dataset_mean_highres = dataset_mean_highres.rename(
            {'temp2': 'temperature'})
    dataset_members_highres = dataset_members_highres.rename(
            {'temp2': 'temperature'})

    # First convert everything to datetime.
    # Subset to valid times. This is because np.datetime only allows dates back to 1687,
    # but on the other hand cftime does not support the full set of pandas slicing features.
    dataset_mean_highres = dataset_mean_highres.sel(time=slice("1800-01", "2001-12"))
    dataset_members_highres = dataset_members_highres.sel(time=slice("1800-01", "2001-12"))

    dataset_mean_highres['time'] = dataset_mean_highres.indexes['time'].to_datetimeindex()
    dataset_members_highres['time'] = dataset_members_highres.indexes['time'].to_datetimeindex()

    # Match time index: except for the instrumental dataset, all are gridded
    # monthly, with index that falls on the middle of the month. Problem is
    # that February (middle is 15 or 16) is treated differently across the
    # datasets.

    # First get the low resolution dataset.
    (dataset_mean_lowres, _, _, _, _)= load_dataset(base_folder, TOT_ENSEMBLES_NUMBER=1, ignore_members=True)

    # Use the index from the low resolution mean dataset, matching to the nearest timestamp.
    dataset_mean_highres = dataset_mean_highres.reindex(
            time=dataset_mean_lowres.time.values, method='nearest')
    dataset_members_highres = dataset_members_highres.reindex(
            time=dataset_mean_lowres.time.values, method='nearest')

    return dataset_mean_highres, dataset_members_highres
