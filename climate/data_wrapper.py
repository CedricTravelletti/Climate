import re
import os
import glob
import itertools
import numpy as np
import xarray as xr


class DatasetWrapper():
    def __init__(self, dataset, chunk_size=1000):
        """

        Parameters
        ----------
        dataset: xarray.Dataset

        """
        self.dataset = dataset
        self.dims = (dataset.dims['latitude'], dataset.dims['longitude'])
        self.n_points = self.dims[0] * self.dims[1]
        self.chunk_size = chunk_size

    def _get_coords(self):
        """ Returns a list of the coordinates of all points in the dataset.
        Note that we always work in radians.

        Returns
        -------
        coords: ndarray (self.n_points, 2)

        """
        lons = np.deg2rad(self.dataset.longitude.values.astype(np.float32))
        lats = np.deg2rad(self.dataset.latitude.values.astype(np.float32))
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

        points_list = np.ascontiguousarray(np.vstack([lat_grid.ravel().T,
                lon_grid.ravel().T]).T)
        return points_list

    @property
    def coords(self):
        return self._get_coords()

    @property
    def anomaly(self):
        """ Getter for the anomaly variable of the dataset.
        This avoids having to call self.dataset.anomaly.

        """
        return self.dataset.anomaly

    def _to_1d_array(self, array):
        return array.ravel()

    def _to_2d_array(self, array):
        return array.reshape(self.dims)

    def empty_like(self, times=None):
        """ Returns a dataset with the same spatial dimensions as the current
        one but without any data (does not include a time dimension).

        """
        latitudes = self.dataset['latitude'].values
        longitudes = self.dataset['longitude'].values

        ds_out = xr.Dataset(
            {
                'latitude': (['latitude'], latitudes),
                'longitude': (['longitude'], longitudes)
                })
        if times is not None:
            ds_out = xr.Dataset(
                    {
                            'latitude': (['latitude'], latitudes),
                            'longitude': (['longitude'], longitudes),
                            'time': (['time'], times)
                            
                            })
        return ds_out

    def get_window_vector(self, time_begin, time_end, indexers=None, variable_name='anomaly'):
        """ Given a time window, returns the stacked vector for the data in
        that period.

        Note that only the 'time', 'latitude' and 'longitude' dimensions get
        stacked, while the remaining ones remain untouched.
        This is useful when working with ensembles, since it allows the
        'member_nr' dimension to remain and one can then use it to perform
        ensemble estimation (covariance matrices for example).

        Parameters
        ----------
        time_begin: string
            Beginning of the window.
            Format is like '2021-01-30'.
        time_end: string
            End of window.
        indexers: dict, defaults to None
            If specified, then gets passed to dataset.sel().
            Can be used to do additional subsetting before windowing, e.g.
            select a given member_nr simulation.

        Returns
        -------
        xarray.DataArray
            Data array with the space-time dimensions concatenated.
            The name of the new stacked dimension is 'stacked_dim'.

        """
        # The data is re-chunked after stacking to make sure the subsequent
        # computations fit in memory.
        stacked_data = self.dataset[variable_name].sel(indexers).sel(
                time=slice(time_begin, time_end)).stack(
                        stacked_dim=('time', 'latitude', 'longitude')).chunk(
                                {'stacked_dim': self.chunk_size})
        return stacked_data

    def unstack_window_vector(self, window_vector, variable_name, time='1961-01-16'):
        """ Given a stacked vector (as produced by get_window_vector, but in
        numpy format), unstack it into a dataset of the original dimension.

        """
        # If the data is just an array, unstack manually.
        if isinstance(window_vector, np.ndarray):
            time_begin = time
            time_end = time

            # TODO: Careful here, we have modified the variable_name part 
            # to make it compatible with the highres dataset (where there is no 
            # anomaly variable). Have to check compatibility.
            data_holder = self.dataset[variable_name].sel(time=time).stack(
                    stacked_dim=('latitude', 'longitude')).copy()
            """
            data_holder = self.dataset.anomaly.sel(time=time).stack(
                    stacked_dim=('latitude', 'longitude')).copy()
            """

            # Perform differently if we get a single vector or and ensemble.
            if len(window_vector.shape) < 2:
                data_holder.values = window_vector.reshape(-1)
                unstacked_data = data_holder.unstack('stacked_dim')
                unstacked_data = unstacked_data.rename(variable_name)
                return unstacked_data
            else:
                """
                # Loop over ensemble members.
                unstacked_members = []
                for i in range(window_vector.shape[0]):
                    # Put data in the anomaly variable.
                    data_holder.values = window_vector[i, :].reshape(-1)
                    unstacked_data = data_holder.unstack('stacked_dim')
                    unstacked_data = unstacked_data.rename(variable_name)
                    unstacked_members.append(unstacked_data.copy())
                unstacked_members = xr.concat(unstacked_members, dim='member_nr')
                """
                # TODO: Try non loop version.
                data_holder.values = window_vector
                unstacked_data = data_holder.unstack('stacked_dim')
                unstacked_data = unstacked_data.rename(variable_name)
                return unstacked_data
        else:
            if time is not None:
                time_begin = time
                time_end = time
            else:
                # Find the corresponding time window.
                time_begin = window_vector.time.values.min()
                time_end = window_vector.time.values.max()
    
            # Copy the spatial structure from a dummy dataset.
            data_holder = self.dataset.sel(time=slice(time_begin, time_end))

            # Have to proceed differently if there is only one time.
            if time_begin == time_end:
                data_holder = data_holder[variable_name].stack(
                            stacked_dim=('latitude', 'longitude'))
            else:
                data_holder = data_holder[variable_name].stack(
                            stacked_dim=('time', 'latitude', 'longitude'))

            unstacked_data = data_holder.copy(data=window_vector.reshape(-1)).unstack('stacked_dim')
            # TODO: Not sure if the below is still needed.
            # unstacked_data = unstacked_data.rename({'anomaly': variable_name})
            return unstacked_data


class ZarrDatasetWrapper():
    """ Wrapper for Zarr datasets. Basically only used for dataset members.
    Note that dataset_members contain 'differences' from the dataset mean.

    """
    def __init__(self, dataset_members_zarr, unstacked_data_holder):
        self.dataset_members = dataset_members_zarr
        self.unstacked_data_holder = unstacked_data_holder

        self.spatial_size = (unstacked_data_holder.latitude.shape[0]
                * unstacked_data_holder.longitude.shape[0])
        self.timestamps = unstacked_data_holder.time.values

    @property
    def latitudes(self):
        return self.unstacked_data_holder.latitude.values

    @property
    def longitudes(self):
        return self.unstacked_data_holder.longitude.values

    def get_window_vector(self, time_begin, time_end, member_nr=None, variable_name='difference'):
        time_index_begin = self.get_time_index(time_begin)
        time_index_end = self.get_time_index(time_end)

        if member_nr is not None:
            vector_members = self.dataset_members[variable_name][member_nr,
                time_index_begin*self.spatial_size:(time_index_end + 1)*self.spatial_size]
        else: 
            vector_members = self.dataset_members[variable_name][:,
                time_index_begin*self.spatial_size:(time_index_end + 1)*self.spatial_size]
        return vector_members

    def get_time_index(self, time_string):
        """ Find the index (in the list of timestamps) of a given date.
        Date are in the format '1988-05-16'.

        """
        time_indices = np.where(self.timestamps==np.datetime64(time_string))[0]
        if not len(time_indices) == 1:
            raise ValueError("Date not found.")
        return time_indices[0]

    @property
    def member_nr(self):
        return self.dataset_members.member_nr.values

    def unstack_window_vector(self, window_vector, variable_name, time='1961-01-16'):
        # If the data is just an array, unstack manually.
        if isinstance(window_vector, np.ndarray):
            time_begin = time
            time_end = time
            # Loop over ensemble members.
            unstacked_members = []
            for i in range(window_vector.shape[0]):
                data_holder = self.unstacked_data_holder.anomaly.sel(time=time).stack(
                    stacked_dim=('latitude', 'longitude')).copy()
                # Put data in the anomaly variable.
                data_holder.values = window_vector[i, :].reshape(-1)
                unstacked_data = data_holder.unstack('stacked_dim')
                unstacked_data = unstacked_data.rename(variable_name)
                unstacked_members.append(unstacked_data.copy())
            unstacked_members = xr.concat(unstacked_members, dim='member_nr')
            return unstacked_members
        # Otherwise if we have a DataArray, unpack automatically.
        else:
            # Find the corresponding time window.
            time_begin = window_vector.time.values.min()
            time_end = window_vector.time.values.max()
    
            # Copy the spatial structure from a dummy dataset.
            data_holder = self.unstacked_data_holder.sel(time=slice(time_begin, time_end))

            # Have to proceed differently if there is only one time.
            if time_begin == time_end:
                data_holder = data_holder.anomaly.stack(
                            stacked_dim=('latitude', 'longitude'))
            else:
                data_holder = data_holder.anomaly.stack(
                            stacked_dim=('time', 'latitude', 'longitude'))

            unstacked_data = data_holder.copy(data=window_vector).unstack('stacked_dim')
            unstacked_data = unstacked_data.rename({'anomaly': variable_name})
            return unstacked_data

    def get_region_flat_inds(self, lat_min, lat_max, lon_min, lon_max):
        """ Selects indices (in flattened array) of a square region.

        Parameters
        ----------
        lat_min: float
            Minimal latitude of the square.
        lat_max: float
            Maximal latitude of the square.
        lon_min: float
            Minimal longitude of the square.
        lon_max: float
            Maximal longitude of the square.

        Returns
        -------
        array [int]
            Array of (flat) indices corresponding to the region of interest.

        """
        lat_inds = np.where((self.latitudes >= lat_min) & (self.latitudes <= lat_max))[0]
        lon_inds = np.where((self.longitudes >= lon_min) & (self.longitudes <= lon_max))[0]

        # Check if we have no points.
        if lat_inds.shape[0] <= 0 or lon_inds.shape[0] <= 0:
            return np.array([[]])

        # Produce all combinations of indices.
        inds_prod = np.array(list(itertools.product(lat_inds, lon_inds)))

        # Return flattend indices.
        flat_inds = np.ravel_multi_index((inds_prod[:, 0], inds_prod[:, 1]),
                (self.latitudes.shape[0],
                    self.longitudes.shape[0]))
        return flat_inds


class StationDataset:
    def __init__(self, base_folder):
        station_files = glob.glob(
                os.path.join(base_folder, "CRUTEM/station_files/*/*.nc"))
        self.station_datasets = []
        for i, f in enumerate(station_files):
            print(i)
            self.station_datasets.append(xr.open_dataset(f))

    def get_station_data(self, year, month, day):
        # Parse to standard date format.
        if month == "02":
            day = "15"
        results = []
        for ds in self.station_datasets:
            try:
                tmp1 = ds.tas.sel(time="{}-{}-{}".format(year, month, day))
                # Get climatology for corresponding month.
                month_index = int(month) - 1
                tmp2 = ds.tas_climatology_normal.isel(climatology_normal_time=month_index)
                try: temp, anomaly = tmp1.values[0], tmp2
                except:
                    print(tmp1.values)
                    print(year)
                    print(month)
                    print(day)

                if np.isnan(temp) or np.isnan(anomaly):
                    continue
                results.append([temp, anomaly, tmp1.latitude.values, tmp1.longitude.values])
            except KeyError:
                pass
        return np.array(results)
