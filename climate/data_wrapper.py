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

    def get_window_vector(self, time_begin, time_end, indexers=None):
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
        stacked_data = self.dataset.anomaly.sel(
                time=slice(time_begin, time_end)).stack(
                        stacked_dim=('time', 'latitude', 'longitude')).chunk(
                                {'stacked_dim': self.chunk_size})
        return stacked_data

    def unstack_window_vector(self, window_vector):
        """ Given a stacked vector (as produced by get_window_vector, but in
        numpy format), unstack it into a dataset of the original dimension.

        """
        # First copy it into a dataset of the current shape.
        result = self.dataset.copy(data=window_vector)
        return result.unstack('stacked_dim')
