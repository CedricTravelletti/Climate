import numpy as np
from climate.data_wrapper import DatasetWrapper, ZarrDatasetWrapper
from climate.utils import build_base_forward
from scipy.linalg import block_diag
import xarray as xr
from dask.array import matmul, eye, transpose
from dask.array.linalg import inv, cholesky
import dask.array as da
import dask
from climate.dask_covariance import cov


# Set default chunk size so that automatic rechunking works.
dask.config.set({"array.chunk-size": "800MiB"})


class EnsembleKalmanFilter():
    """ Implementation of the ensemble Kalman filter.

    """
    def __init__(self, dataset_mean, dataset_members_zarr, dataset_instrumental,
            dask_client,
            chunk_size=1000):
        self.dask_client = dask_client
        self.chunk_size = chunk_size

        # First load everything into distributed memory.
        dataset_mean = self.dask_client.persist(dataset_mean)
        dataset_members_zarr = self.dask_client.persist(dataset_members_zarr)
        dataset_instrumental = self.dask_client.persist(dataset_instrumental)

        # Wrap the datasets.
        self.dataset_mean = DatasetWrapper(dataset_mean, chunk_size)
        self.dataset_members = ZarrDatasetWrapper(dataset_members_zarr,
                dataset_mean.copy(deep=True))
        self.dataset_instrumental = DatasetWrapper(dataset_instrumental, chunk_size)

        # Get the base forward (the matrix making the translation between the
        # instrumental dataset and the other ones).
        self.G_base = build_base_forward(dataset_mean,
                dataset_instrumental)
    
    def get_forward_for_window(self, time_begin, time_end, n_months):
        """ Computes the forward operator for a given time window.

        The forward is a matrix acting on the stacked model vector to produce
        the stacked observations.

        Note that there is a subtlety due to the stacking: the base forward has
        ones at indices that correspond to a given model cell. When we worked
        with stacked vectors, these ones should be at the index of the
        corresponding model cell in the STACKED vector.
        This can be solved by stacking the forwards for each month in a block
        diagonal fashion (think about it).

        Parameters
        ----------
        time_begin: string
            Beginning of the window.
            Format is like '2021-01-30'.
        time_end: string
            End of window.
        n_months: int
            Number of months in the window.

        Returns
        -------
        ndarray (n_data, n_model * n_months)
            Stacked forward operator.

        """
        # First subset to the window.
        dataset_instrumental = self.dataset_instrumental.anomaly.sel(
                time=slice(time_begin, time_end))

        # Loop over the months in the window and stack the operators.
        # The operator is builts by arranging the monthly operators in a block
        # diagonal matrix.
        monthly_data = self.dataset_instrumental._to_1d_array(
                    dataset_instrumental.isel(time=0).values)

        # We use the base G which is basically a translation between the 
        # two grids and only activate it at places where we have data.
        G_nonan = self.G_base[np.logical_not(np.isnan(monthly_data)), :]

        for i in range(1, n_months):
            monthly_data = self.dataset_instrumental._to_1d_array(
                    dataset_instrumental.isel(time=i).values)
            G_tmp = self.G_base[np.logical_not(np.isnan(monthly_data)), :]

            # Append block diagonally.
            G_nonan = block_diag(G_nonan, G_tmp)

        return G_nonan

    def get_ensemble_covariance(self, time_begin, time_end):
        """ Get the (lazy) covariance matrix for a given time window. 
        Times are given by the index in the list of the timestamps of the dataset.

        """
        # Get the stacked window vectors for each ensemble member.
        vector_members = self.dataset_members.get_window_vector(time_begin, time_end)

        # Scatter everywhere so easier to compute covariance matrix.
        vector_members = self.dask_client.scatter(vector_members, broadcast=True)

        # Compute covariance matrix acrosss the different members.
        # Note that the below is a lazy operation (dask).
        cov_matrix = cov(vector_members, rowvar=False)
        return cov_matrix

    def update_mean_window(self, time_begin, time_end, n_months, data_var):
        """ Ensemble Kalman filter updating over an assimilation window.

        Parameters
        ----------
        time_begin: str
            Begin (included) of updating window. Format is '1961-12-31'.
            Note that this does not need to be a date that is present in the dataset.
            The datasets used here contain the 16th of every month, so valid date is 
            '1961-01-16'.
        time_end: str
            End of the updating window (included). Same format as above.
        n_months: int
            Number of datapoints contained in the window.
            The datasets used here contain datapoins each 16th of the month, so if 
            begin is '1961-01-01' and end is '1961-06-28' then the number of months (datapoints) will be 6.
            Will be automated in the future.
        data_var: float
            Variance of the observational noise.

        Returns
        -------
        mean_updated: xarray.Dataset
            Dataset with updated mean. Note that the data format is lazy. One should call 
            the corresponding .compute() method to trigger computing.
        members_updated: xarray.Dataset
            Dataset with updated members. Note that the data format is lazy. One should call 
            the corresponding .compute() method to trigger computing.

        """
        # First get the mean vector and data vector (stacked for the window).
        vector_mean = self.dataset_mean.get_window_vector(time_begin, time_end)
        vector_data = self.dataset_instrumental.get_window_vector(time_begin, time_end)

        # Scatter to all client (data is small).
        vector_mean = self.dask_client.scatter(vector_mean, broadcast=True).result()
        vector_data = self.dask_client.scatter(vector_data, broadcast=True).result()

        # Get rid of the Nans.
        vector_data = vector_data[np.logical_not(np.isnan(vector_data))]

        # Get covariance matrix and forward.
        G = self.get_forward_for_window(time_begin, time_end, n_months)
        G = da.from_array(G)

        cov = self.get_ensemble_covariance(time_begin, time_end)

        # Persist the covariance matrix into distributed memory.
        cov = self.dask_client.persist(cov)

        # Trigger computation of covariance pushforward and persist it on the 
        # cluster, since we will do several operations with it.
        cov_pushfwd = self.dask_client.persist(matmul(cov, transpose(G)))
        data_cov = data_var * eye(vector_data.shape[0])
        data_cov = data_cov.rechunk(data_cov.shape[0], data_cov.shape[1])

        # Define the graph for computing the updated mean vector.
        to_invert = (matmul(
                        G,
                        cov_pushfwd)
                    + data_cov)
        to_invert = to_invert.rechunk(to_invert.shape[0], to_invert.shape[1])
        print(to_invert)
        sqrt = cholesky(to_invert, lower=True)

        kalman_gain = matmul(
                cov_pushfwd,
                inv(to_invert))
        kalman_gain_tilde = matmul(
                cov_pushfwd,
                matmul(
                    transpose(inv(sqrt)),
                    inv(sqrt + cholesky(data_cov, lower=True))
                    )
                )

        prior_misfit = vector_data - matmul(G, vector_mean)
        vector_mean_updated = (
                vector_mean +
                matmul(kalman_gain, prior_misfit)
                )
        vector_mean_updated = self.dask_client.persist(vector_mean_updated)

        import time
        time.sleep(7*60)

        # Loop over members.
        vector_members_updated = []
        # for i in self.dataset_members.member_nr:
        for i in range(1, 3):
            print(i)
            vector_member = self.dataset_members.get_window_vector(
                    time_begin, time_end, member_nr=i)
            vector_member_updated = (
                    vector_member +
                    matmul(kalman_gain_tilde, matmul(G, vector_member))
                    )
            print(vector_member_updated)
            vector_member_updated = self.dask_client.persist(vector_member_updated)

            # member_updated = vector_member_updated.unstack('stacked_dim')
            vector_members_updated.append(vector_member_updated)

        time.sleep(7*60)
        return vector_mean_updated, vector_members_updated
        """

        # Unstack the vector to go back to the grid format.
        mean_updated = self.dask_client.compute(vector_mean_updated.unstack('stacked_dim')).result()

        # Put into single dataset.
        members_updated = xr.concat(vector_members_updated, dim='member_nr')

        return mean_updated, members_updated
        """


class DryRunEnsembleKalmanFilter(EnsembleKalmanFilter):
    """ Mock variant for debug and diagnostics.

    """
    def update_mean_window(self, time_begin, time_end, n_months, data_var):
        # First get the mean vector and data vector (stacked for the window).
        vector_mean = self.dataset_mean.get_window_vector(time_begin, time_end)
        vector_data = self.dataset_instrumental.get_window_vector(time_begin, time_end)

        # Get rid of the Nans.
        vector_data = vector_data[np.logical_not(np.isnan(vector_data))]

        # Get covariance matrix and forward.
        G = self.get_forward_for_window(time_begin, time_end, n_months)
        G = da.from_array(G)
        cov = self.get_ensemble_covariance(time_begin, time_end)

        data_cov = data_var * eye(vector_data.shape[0], chunks=self.chunk_size).rechunk()

        # Define the graph for computing the updated mean vector.
        to_invert = (matmul(
                        G,
                        matmul(cov, transpose(G)))
                    + data_cov).rechunk()
        return(to_invert)

class EnsembleKalmanFilterScatter():
    """ Implementation of the ensemble Kalman filter.

    """
    def __init__(self, dataset_mean, dataset_members_zarr, dataset_instrumental,
            dask_client,
            chunk_size=1000):
        self.dask_client = dask_client
        self.chunk_size = chunk_size

        # Wrap the datasets.
        self.dataset_mean = DatasetWrapper(dataset_mean, chunk_size)
        self.dataset_members = ZarrDatasetWrapper(dataset_members_zarr,
                dataset_mean.copy(deep=True))
        self.dataset_instrumental = DatasetWrapper(dataset_instrumental, chunk_size)

        # Get the base forward (the matrix making the translation between the
        # instrumental dataset and the other ones).
        self.G_base = build_base_forward(dataset_mean,
                dataset_instrumental)
    
    def get_forward_for_window(self, time_begin, time_end, n_months):
        """ Computes the forward operator for a given time window.

        The forward is a matrix acting on the stacked model vector to produce
        the stacked observations.

        Note that there is a subtlety due to the stacking: the base forward has
        ones at indices that correspond to a given model cell. When we worked
        with stacked vectors, these ones should be at the index of the
        corresponding model cell in the STACKED vector.
        This can be solved by stacking the forwards for each month in a block
        diagonal fashion (think about it).

        Parameters
        ----------
        time_begin: string
            Beginning of the window.
            Format is like '2021-01-30'.
        time_end: string
            End of window.
        n_months: int
            Number of months in the window.

        Returns
        -------
        ndarray (n_data, n_model * n_months)
            Stacked forward operator.

        """
        # First subset to the window.
        dataset_instrumental = self.dataset_instrumental.anomaly.sel(
                time=slice(time_begin, time_end))

        # Loop over the months in the window and stack the operators.
        # The operator is builts by arranging the monthly operators in a block
        # diagonal matrix.
        monthly_data = self.dataset_instrumental._to_1d_array(
                    dataset_instrumental.isel(time=0).values)

        # We use the base G which is basically a translation between the 
        # two grids and only activate it at places where we have data.
        G_nonan = self.G_base[np.logical_not(np.isnan(monthly_data)), :]

        for i in range(1, n_months):
            monthly_data = self.dataset_instrumental._to_1d_array(
                    dataset_instrumental.isel(time=i).values)
            G_tmp = self.G_base[np.logical_not(np.isnan(monthly_data)), :]

            # Append block diagonally.
            G_nonan = block_diag(G_nonan, G_tmp)

        return G_nonan

    def get_ensemble_covariance(self, time_begin, time_end):
        """ Get the (lazy) covariance matrix for a given time window. 
        Times are given by the index in the list of the timestamps of the dataset.

        """
        # Get the stacked window vectors for each ensemble member.
        vector_members = self.dataset_members.get_window_vector(time_begin, time_end)

        # Scatter everywhere so easier to compute covariance matrix.
        vector_members = self.dask_client.scatter(vector_members, broadcast=True)

        # Compute covariance matrix acrosss the different members.
        # Note that the below is a lazy operation (dask).
        cov_matrix = cov(vector_members, rowvar=False)
        return cov_matrix

    def update_mean_window(self, time_begin, time_end, n_months, data_var):
        """ Ensemble Kalman filter updating over an assimilation window.

        Parameters
        ----------
        time_begin: str
            Begin (included) of updating window. Format is '1961-12-31'.
            Note that this does not need to be a date that is present in the dataset.
            The datasets used here contain the 16th of every month, so valid date is 
            '1961-01-16'.
        time_end: str
            End of the updating window (included). Same format as above.
        n_months: int
            Number of datapoints contained in the window.
            The datasets used here contain datapoins each 16th of the month, so if 
            begin is '1961-01-01' and end is '1961-06-28' then the number of months (datapoints) will be 6.
            Will be automated in the future.
        data_var: float
            Variance of the observational noise.

        Returns
        -------
        mean_updated: xarray.Dataset
            Dataset with updated mean. Note that the data format is lazy. One should call 
            the corresponding .compute() method to trigger computing.
        members_updated: xarray.Dataset
            Dataset with updated members. Note that the data format is lazy. One should call 
            the corresponding .compute() method to trigger computing.

        """
        # First get the mean vector and data vector (stacked for the window).
        vector_mean = self.dataset_mean.get_window_vector(time_begin, time_end)
        vector_data = self.dataset_instrumental.get_window_vector(time_begin, time_end)

        # Get rid of the Nans.
        vector_data = vector_data[np.logical_not(np.isnan(vector_data))]

        # Scatter to all client (data is small).
        # We get back futures.
        vector_mean_fut = self.dask_client.scatter(vector_mean, broadcast=True)
        vector_data_fut = self.dask_client.scatter(vector_data, broadcast=True)


        # Get covariance matrix and forward.
        G = self.get_forward_for_window(time_begin, time_end, n_months)
        G = da.from_array(G)
        G_fut = self.dask_client.scatter(G, broadcast=True)

        cov = self.get_ensemble_covariance(time_begin, time_end)

        # Persist the covariance matrix into distributed memory.
        cov = self.dask_client.persist(cov)
        import time
        print("Sleeping")
        time.sleep(10*60)

        """
        # Trigger computation of covariance pushforward and persist it on the 
        # cluster, since we will do several operations with it.
        cov_pushfwd = self.dask_client.persist(matmul(cov, transpose(G)))
        data_cov = data_var * eye(vector_data.shape[0])
        data_cov = data_cov.rechunk(data_cov.shape[0], data_cov.shape[1])

        # Define the graph for computing the updated mean vector.
        to_invert = (matmul(
                        G,
                        cov_pushfwd)
                    + data_cov)
        to_invert = to_invert.rechunk(to_invert.shape[0], to_invert.shape[1])
        print(to_invert)
        sqrt = cholesky(to_invert, lower=True)

        kalman_gain = matmul(
                cov_pushfwd,
                inv(to_invert))
        kalman_gain_tilde = matmul(
                cov_pushfwd,
                matmul(
                    transpose(inv(sqrt)),
                    inv(sqrt + cholesky(data_cov, lower=True))
                    )
                )

        prior_misfit = vector_data - matmul(G, vector_mean)
        vector_mean_updated = (
                vector_mean +
                matmul(kalman_gain, prior_misfit)
                )
        vector_mean_updated = self.dask_client.persist(vector_mean_updated)

        # Loop over members.
        vector_members_updated = []
        # for i in self.dataset_members.member_nr:
        for i in range(1, 3):
            print(i)
            vector_member = self.dataset_members.get_window_vector(
                    time_begin, time_end, member_nr=i)
            vector_member_updated = (
                    vector_member +
                    matmul(kalman_gain_tilde, matmul(G, vector_member))
                    )
            print(vector_member_updated)
            vector_member_updated = self.dask_client.persist(vector_member_updated)

            # member_updated = vector_member_updated.unstack('stacked_dim')
            vector_members_updated.append(vector_member_updated)

        time.sleep(7*60)
        return vector_mean_updated, vector_members_updated
        """
        return 1.0, 2.0
        """

        # Unstack the vector to go back to the grid format.
        mean_updated = self.dask_client.compute(vector_mean_updated.unstack('stacked_dim')).result()

        # Put into single dataset.
        members_updated = xr.concat(vector_members_updated, dim='member_nr')

        return mean_updated, members_updated
        """
