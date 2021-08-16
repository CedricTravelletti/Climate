import numpy as np
from climate.data_wrapper import DatasetWrapper
from climate.utils import build_base_forward
from scipy.linalg import block_diag
from dask.array import matmul, eye, transpose, cov
from dask.array.linalg import inv
import dask.array as da


class EnsembleKalmanFilter():
    """ Implementation of the ensemble Kalman filter.

    """
    def __init__(self, model_mean_dataset, model_members_dataset, data_dataset,
            chunk_size=1000):
        # First wrap the datasets.
        self.dataset_mean = DatasetWrapper(model_mean_dataset, chunk_size)
        self.dataset_members = DatasetWrapper(model_members_dataset, chunk_size)
        self.dataset_instrumental = DatasetWrapper(data_dataset, chunk_size)

        self.chunk_size = chunk_size

        # Get the base forward (the matrix making the translation between the
        # instrumental dataset and the other ones).
        self.G_base = build_base_forward(model_mean_dataset,
                data_dataset)
    
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
        G_nonan = self.G_base[np.logical_not(np.isnan(monthly_data)), :]

        for i in range(1, n_months):
            monthly_data = self.dataset_instrumental._to_1d_array(
                    dataset_instrumental.isel(time=i).values)
            G_tmp = self.G_base[np.logical_not(np.isnan(monthly_data)), :]

            # Append block diagonally.
            G_nonan = block_diag(G_nonan, G_tmp)

        return G_nonan

    def get_ensemble_covariance(self, time_begin, time_end):
        # Get the stacked window vectors for each ensemble member.
        vector_members = self.dataset_members.get_window_vector(time_begin, time_end)

        # Compute covariance matrix acrosss the different members.
        # Note that the below is a lazy operation (dask).
        cov_matrix = cov(vector_members, rowvar=False)
        return cov_matrix

    def update_mean_window(self, time_begin, time_end, n_months, data_var):
        """ 

        """
        # First get the mean vector and data vector (stacked for the window).
        vector_mean = self.dataset_mean.get_window_vector(time_begin, time_end)
        vector_data = self.dataset_instrumental.get_window_vector(time_begin, time_end)

        # Get rid of the Nans.
        vector_data = vector_data[np.logical_not(np.isnan(vector_data))]

        # Get covariance matrix and forward.
        G = self.get_forward_for_window(time_begin, time_end, n_months)
        G = da.from_array(G)
        cov = self.get_ensemble_covariance(time_begin, time_end)

        data_cov = data_var * eye(vector_data.shape[0], chunks=self.chunk_size)

        # Define the graph for computing the updated mean vector.
        to_invert = (matmul(
                        G,
                        matmul(cov, transpose(G)))
                    + data_cov).rechunk()
        kalman_gain = matmul(
                matmul(cov, transpose(G)),
                inv(to_invert))

        prior_misfit = vector_data - matmul(G, vector_mean)

        vector_mean_updated = (
                vector_mean +
                matmul(kalman_gain, prior_misfit)
                )
        return vector_mean_updated.compute()
