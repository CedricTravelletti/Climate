import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree


def match_datasets(base_dataset, dataset_tomatch):
    """" Match two datasets defined on different grid.

    Given a base dataset and a dataset to be matched, find for each point in
    the dataset to mathc the closest cell in the base dataset and return its
    index.

    Parameters
    ----------
    base_dataset: xarray.Dataset
    dataset_tomatch: xarray.Dataset

    """
    # Define original lat-lon grid
    # Creates new columns converting coordinate degrees to radians.
    lon_rad = np.deg2rad(dataset_members.longitude.values)
    lat_rad = np.deg2rad(dataset_members.latitude.values)
    lat_grid, lon_grid = np.meshgrid(lat_rad, lon_rad, indexing='ij')
    
    # Define grid to be matched.
    lon_tomatch = np.deg2rad(dataset_instrumental.longitude.values)
    lat_tomatch = np.deg2rad(dataset_instrumental.latitude.values)
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


class Dataset():
    def __init__(self, dataset):
        """

        Parameters
        ----------
        dataset: xarray.Dataset

        """
        self.dataset = dataset
        self.dims = (dataset.dims['latitude'], dataset.dims['longitude'])
        self.n_points = self.dims[0] * self.dims[1]

    def _get_coords(self):
        """ Returns a list of the coordinates of all points in the dataset.
        Note that we always work in radians.

        Returns
        -------
        coords: ndarray (self.n_points, 2)

        """
        lons = np.deg2rad(self.dataset.longitude.values)
        lats = np.deg2rad(self.dataset.latitude.values)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

        points_list = np.vstack([lat_grid.ravel().T, lon_grid.ravel().T]).T
        return points_list

    @property
    def coords(self):
        return self._get_coords()

    def _to_1d_array(self, array):
        return array.ravel()

    def _to_2d_array(self, array):
        return array.reshape(self.dims)
