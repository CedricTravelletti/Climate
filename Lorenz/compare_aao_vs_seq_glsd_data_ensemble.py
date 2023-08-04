""" Compare all-at-once (aao) and sequential (seq) assimilation on the GLSD dataset. 
The data assimilated here is the station data from the GLSD dataset.

Reconstructions are compared to the reference dataset CRU TS 4. 
The goal is to show that aao is able to better reconstruct the reference than seq. 

Currently (25.04.2023) this script works (aao performs better than seq) and should 
be considered the state-of-the-art for the upcoming paper.

ENSEMBLE VERSION (10.06.2023): The previous version only computed the means. This 
one computes the full ensemble, so that finer scores can be used.

Remarks
-------
For a given month in the late 20th century, there are usually around 13k station 
data points to be assimilated.

"""
import os
import numpy as np
import dask
import pandas as pd
import dask.array as da
import xarray as xr

from climate.utils import load_dataset, match_vectors_indices
from climate.kalman_filter import EnsembleKalmanFilterScatter
from dask.distributed import Client, wait, progress                             
import diesel as ds                                                             
from diesel.scoring import compute_RE_score, compute_CRPS, compute_energy_score, compute_RMSE


base_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
results_folder = "/storage/homefs/ct19x463/Dev/Climate/reporting/all_at_once_vs_sequential/all_at_once_vs_sequential_68_ens/"


# Build Cluster
cluster = ds.cluster.UbelixCluster(n_nodes=12, mem_per_node=64, cores_per_node=3,
            partition="gpu", qos="job_gpu")                                     
cluster.scale(12)                                                           
client = Client(cluster)                                                    
                                                                                
# Add to builtins so we have one global client.
# Note that this is necessary before importing the EnsembleKalmanFilter module, so that the module is aware of the cluster.
__builtins__.CLIENT = client                                                
from diesel.kalman_filtering import EnsembleKalmanFilter 
from diesel.estimation import localize_covariance # Note that these two need to be imported 
# after the client has been created, since they need to know about the globals.

# Load the data.
TOT_ENSEMBLES_NUMBER = 30
(dataset_mean, dataset_members,
    dataset_instrumental, dataset_reference,
    dataset_members_zarr)= load_dataset(
    base_folder, TOT_ENSEMBLES_NUMBER, ignore_members=True)
print("Loading done.")

# This helper filter is only used for its data reshaping capabilites 
# It is used mostly for the get_window_vector function.
helper_filter = EnsembleKalmanFilterScatter(dataset_mean, dataset_members_zarr, dataset_instrumental, client)


def run_single_month(assimilation_date):
    # Extract the (vectorized/stacked) data for the current month.
    mean_ds = helper_filter.dataset_mean.get_window_vector(assimilation_date, assimilation_date, variable='temperature')
    ensemble_ds = helper_filter.dataset_members.get_window_vector(assimilation_date, assimilation_date, variable='temperature')
    mean_ds, ensemble_ds = client.persist(mean_ds), client.persist(ensemble_ds)
    
    
    # Load Station Data for the current month.
    year = int(assimilation_date[:4])
    data_df = pd.read_csv(os.path.join(base_folder, "Instrumental/GLSD/yearly_csv/temperature_{}.csv".format(year)), index_col=0)
    data_ds = xr.Dataset.from_dataframe(data_df)
    
    # Rename the date variable and make latitude/longitude into coordinates.
    data_ds = data_ds.rename({'date': 'time'})
    data_ds = data_ds.set_coords(['time', 'latitude', 'longitude'])
    data_ds = data_ds['temperature']
    
    # Prepare Forward Operator
    # Select one month.
    # Note that GLSD uses different reference for month (01 instead of 16), so have to replace.
    assimilation_date_datasel= assimilation_date[:-2] + '01'
    data_month_ds = data_ds.where(data_ds.time==assimilation_date_datasel, drop=True)
    
    # WARNING: THIS IS IMPORTANT.
    # Need to clean data since dataset contains erroneous measurements, i.e. 
    # either extreme values (10^30) or values that are exactly zero for a given station across time.
    data_month_ds = data_month_ds.where((data_month_ds > -100.0) & (data_month_ds < 100.0) & (da.abs(data_month_ds) > 0.0001), drop=True)
    data_vector = client.persist(da.from_array(data_month_ds.data))
    
    # Get the model cell index corresponding to each observations.
    # Here we basically match stations to grid points.
    matched_inds = match_vectors_indices(mean_ds, data_month_ds)
    
    # WARNING: Never try to execute bare loops in DASK, it will exceed the maximal graph depth.
    G = np.zeros((data_month_ds.shape[0], mean_ds.shape[0]))
    for obs_nr, model_cell_ind in enumerate(matched_inds):
        G[obs_nr, model_cell_ind] = 1.0
    
    G = da.from_array(G)
    G = client.persist(G)
    
    # Estimate Covariance 
    # Estimate covariance using empirical covariance of the ensemble.       
    raw_estimated_cov_lazy = ds.estimation.empirical_covariance(ensemble_ds.chunk((1, 1800)))  
                                                                                   
    # Persist the covariance on the cluster.                                
    raw_estimated_cov = client.persist(raw_estimated_cov_lazy) 
    progress(raw_estimated_cov)
    
    # Construct (lazy) localization matrix.                                       
    lambda0 = 1500 # Localization in kilometers.
    lengthscales = da.from_array([lambda0])   
    kernel = ds.covariance.squared_exponential(lengthscales)
    
    # Perform covariance localization.
    grid_pts = da.vstack([mean_ds.latitude, mean_ds.longitude]).T
    grid_pts = client.persist(grid_pts.rechunk((1800, 2)))
    localization_matrix = kernel.covariance_matrix(grid_pts, grid_pts, metric='haversine') 
    localization_matrix = client.persist(localization_matrix)
    progress(localization_matrix)
    
    # TODO: Here we have added multiplicative inflation.
    loc_estimated_cov = localize_covariance(raw_estimated_cov, localization_matrix)
    loc_estimated_cov = client.persist(loc_estimated_cov)
    progress(loc_estimated_cov)
    
    
    # Run Assimilation: All-at-once (aao) vs sequential (seq).
    # Run data assimilation using an ensemble Kalman filter.                
    my_filter = EnsembleKalmanFilter()                                      
    
    data_std = 1.0 # TODO: This seems big for data std.
    
    # Assimilate all data.
    mean_updated_aao, ensemble_updated_aao = my_filter.update_ensemble(
        mean_ds.data, ensemble_ds.data, G,
        data_vector, data_std, loc_estimated_cov)
    
    # Trigger computations and block. Otherwise will clutter the scheduler. 
    mean_updated_aao = client.persist(mean_updated_aao)                
    ensemble_updated_aao = client.persist(ensemble_updated_aao)
    # progress(mean_updated_aao) # Block till end of computations.                               
    
    # Run the sequential version.
    mean_updated_seq, ensemble_updated_seq = my_filter.update_ensemble_sequential_nondask(
             mean_ds.data, ensemble_ds.data, G,
             data_vector, data_std, loc_estimated_cov)
    
    
    # Unstack so we put everything back in 2D format.
    unstacked_updated_mean_aao = helper_filter.dataset_mean.unstack_window_vector(mean_updated_aao.compute(), time=assimilation_date, variable_name='temperature')
    unstacked_updated_mean_seq = helper_filter.dataset_mean.unstack_window_vector(mean_updated_seq, time=assimilation_date, variable_name='temperature')
    unstacked_prior_mean = helper_filter.dataset_mean.unstack_window_vector(mean_ds.values.reshape(-1), time=assimilation_date, variable_name='temperature')

    # Unstack the ensembles too.
    unstacked_updated_ensemble_aao = helper_filter.dataset_members.unstack_window_vector(
            ensemble_updated_aao.compute(), time=assimilation_date, variable_name='temperature')
    unstacked_updated_ensemble_seq = helper_filter.dataset_members.unstack_window_vector(
            ensemble_updated_seq, time=assimilation_date, variable_name='temperature')
    
    return (unstacked_updated_mean_aao, unstacked_updated_ensemble_aao,
            unstacked_updated_mean_seq, unstacked_updated_ensemble_seq,
            unstacked_prior_mean)

# Run for several years.
updated_means_aao, updated_ensembles_aao, updated_means_seq, updated_ensembles_seq, prior_means, references = [], [], [], [], [], []
for year in range(1969, 1970):
    for month in range(1, 13):
        month_str = str(month).zfill(2)
        assimilation_date = str(year) + '-' + month_str + '-16'
        print("Assimilating {}.".format(assimilation_date))
        
        # Run assimilation.
        (unstacked_updated_mean_aao, unstacked_updated_ensemble_aao,
                unstacked_updated_mean_seq, unstacked_updated_ensemble_seq,
                unstacked_prior_mean) = run_single_month(assimilation_date)

        # Compare the different updates.
        # ## Compute accuracy metrics.
        ref = dataset_reference.temperature.sel(time=assimilation_date)
        stacked_ref = ref.stack(stacked_dim=('latitude', 'longitude'))
        
        stacked_prior_mean = unstacked_prior_mean.stack(stacked_dim=('latitude', 'longitude'))
        stacked_updated_mean_seq = unstacked_updated_mean_seq.stack(stacked_dim=('latitude', 'longitude'))
        stacked_updated_mean_aao = unstacked_updated_mean_aao.stack(stacked_dim=('latitude', 'longitude'))
        
        print("Prior RMSE: {}".format(compute_RMSE(stacked_prior_mean.values, stacked_ref, min_lat=-70, max_lat=70)))
        print("seq RMSE: {}".format(compute_RMSE(stacked_updated_mean_seq.values, stacked_ref, min_lat=-70, max_lat=70)))
        print("aao RMSE: {}".format(compute_RMSE(stacked_updated_mean_aao.values, stacked_ref, min_lat=-70, max_lat=70)))
    
        # Save in lists.
        updated_means_aao.append(unstacked_updated_mean_aao.copy())
        updated_ensembles_aao.append(unstacked_updated_ensemble_aao.copy())
        updated_means_seq.append(unstacked_updated_mean_seq.copy())
        updated_ensembles_seq.append(unstacked_updated_ensemble_seq.copy())
        prior_means.append(unstacked_prior_mean.copy())
        references.append(ref)
        
    # Save results (updated temperature fields) each year.
    # Put everything into xarray Datasets (concatenated along time).
    updated_means_aao_ds = xr.concat(updated_means_aao, dim='time')
    updated_ensembles_aao_ds = xr.concat(updated_ensembles_aao, dim='time')
    updated_means_seq_ds = xr.concat(updated_means_seq, dim='time')
    updated_ensembles_seq_ds = xr.concat(updated_ensembles_seq, dim='time')
    prior_means_ds = xr.concat(prior_means, dim='time')
    references_ds = xr.concat(references, dim='time')
    
    # Save to disk.
    updated_means_aao_ds.to_netcdf(os.path.join(results_folder, "updated_means_aao.nc"))
    updated_ensembles_aao_ds.to_netcdf(os.path.join(results_folder, "updated_ensembles_aao.nc"))
    updated_means_seq_ds.to_netcdf(os.path.join(results_folder, "updated_means_seq.nc"))
    updated_ensembles_seq_ds.to_netcdf(os.path.join(results_folder, "updated_ensembles_seq.nc"))
    prior_means_ds.to_netcdf(os.path.join(results_folder, "prior_means.nc"))
    references_ds.to_netcdf(os.path.join(results_folder, "references.nc"))
