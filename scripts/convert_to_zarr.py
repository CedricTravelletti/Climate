""" Convert the ensemble_mean and ensemble_members to single file zarr arrays. 
This allows for easier alignment of the chunks, which will help avoid out-of-memory 
errors when computing covariances.

"""
import os
import sys
from climate.utils import _load_dataset


def main(base_folder, TOT_ENSEMBLES_NUMBER):
    output_path_mean = os.path.join(base_folder, "ensemble_mean.zarr")
    output_path_members = os.path.join(base_folder, "ensemble_members.zarr")
    
    dataset_mean, dataset_members, dataset_instrumental, dataset_reference = _load_dataset(
            base_folder, TOT_ENSEMBLES_NUMBER)
    
    chunk_size = 500000 # Produces squared covariance chunks of size 400MB.
    
    # Remove unused variables. Need to reset_index since zarr 
    # does not allow multi-indices. This means we have to take care about order 
    # when we unstack.
    stacked_mean = dataset_mean.stack(
            stacked_dim=('time', 'latitude', 'longitude')).chunk(chunk_size)
    stacked_mean = stacked_mean.reset_index('stacked_dim')
    print(stacked_mean)
    stacked_mean.to_zarr(output_path_mean, mode="w")
    
    stacked_members = dataset_members.stack(
            stacked_dim=('time', 'latitude', 'longitude')).chunk(
                    {'member_nr': 1, 'stacked_dim': chunk_size})
    stacked_members = stacked_members.reset_index('stacked_dim')
    print(stacked_members)
    stacked_members.to_zarr(output_path_members, mode="w")
    

if __name__ == "__main__":
        main(sys.argv[1], sys.argv[2])
