""" Compute the H matrix / forward operator for the HadCRUT dataset.

"""
import os
import numpy as np
from climate.utils import load_dataset
from climate.data_wrapper import build_base_forward


base_folder = "/home/cedric/PHD/Dev/Climate/Data/"
TOT_ENSEMBLES_NUMBER = 30

dataset_members, dataset_mean, dataset_instrumental = load_dataset(base_folder, TOT_ENSEMBLES_NUMBER)

G = build_base_forward(dataset_members, dataset_instrumental)
np.save(os.path.join(base_folder, "Computed/forward_operator.npy"), G)
