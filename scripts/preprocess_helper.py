""" Helper python script for the script download_and_preprocess.sh.
Takes care of adding a variable describing the ensemble member to all the
files.

This script is not supposed to be used directly.
"""
import os
import sys
import glob
import numpy as np
import netCDF4


def main(ENSEMBLE_FOLDER, TOT_ENSEMBLES_NUMBER):
    for i in range(1, int(TOT_ENSEMBLES_NUMBER) + 1):
        print("Processing Ensemble {}".format(i))
        paths = glob.glob(os.path.join(ENSEMBLE_FOLDER, "member_{}/".format(i)) + "*.nc")
        for path in paths:
            print("Processing file {}".format(path))
            dset = netCDF4.Dataset(path, 'r+')
            try:
                member_dim = dset.createDimension('member_nr', 1)
            except:
                pass
            try:
                member = dset.createVariable('member_nr', np.int32, ('member_nr',))
                member.units = 'member_nr'
                member.long_name = 'member_nr'
            except:
                pass
            dset.variables['member_nr'][:] = i
            dset.close()


if __name__ == "__main__":
        main(sys.argv[1], sys.argv[2])
