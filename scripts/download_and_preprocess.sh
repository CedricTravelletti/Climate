#!/bin/bash
# Script for downloading the Climate data and re-arranging it in usable format.
#
# -----------
# USER MANUAL
# -----------
# User should provide a BASE_FOLDER where data will be stored.
# Seen from BASE_FOLDER, the resulting data structure will be as follows:
#
# ./Ensembles/
#   ./Means/
#        ./CCC400_ensmean_1602_v2.0.nc
# With one file per year.
#   ./Members/
#       ./member_1/
#           ./CCC400_ens_mem_1_1603_v2.0.nc
# With one folder per ensemble member (simulation run) and in each folder one
# file per year.
# ./Instrumental/
#   ./HadCRUT.4.6.0.0.median.nc
# File with instrumental temperature dataset.
#
# -------------------------------------------
# -------------------------------------------


# The base folder where data will be stored.
# BASE_FOLDER="/storage/homefs/ct19x463/Dev/Climate/Data"
BASE_FOLDER="/home/cedric/PHD/Dev/Climate/Data/"
TOT_ENS_NUMBER=30 # Total number of different ensembles.


initial_wd=`pwd` # Save initial directory.

mkdir -p $BASE_FOLDER
cd $BASE_FOLDER

# Instrumental Data.
mkdir -p "Instrumental"
cd "./Instrumental/"
# Get HadCRUT.
wget "https://crudata.uea.ac.uk/cru/data/temperature/HadCRUT.4.6.0.0.median.nc"

# Get Global Land surface Databank.
mkdir -p "GLSD"
cd "./GLSD/"
wget ftp://ftp.ncdc.noaa.gov:21/pub/data/globaldatabank/monthly/stage3/recommended/results/recommended-netcdf_format.monthly.stage3.v1.1.1.20181009.tar.gz
tar -zxvf recommended-netcdf_format.monthly.stage3.v1.1.1.20181009.tar.gz

cd ../..

# Reference Data.
mkdir -p "Reference"
cd "./Reference/"
wget https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.05/cruts.2103051243.v4.05/tmp/cru_ts4.05.1901.2020.tmp.dat.nc.gz
gunzip cru_ts4.05.1901.2020.tmp.dat.nc.gz
cd ..

# Ensemble Data.
mkdir -p "Ensembles"
cd "./Ensembles/"

# Get Means URLs.
echo "Getting URLs".
curl http://giub-torrent.unibe.ch/DATA/REUSE/CCC400_ensmean/ | grep -i nc | sed -n 's/.*href="\([^"]*\).*/\1/p' >> urls_ensmean.txt

# Get Members URLs.
curl http://giub-torrent.unibe.ch/DATA/REUSE/CCC400_ens_mem/ | grep -i nc | sed -n 's/.*href="\([^"]*\).*/\1/p' >> urls_ens_mem.txt

# Make directories and download.
echo "Got URLs. Starting download."
mkdir -p "Means"
cd "./Means/"
cat "../urls_ensmean.txt" | parallel -j 10 --gnu "wget http://giub-torrent.unibe.ch/DATA/REUSE/CCC400_ensmean/{}"

cd ..

mkdir -p "Members"
cd "./Members/"
ensemble_members_folder=`pwd` # Save directory location.
cat "../urls_ens_mem.txt" | parallel -j 10 --gnu "wget http://giub-torrent.unibe.ch/DATA/REUSE/CCC400_ens_mem/{}"

echo "Download Finished. Starting postprocessing."

# For each different member number, create a directory and move corresponding
# data in there.
for i in $(ls | perl -ne 's/.*mem_(\d+).*/$1/;print if /^\d+$/;' | uniq)
do
    echo "Processing member $i"
    mkdir -p "member_$i"
    for FILE in $(find . -regextype sed -regex '.*/.*_mem_'"$i"'_.*')
    do
        echo "Moving $FILE"
        mv "$FILE" "./member_$i/"
    done
done
echo "Done moving each ensemble member to its own directory."

# Finally, run python script to add a variable describing the ensemble member
# to each NetCDF4 file.
cd $initial_wd
echo $ensemble_members_folder
python preprocess_helper.py "$ensemble_members_folder" "$TOT_ENS_NUMBER"

# Last, we also process the ensemble members and mean to zarr files, 
# for easier chunk alignment.
echo "Converting to zarr."
python convert_to_zarr.py "$BASE_FOLDER" $TOT_ENS_NUMBER
echo "Done converting to zarr.\n Preprocessing completed."
