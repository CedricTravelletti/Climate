#!/bin/bash
# Script for downloading the high-resolution climate data (August 2023) 
# and preprocessing it. 
# See also Climate/Data/high_res/README.txt for more information.


# The base folder where data will be stored.
BASE_FOLDER="/storage/homefs/ct19x463/Dev/Climate/Data"
# BASE_FOLDER="/home/cedric/PHD/Dev/Climate/Data/"
TOT_ENS_NUMBER=5 # Total number of different ensembles.

initial_wd=`pwd` # Save initial directory.

mkdir -p $BASE_FOLDER
cd $BASE_FOLDER

# Focus on high-resolution data.
mkdir -p "high_res"
cd "./high_res/"

# Get the urls.
# The goal here is to find the ensemble members.
rm urls_simulations.txt
curl -s http://giub-torrent.unibe.ch/PALAEO-RA/ModE-Sim/hires_5mem/abs/ | grep -E "m0|ens" | sed -n 's/.*href="\([^"]*\).*/\1/p' | awk '{print "http://giub-torrent.unibe.ch/PALAEO-RA/ModE-Sim/hires_5mem/abs/" $0 "by_var/"}' >> urls_simulations.txt
echo "Got URLs. Starting download."

mkdir -p "simulations"
cd "./simulations/"

# Loop over URLS (ensemble members) and download.
# For each member (and mean) only download temperature.
while IFS="" read -r p || [ -n "$p" ]
do
      url=$p
      curl -s "$url" | sed -n 's/.*href="\([^"]*\).*/\1/p' | grep -E "m0|mean" | grep "temp" | awk '{print url $0}' url=$url | parallel -j 5 --gnu "wget {}"
  done < ../urls_simulations.txt
