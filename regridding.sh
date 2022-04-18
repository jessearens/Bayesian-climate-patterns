#!/bin/bash
# A script for CMIP5 data preprocessing
# Retrieves prior downloaded temperature (tas) and precipitation (pr) model outputs.
# Computes a yearly mean from monthly mean data, taking into account the number of days in each month.
# Regrids data from varying model output resolutions into a 4x4 degree lat/lon grid.
# Saves the yearly mean & regridded data as a new file, for further processing.
# Move from shell script location to data location
# In our case, we run the shell script from a /Code folder, which exists in the same directory as the /Data folder.
# In our case, each model is accessed through the directory /Data/CMIP5/<parameter>/<model>
# Run this script by opening a Linux shell in the /Code folder and running "bash regridding.sh"


# Part 1: declare function that performs regridding
remap_data()
{	
	# Firstly, generate a regridding output folder.
	mkdir -p cdo_results
	
	# Importing and merging the historical and rcp8.5 experiments from the next climate model output. Commands used:
	# -mergetime: merge files based on the time field
	# -seltime: select specific time fields, in this case all data from 1920 up till 2005 (historical experiment) and 2006 up till 2100 (rcp8.5 experiment)
	# Last two fields are input & output. Usage of * means it imports all files that contain the text specified in their file name.
	cdo -selyear,1920/2005 -mergetime *Amon*historical*.nc cdo_results/"${PWD##*/}"_historical.nc
	cdo -selyear,2005/2100 -mergetime *Amon*rcp85*.nc cdo_results/"${PWD##*/}"_rcp85.nc
	cdo -mergetime cdo_results/*historical* cdo_results/*rcp85* cdo_results/"${PWD##*/}"_combined.nc

	
	# Converting monthly data into yearly mean data and regridding to a 4x4 degree grid. Commands used:
	# -yearmonmean: compute yearly mean, based on monthly data and adjusting for month length differences.
	# -remapcon,r90x45: remap the data (ie change grid) using a conservative algorithm(con), with output of 90x45 cells (= 4x4 degree lat/lon).
	# Last two fields are input & output
	cdo -yearmonmean -remapcon,r90x45 cdo_results/"${PWD##*/}"_combined.nc cdo_results/"${PWD##*/}"_yearmean_regrid.nc
}

# Part 2: Main script, loops through folders and calls the remap_data function in each climate model's folder.

# Start in general data folder
cd ../Data/CMIP5/

# Loop through folders in CMIP5 data directory.
# First */ refers to looping through all parameter folders (in this case "tas" and "pr")
# Second */ refers to looping through all climate model folders
for dir in */*/; do
	# Move to climate model folder
	echo "Now processing $dir model data."
	cd $dir
   
	# Uncomment the next line to reset & reprocess all.
	# rm -r cdo_results
	
	# Compute regridding iff there is no regridding output folder present yet.
	if [ -d cdo_results ]; then
		echo "Found cdo output already exists, processing skipped."
	else 
		echo "Computing remap of $dir data"
		remap_data
		echo "Processing of $dir model data completed."
	fi
	
	# Move path back to the general folder for the next model in the list.
   cd ../..
done