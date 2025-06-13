# =============================================================================
# CALIPSO L1 DATA PROCESSING FOR RS LAUNCH LOCATION INTERSECTIONS
#
# Description:
# This script processes CALIPSO Level 1 backscatter measurement files and extracts 
# 2D vertical profiles that intersect with radiosonde (RS) launching stations. 
# It does this by first determining the RS station coordinates, creating a spatial 
# window around them, and then filtering CALIPSO overpasses to retain only the 
# relevant segments. The final output is a compact dataset suitable for ML applications.
#
# Key Steps:
# - Load radiosonde station locations from pre-processed pickle
# - Define spatial windows (±1.0°) around each RS station
# - Process all CALIPSO overpass files using multithreaded parallelism
# - Save the processed CALIPSO data to a pickle file for downstream use
#
# Notes:
# - Spatial filtering helps reduce memory usage and processing time
# - Processing uses a high number of threads for performance (default: 256)
#
# Dependencies:
# - Requires `calipso_utilities.py` for custom file reading and processing functions
# - Python packages: pandas, numpy, gc, os, concurrent.futures
#
# Author: Andreu Salcedo Bosch
# Date: 13/06/2025
# =============================================================================

# Import the custom functions and requirements
from calipso_utilities import *

# GET THE RADIOSONDE DATASET

# Set the dimension of the lat/lon window around RS launching station
res = 1.0

# Load Radiosonde data
df_rs = pd.read_pickle(os.path.join(pickles_path,'radiosonde_pblh.pkl'))

# Get unique lat/lon coordinates set for all RS stations
df_rs['lon'] = df_rs['lon'].astype(float).round(1)
df_rs['lat'] = df_rs['lat'].astype(float).round(1)
df_rs = df_rs[['lon','lat']].drop_duplicates()

# Generate the lat/lon windows arround the RS launching stations 
df_rs['lon_min'] = df_rs['lon']-res
df_rs['lon_max'] = df_rs['lon']+res
df_rs['lat_min'] = df_rs['lat']-res
df_rs['lat_max'] = df_rs['lat']+res

# PROCESS THE CALIPSO L1 MEASUREMENT FILES
# Set the number of parallelization threads
num_threads = 256

# Process all the files
df_out = read_all_CALIPSO_data(c_path, df_rs, num_threads= num_threads)

# Save the obtained dataframe to a pickle
df_out.to_pickle(os.path.join(pickles_path,'calipso.pkl'))

del df_out
gc.collect()
