# =============================================================================
# RADIOSONDE FILE PROCESSING TO GENERATE PBLH DATASET
#
# Description:
# This script processes radiosonde (RS) profile measurements from multiple sources 
# (UWYO, DWD, NOAA, GRUAN) and extracts Planetary Boundary Layer Height (PBLH) 
# information into a unified dataset. It relies on consistent file naming and 
# directory structure conventions for each data source.
#
# Data Source Expectations:
# - UWYO: Files must be named 'sounding*.txt'
# - DWD: Files must be named 'produkt*.txt'
# - NOAA: Files must be stored in a sub-folder named 'NOAA'
# - GRUAN: Files must be in NetCDF format
#
# Key Steps:
# - Invoke a multithreaded loader to process all valid files from the specified sources
# - Extract and compute PBLH-related features using custom parsing logic
# - Consolidate the data into a single pandas DataFrame
# - Save the resulting dataset to a pickle for further analysis
#
# Parameters:
# - `num_threads`: Controls the number of parallel processing threads (default: 16)
#
# Dependencies:
# - Custom module: `RS_utilities.py` for file-specific parsers and logic
# - Python packages: pandas, numpy, os, gc, concurrent.futures
#
# Output:
# - `radiosonde_pblh.pkl`: A pickle file containing the processed PBLH dataset
#
# Author: Andreu Salcedo-Bosch
# Date: 13/06/2025
# =============================================================================

# Import the custom functions and requirements
from RS_utilities import *

# Number of threads for parallelization
num_threads = 16

############ READ THE RADIOSONDE FILES FROM EACH SOURCE ############################
# sources meaning:
# DWD: German Weather Service
# NOAA: National Oceanic and Atmospheric Administration
# GRUAN: Global Climate Observing System Reference Upper-Air Network
# UWYO: University of Wyoming upper-air sounding data-set

df_ablh = read_all_RS_data(rs_path, sources=['UWYO','DWD','NOAA','GRUAN'], num_threads= num_threads)

# Save the obtained dataframe to a pickle
df_ablh.to_pickle(os.path.join(pickles_path,'radiosonde_pblh.pkl'))