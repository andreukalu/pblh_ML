########### CODE TO PROCESS THE RADIOSONDE FILES FROM EACH SOURCE AND OBTAIN THE PBLH DATASET ################
# The code is based on the file structure of its creator. They can be adapted to the user.
# Radiosonde measurement files should be stored following these characteristics:
# UWYO: name of the file has to be 'sounding*.txt'
# GRUAN: has to be a netcdf
# NOAA: files have to be stored in a sub-folder called NOAA
# DWD: name of the file has to be 'produkt*.txt'

# Import the custom functions and requirements
from RS_utilities import *

# Set the path at which the radiosonde files are stored
base_path = ''

# Set the output path for the obtained dataframe
output_path = ''

# Number of threads for parallelization
num_threads = 16

############ READ THE RADIOSONDE FILES FROM EACH SOURCE ############################
# sources meaning:
# DWD: German Weather Service
# NOAA: National Oceanic and Atmospheric Administration
# GRUAN: Global Climate Observing System Reference Upper-Air Network
# UWYO: University of Wyoming upper-air sounding data-set

df_ablh = read_all_RS_data(base_path, sources=['UWYO','DWD','NOAA','GRUAN'], num_threads= num_threads)

# Save the obtained dataframe to a pickle
df_ablh.to_pickle(output_path)