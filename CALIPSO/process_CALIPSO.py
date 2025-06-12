# CODE TO PROCESS CALIPSO MEASUREMENT FILES AND GET 2D PROFILES INTERSECTING RS LAUNCHING STATIONS #
# In order to save memory space, we first need to get the RS launching station locations,
# and then get and process the calipso over-passes

# Import the custom functions and requirements
from calipso_utilities import *

# GET THE RADIOSONDE DATASET

# Set the dimension of the lat/lon window around RS launching station
res = 1.0

# Read the pickle file containing the RS-measured PBLH
base_path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES'
df_rs = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh'))

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

# Set the path at which CALIOP files are stored
base_path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/CALIPSO/uw_coincidental_hr/'

# Set the output pickle file path
output_path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/calipso_CNN_uw'

# Process all the files
df_out = read_all_CALIPSO_data(base_path, df_rs, num_threads= num_threads)

# Save the obtained dataframe to a pickle
df_out.to_pickle(output_path)

del df_out
gc.collect()
