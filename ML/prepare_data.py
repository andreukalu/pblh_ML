from utilities_ML import *

# Read the pickle file containing the RS-measured PBLH
rs_path = ''
# Set the path at which CALIOP files are stored
c_path = ''
# Set the output pickle file path
output_path = ''

# Read radiosonde data
df_rs = pd.read_pickle(os.path.join(rs_path,'radiosonde_ablh'))

#Remove non-trustworthy stations
df_rs = df_rs[df_rs['id']!=723.0]
df_rs = df_rs[df_rs['id']!='USM00072388']
df_rs = df_rs[df_rs['id']!=726.0]
df_rs = df_rs[df_rs['id']!='USM00072681']
df_rs = df_rs[df_rs['id']!=348.0]

# Read calipso data
df_c = pd.read_pickle(os.path.join(c_path,'calipso_CNN_gruan'))

########### DATA INTERSECTION ########################
# Intersect in time and space radiosonde and calipso
# First in space
grid_res = 0.5
df_c['lon'] = df_c['lon'].apply(lambda x: (x[0]+x[1])/2).astype(float)
df_c['lat'] = df_c['lat'].apply(lambda x: (x[0]+x[1])/2).astype(float)
df_c['lon_d'] = df_c['lon'].astype(float).apply(lambda x: round(x * 1/grid_res) / (1/grid_res) if not pd.isna(x) else np.nan)
df_c['lat_d'] = df_c['lat'].astype(float).apply(lambda x: round(x * 1/grid_res) / (1/grid_res) if not pd.isna(x) else np.nan)
df_rs['lon_d'] = df_rs['lon'].astype(float).apply(lambda x: round(x * 1/grid_res) / (1/grid_res) if not pd.isna(x) else np.nan)
df_rs['lat_d'] = df_rs['lat'].astype(float).apply(lambda x: round(x * 1/grid_res) / (1/grid_res) if not pd.isna(x) else np.nan)

# Then in time
df_rs['time_RS'] = df_rs['time'].apply(round_to_6_hour_interval)
df_c['time_RS'] = df_c['time'].apply(round_to_6_hour_interval)

# Drop unnecessary columns
df_c = df_c.drop(columns=['lon','lat','time'])
df_rs = df_rs.drop(columns=['time'])

# Merge calipso and radiosonde dataframes
df_out = df_c.merge(df_rs,on=['lon_d','lat_d','time_RS']).drop(columns=['id'])

# Rename the columns to a more ML-like notation
df_out.rename(columns={'backscatter':'X','ablh_rs':'y'}, inplace=True)

################ FEATURE ENGINEERING ########################
# Remove outliers -> PBLH<0 m, PBLH>3500 m, and RS gaps > 900 m
df_out = df_out[(df_out['y']>df_out['min_altitude_rs'])&((df_out['y']>0))]
df_out = df_out[(df_out['min_altitude_rs']>0)]
df_out = df_out[df_out['y']<3500]
df_out = df_out[df_out['altitude_res']<900]

# Normalize the PBLH
df_out['y'] = df_out['y']/3500

# Remove TAB below 0 m a.s.l.
df_out['X'] = df_out['X'].apply(lambda x: x[:,25:])
df_out['power'] = df_out['X'].apply(lambda x: (x**2).sum())

# Normalize the TAB by the signal power
df_out['X'] = df_out['X']/df_out['power']

df_out['count'] = df_out['X'].apply(lambda x: np.sum(x>=0.97))
df_out = df_out[df_out['count']<1000]
df_out = df_out.drop(columns=['count'])

############## GET CONTEXTUAL FEATURES ##########################
# Compute Solar Zenith Angle
df_out['angle'] = df_out.apply(lambda x: solar_zenith(x['lat_d'], x['lon_d'], x['time_RS']), axis=1)

# Get the distance to coast
df_out['distance_to_coast'] = fast_distance_to_coast(df_out['lat'].values, df_out['lon'].values)  # Convert degrees to km

# Normalize the minimum altitude
df_out['min_altitude_rs'] = df_out['min_altitude_rs']/3500

# Get the measurement month
df_out['month'] = df_out.apply(lambda x: x['time_RS'].month, axis=1)

############## GET STATISTICAL FEATURES ##########################
# Reduce the 2D TAB into 1D vertical profiles
df_out['mean_vector'] = df_out['X'].apply(lambda x: compute_mean(x, axis=0))  # Averaging along rows

# Get statistical parameters
df_out['power'] = df_out['mean_vector'].apply(lambda x: (x**2).sum())
df_out['skew'] = df_out['mean_vector'].apply(lambda x: skew(x))
df_out['kurt'] = df_out['mean_vector'].apply(lambda x: kurtosis(x))

# Set extra features to be used
extra_feature_cols = ['angle','distance_to_coast','power','lon_d','lat_d',\
        'month','y','min_altitude_rs','altitude_res','skew','kurt']

# Prepare the output dataset combining TAB vertical profiles and extra featuers
df_out_aux = df_out[extra_feature_cols].reset_index()
df_out_aux_2 = pd.DataFrame(df_out['mean_vector'].to_list(), columns=[f'avg_col_{i}' for i in range(df_out['mean_vector'].iloc[0].shape[0])])

df_out = pd.concat([df_out_aux,df_out_aux_2],axis=1).dropna()

############## SPLIT INTO TRAIN AND TEST ################
# Step 1: Group by unique locations
grouped = df_out.groupby(['lon_d', 'lat_d'])

# Step 2: Define Europe and US Test Regions
test_regions = [
    {"min_lat": 35, "max_lat": 60, "min_lon": -20, "max_lon": 30},  # Region 1
    {"min_lat": 25, "max_lat": 50, "min_lon": -125, "max_lon": -60}   # Region 2
]

# Function to check if a location is inside ANY of the test regions
def is_in_test_region(lat, lon, regions):
    return any((r["min_lat"] <= lat <= r["max_lat"]) and (r["min_lon"] <= lon <= r["max_lon"]) for r in regions)

# Step 3: Identify locations inside test regions
region_keys = [key for key in grouped.groups.keys() if is_in_test_region(key[1], key[0], test_regions)]

# Step 4: Determine how many samples should be in the test set (10% of total data)
total_samples = len(df_out)
test_target_size = int(0.1 * total_samples)

# Step 5: Randomly select test locations until we reach the test size limit
random_state = 14
test_keys_selected = []
current_test_size = 0

# Shuffle region keys
np.random.seed(random_state)
np.random.shuffle(region_keys)

# Select test groups until we reach the required test size
for key in region_keys:
    group_size = len(grouped.get_group(key))
    if current_test_size + group_size <= test_target_size:
        test_keys_selected.append(key)
        current_test_size += group_size
    if current_test_size >= test_target_size:
        break

# Step 6: Get remaining keys for train + validation (includes non-test region keys + remaining region keys)
remaining_keys = [key for key in grouped.groups.keys() if key not in test_keys_selected]

# Shuffle the remaining keys
remaining_keys_shuffled = pd.Series(remaining_keys).sample(frac=1, random_state=random_state).tolist()

# Step 7: Split remaining keys into Train (80%) and Validation (10%) - These make up 90% of total data
train_keys, val_keys = train_test_split(remaining_keys_shuffled, test_size=0.1/0.9, random_state=random_state)

# Step 8: Create DataFrames for each set
train_df = pd.concat([grouped.get_group(key) for key in train_keys], axis=0)
val_df = pd.concat([grouped.get_group(key) for key in val_keys], axis=0)
test_df = pd.concat([grouped.get_group(key) for key in test_keys_selected], axis=0)

train_groups = set(train_df[['lon_d', 'lat_d']].drop_duplicates().apply(tuple, axis=1))
test_groups = set(test_df[['lon_d', 'lat_d']].drop_duplicates().apply(tuple, axis=1))
val_groups = set(val_df[['lon_d', 'lat_d']].drop_duplicates().apply(tuple, axis=1))

print("Overlap between train and test:", train_groups & test_groups)
print("Overlap between val and test:", val_groups & test_groups)
print("Overlap between val and train:"
, val_groups & train_groups)

print("Train df size: ",len(train_df))
print("Test df size: ",len(test_df))
print("Validation df size: ",len(val_df))

train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)
val_df = val_df.sample(frac=1)
train_df.to_pickle(os.path.join(output_path,'Xtrain'))
val_df.to_pickle(os.path.join(output_path,'Xval'))
test_df.to_pickle(os.path.join(output_path,'Xtest'))