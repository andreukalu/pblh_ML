# =============================================================================
# FILE: generate_train_test_from_processed.py
# =============================================================================
# DESCRIPTION:
# This script generates training, validation, and test datasets to estimate
# the Planetary Boundary Layer Height (PBLH) from CALIOP TAB profiles using
# machine learning techniques. It combines and filters processed radiosonde and
# CALIOP data, performs feature engineering, and applies a geographically-aware
# train-test split.
#
# REQUIREMENTS:
# - Processed radiosonde and CALIOP data in pickle format
# - Functions from utilities_ML.py
# - paths_definition.py with correctly set `pickles_path` and `output_path`
#
# OUTPUT:
# - Pickled train, validation, and test datasets saved to output_path
#
# Author: Andreu Salcedo Bosch
# Date: 13/06/2025
# =============================================================================

from utilities_ML import *

# Load radiosonde data
df_rs = pd.read_pickle(os.path.join(pickles_path, 'radiosonde_pblh.pkl'))

# Remove non-trustworthy stations
bad_ids = [723.0, 'USM00072388', 726.0, 'USM00072681', 348.0]
df_rs = df_rs[~df_rs['id'].isin(bad_ids)]

# Load CALIOP data
df_c = pd.read_pickle(os.path.join(pickles_path, 'calipso.pkl'))

# === DATA INTERSECTION ===
# Intersect data in time and space
grid_res = 0.5

# Average bounding boxes to get center coordinates
df_c['lon'] = df_c['lon'].apply(lambda x: (x[0] + x[1]) / 2).astype(float)
df_c['lat'] = df_c['lat'].apply(lambda x: (x[0] + x[1]) / 2).astype(float)

# Discretize spatial coordinates
df_c['lon_d'] = df_c['lon'].apply(lambda x: round(x / grid_res) * grid_res)
df_c['lat_d'] = df_c['lat'].apply(lambda x: round(x / grid_res) * grid_res)
df_rs['lon_d'] = df_rs['lon'].apply(lambda x: round(x / grid_res) * grid_res)
df_rs['lat_d'] = df_rs['lat'].apply(lambda x: round(x / grid_res) * grid_res)

# Round time to 6-hour interval
df_rs['time_RS'] = df_rs['time'].apply(round_to_6_hour_interval)
df_c['time_RS'] = df_c['time'].apply(round_to_6_hour_interval)

# Drop redundant columns
df_c.drop(columns=['lon', 'lat', 'time'], inplace=True)
df_rs.drop(columns=['time'], inplace=True)

# Merge datasets
df_out = df_c.merge(df_rs, on=['lon_d', 'lat_d', 'time_RS']).drop(columns=['id'])
df_out.rename(columns={'backscatter': 'X', 'ablh_rs': 'y'}, inplace=True)

# === FEATURE ENGINEERING ===
# Filter out invalid and extreme PBLH values
df_out = df_out[
    (df_out['y'] > df_out['min_altitude_rs']) &
    (df_out['y'] > 0) &
    (df_out['min_altitude_rs'] > 0) &
    (df_out['y'] < 3500) &
    (df_out['altitude_res'] < 900)
]

# Normalize PBLH
df_out['y'] /= 3500

# Remove TAB data below 0 m a.s.l.
df_out['X'] = df_out['X'].apply(lambda x: x[:, 25:])

# Normalize TAB by signal power
df_out['power'] = df_out['X'].apply(lambda x: (x**2).sum())
df_out['X'] = df_out['X'] / df_out['power']

# Filter outliers with extreme signal concentration
df_out['count'] = df_out['X'].apply(lambda x: np.sum(x >= 0.97))
df_out = df_out[df_out['count'] < 1000].drop(columns=['count'])

# === CONTEXTUAL FEATURES ===
df_out['angle'] = df_out.apply(lambda x: solar_zenith(x['lat_d'], x['lon_d'], x['time_RS']), axis=1)
df_out['distance_to_coast'] = fast_distance_to_coast(df_out['lat'].values, df_out['lon'].values)
df_out['min_altitude_rs'] /= 3500
df_out['month'] = df_out['time_RS'].apply(lambda x: x.month)

# === STATISTICAL FEATURES ===
df_out['mean_vector'] = df_out['X'].apply(lambda x: compute_mean(x, axis=0))
df_out['power'] = df_out['mean_vector'].apply(lambda x: (x**2).sum())
df_out['skew'] = df_out['mean_vector'].apply(skew)
df_out['kurt'] = df_out['mean_vector'].apply(kurtosis)

# Combine features into a flat table
extra_feature_cols = [
    'angle', 'distance_to_coast', 'power', 'lon_d', 'lat_d',
    'month', 'y', 'min_altitude_rs', 'altitude_res', 'skew', 'kurt'
]

df_out_aux = df_out[extra_feature_cols].reset_index()
df_out_aux_2 = pd.DataFrame(
    df_out['mean_vector'].tolist(),
    columns=[f'avg_col_{i}' for i in range(df_out['mean_vector'].iloc[0].shape[0])]
)

df_out = pd.concat([df_out_aux, df_out_aux_2], axis=1).dropna()

# === TRAIN/VAL/TEST SPLIT ===
grouped = df_out.groupby(['lon_d', 'lat_d'])

# Define test regions (Europe + US)
test_regions = [
    {"min_lat": 35, "max_lat": 60, "min_lon": -20, "max_lon": 30},
    {"min_lat": 25, "max_lat": 50, "min_lon": -125, "max_lon": -60}
]

region_keys = [key for key in grouped.groups.keys() if is_in_test_region(key[1], key[0], test_regions)]

# Select test set (~10% of total)
total_samples = len(df_out)
test_target_size = int(0.1 * total_samples)
random_state = 14

np.random.seed(random_state)
np.random.shuffle(region_keys)

test_keys_selected = []
current_test_size = 0

for key in region_keys:
    group_size = len(grouped.get_group(key))
    if current_test_size + group_size <= test_target_size:
        test_keys_selected.append(key)
        current_test_size += group_size
    if current_test_size >= test_target_size:
        break

# Split remaining into train/val
remaining_keys = [key for key in grouped.groups.keys() if key not in test_keys_selected]
remaining_keys_shuffled = pd.Series(remaining_keys).sample(frac=1, random_state=random_state).tolist()
train_keys, val_keys = train_test_split(remaining_keys_shuffled, test_size=0.1/0.9, random_state=random_state)

# Create final splits
train_df = pd.concat([grouped.get_group(k) for k in train_keys])
val_df = pd.concat([grouped.get_group(k) for k in val_keys])
test_df = pd.concat([grouped.get_group(k) for k in test_keys_selected])

# Debug print overlaps
print("Overlap between train and test:", set(train_keys) & set(test_keys_selected))
print("Overlap between val and test:", set(val_keys) & set(test_keys_selected))
print("Overlap between val and train:", set(val_keys) & set(train_keys))

print(f"Train df size: {len(train_df)}")
print(f"Test df size: {len(test_df)}")
print(f"Validation df size: {len(val_df)}")

# Save datasets
train_df = train_df.sample(frac=1)
val_df = val_df.sample(frac=1)
test_df = test_df.sample(frac=1)

train_df.to_pickle(os.path.join(pickles_path, 'Xtrain'))
val_df.to_pickle(os.path.join(pickles_path, 'Xval'))
test_df.to_pickle(os.path.join(pickles_path, 'Xtest'))
