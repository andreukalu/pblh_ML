from utilities import *

df_rs_1 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh'))
df_rs_2 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh_south_america'))
df_rs_2['altitude_res'] = 100
df_rs_3 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh_remaining'))
df_rs_3['altitude_res'] = 100
df_rs_4 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh_china'))
df_rs_4['altitude_res'] = 100
df_rs_5 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh_africa'))
df_rs_5['altitude_res'] = 100
df_rs = pd.concat([df_rs_1,df_rs_2,df_rs_3,df_rs_4,df_rs_5],axis=0).reset_index()

df_c1 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_us'))
df_c2 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_gruan'))
df_c3 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_germany'))
# df_c4 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_bcn'))
df_c5 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_uw'))
df_c6 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_ch'))
df_c7 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_rem'))
df_c = pd.concat([df_c1,df_c2,df_c3,df_c5,df_c6,df_c7],axis=0).reset_index()

print('CALIOP DATA SHAPE: ' + str(df_c.iloc[0]['backscatter'].shape))

grid_res = 0.5
df_c['lon'] = df_c['lon'].apply(lambda x: (x[0]+x[1])/2).astype(float)
df_c['lat'] = df_c['lat'].apply(lambda x: (x[0]+x[1])/2).astype(float)
df_c['lon_d'] = df_c['lon'].astype(float).apply(lambda x: round(x * 1/grid_res) / (1/grid_res) if not pd.isna(x) else np.nan)
df_c['lat_d'] = df_c['lat'].astype(float).apply(lambda x: round(x * 1/grid_res) / (1/grid_res) if not pd.isna(x) else np.nan)
df_rs['lon_d'] = df_rs['lon'].astype(float).apply(lambda x: round(x * 1/grid_res) / (1/grid_res) if not pd.isna(x) else np.nan)
df_rs['lat_d'] = df_rs['lat'].astype(float).apply(lambda x: round(x * 1/grid_res) / (1/grid_res) if not pd.isna(x) else np.nan)

df_rs['time_RS'] = df_rs['time'].apply(round_to_6_hour_interval)
df_c['time_RS'] = df_c['time'].apply(round_to_6_hour_interval)
df_c['day'] = df_c.apply(lambda x: is_daytime_approx(x['lat_d'], x['lon_d'], x['time_RS']), axis=1)
df_c['angle'] = df_c.apply(lambda x: solar_zenith(x['lat_d'], x['lon_d'], x['time_RS']), axis=1)
df_c['angle_c'] = df_c.apply(lambda x: solar_zenith(x['lat_d'], x['lon_d'], x['time']), axis=1)
df_c['distance_to_coast'] = fast_distance_to_coast(df_c['lat'].values, df_c['lon'].values)  # Convert degrees to km

df_c['time_c'] = df_c['time']
df_rs['time_rs'] = df_rs['time']
df_c['time_diff'] = (df_c['time_c'] - df_c['time_RS']).dt.total_seconds().abs() / 3600
df_c = df_c.drop(columns=['lon','lat','time','hour','month'])
df_rs = df_rs.drop(columns=['time'])

#Remove Stations
df_rs = df_rs[df_rs['id']!=723.0]
df_rs = df_rs[df_rs['id']!='USM00072388']
df_rs = df_rs[df_rs['id']!=726.0]
df_rs = df_rs[df_rs['id']!='USM00072681']
df_rs = df_rs[df_rs['id']!=348.0]

#Remove Duplicates
# df_rs = df_rs.sort_values(['lat_d', 'lon_d', 'time_RS', 'altitude_res']).drop_duplicates(['lat_d', 'lon_d', 'time_RS'], keep='first').reset_index(drop=True)

df_out = df_c.merge(df_rs,on=['lon_d','lat_d','time_RS']).drop(columns=['id'])
df_out['lon_diff'] = df_out['lon_d']-df_out['lon']
df_out['lat_diff'] = df_out['lat_d']-df_out['lat']

df_out.rename(columns={'backscatter':'X','ablh_rs':'y'}, inplace=True)
print(df_out.head(20))
# df_out['y'] = df_out['y'] - df_out['min_altitude_rs']

df_out = df_out[(df_out['y']>df_out['min_altitude_rs'])&((df_out['y']>0))]
df_out = df_out[df_out['y']<3500]
df_out = df_out[df_out['altitude_res']<900]
# df_out = df_out[(df_out['lat_d']>0)&(df_out['lat_d']<70)]
# df_out = df_out[(df_out['y']<3500)&(df_out['altitude_res']<900)&(df_out['time_diff']<2)].dropna()
# df_out = df_out.drop(df_out[(df_out['lat_d']>60)|(df_out['lat_d']<-60)].index)
# df_out = df_out.drop(df_out[(df_out['lat_d']<35)&(df_out['lat_d']>0)&(df_out['lon_d']>-20)&(df_out['lon_d']<60)].index)
# df_out = df_out.drop(df_out[(df_out['lat_d']<50)&(df_out['lat_d']>23)&(df_out['lon_d']>-120)&(df_out['lon_d']<-100)].index)


    # &((df_out['lon_d']<30)&(df_out['lon_d']>-100))\
                    # &((df_out['lon_d']<-120)&(df_out['lon_d']>-180))].dropna()

# df_out[df_out['y']<df_out['min_altitude_rs']]['y'] = df_out['min_altitude_rs']
print(df_out['y'].max()) #3500
print(df_out['y'].min()) #3500
df_out = df_out.drop(df_out[(df_out['y']>1000)&(df_out['day']==False)].index)
df_out['y'] = df_out['y']/3500#df_out['y'].max()
df_out['X'] = df_out['X'].apply(lambda x: x[:,25:])
df_out['power'] = df_out['X'].apply(lambda x: (x**2).sum())
df_out['cloud_flag'] = df_out['X'].apply(lambda x: int((x > 0.2).any()))
# df_out['X'] = df_out['X'].apply(lambda arr: (arr - np.median(arr,0)) / np.median(arr))
df_out['X'] = df_out['X']/df_out['power']
df_out['count'] = df_out['X'].apply(lambda x: np.sum(x>=0.97))
df_out = df_out[df_out['count']<1000]
df_out = df_out.drop(columns=['count'])

df_out = pd.concat([df_out,pd.get_dummies(df_out['day'],'day').astype(int)],axis=1).drop(columns=['day'])
df_out['month'] = df_out.apply(lambda x: x['time_RS'].month, axis=1)
df_out['hour'] = df_out.apply(lambda x: x['time_RS'].hour,axis=1)
df_out['lat_d_1'] = df_out['lat_d'].apply(lambda x: (np.sin(2*np.pi*((x+90)/180.0))/2))+0.5
df_out['lat_d_2'] = df_out['lat_d'].apply(lambda x: (np.cos(2*np.pi*((x+90)/180.0))/2))+0.5
df_out['lon_d_1'] = df_out['lon_d'].apply(lambda x: (np.sin(2*np.pi*((x+180)/360.0))/2))+0.5
df_out['lon_d_2'] = df_out['lon_d'].apply(lambda x: (np.cos(2*np.pi*((x+180)/360.0))/2))+0.5
df_out['month_1'] = df_out['month'].apply(lambda x: (np.sin(2*np.pi*x/365.0)/2))+0.5
df_out['month_2'] = df_out['month'].apply(lambda x: (np.cos(2*np.pi*x/365.0)/2))+0.5
df_out['angle_1'] = df_out['angle'].apply(lambda x: (np.sin(2*np.pi*x/180.0)/2))+0.5
df_out['angle_2'] = df_out['angle'].apply(lambda x: (np.cos(2*np.pi*x/180.0)/2))+0.5
df_out['min_altitude_rs'] = df_out['min_altitude_rs']/3500
df_out['hemisphere'] = np.sign(df_out['lat_d'])
df_out = df_out[(df_out['min_altitude_rs']>0)]
# df_out['lat_d'] = (df_out['lat_d']+90)/180
# df_out['lon_d'] = (df_out['lon_d']+180)/360

print(f"Data without clouds {df_out[df_out['cloud_flag']==0].shape}")
print(f"Data with clouds {df_out[df_out['cloud_flag']==1].shape}")
# df_out = df_out.drop(df_out[(df_out['y']<500)&(df_out['angle']<90.0)&(df_out['month']>=5)&(df_out['month']<=8)].index)

def compute_mean(matrix, axis=0):    
    mean_vertical_profile = np.mean(matrix, axis=axis)
    # mean_vertical_profile = (mean_vertical_profile - np.median(mean_vertical_profile))/(np.median(mean_vertical_profile))
    out = scipy.signal.medfilt(mean_vertical_profile,11)
    return out

def compute_sum(matrix, axis=0):    
    out = np.cumsum(matrix)
    return out

def compute_sector_mean(vec,sec=10):
    means = []
    segment_length = int(np.round(len(vec)/sec))
    
    for i in range(sec):
        vec_aux = vec[i*segment_length:(i+1)*segment_length]
        m = np.mean(vec_aux)
        means.append(m)
    return np.array(means)

def compute_wt(x,deltah=300):
    tr = np.array([])
    altitude = np.arange(0,len(x)*30,30)
    for z in altitude:
        idx_pos = (altitude >= z - deltah/2) & (altitude <= z)
        idx_neg = (altitude >= z) & (altitude <= z + deltah/2)
    
        tf_pos = np.mean(x[idx_pos])
        tf_neg = -np.mean(x[idx_neg])
    
        tf = tf_pos + tf_neg
        tr = np.append(tr,tf)
    return np.argmax(tr)

# Function to find the index of the closest value to the 25th percentile
def get_percentile_index(vector, percentile=25):
    prc_value = np.percentile(vector, percentile)  # Get percentile value
    return np.searchsorted(vector, prc_value)  # Find first index where cumsum exceeds the percentile value

extra_feature_cols = ['angle','angle_1','angle_2','distance_to_coast','power','lon','lat','lon_d','lat_d',\
        'month','month_1','month_2','hour','y','time_RS','min_altitude_rs','time_diff','altitude_res','prc25',\
        'prc50','prc75','prc25_i','prc50_i','prc75_i','pblh_wt','skew','kurt','cloud_flag']
df_out['mean_vector'] = df_out['X'].apply(lambda x: compute_mean(x, axis=0))  # Averaging along rows
df_out['power'] = df_out['mean_vector'].apply(lambda x: (x**2).sum())

# df_out['mean_vector'] = df_out['mean_vector']/df_out['power']  # Averaging along rows
# df_out['mean_vector'] = df_out['mean_vector'].apply(lambda x: (x - x.min())/(x.max() - x.min()))
# df_out['s_avg'] = df_out['mean_vector'].apply(lambda x: compute_sector_mean(x))
df_out['pblh_wt'] = df_out['mean_vector'].apply(lambda x: compute_wt(x))

# Apply function to each row
df_out['prc25'] = df_out['mean_vector'].apply(lambda x: np.percentile(compute_sum(x), 25))
df_out['prc50'] = df_out['mean_vector'].apply(lambda x: np.percentile(compute_sum(x), 50))
df_out['prc75'] = df_out['mean_vector'].apply(lambda x: np.percentile(compute_sum(x), 75))
df_out['skew'] = df_out['mean_vector'].apply(lambda x: skew(x))
df_out['kurt'] = df_out['mean_vector'].apply(lambda x: kurtosis(x))
df_out['prc25_i'] = df_out['mean_vector'].apply(lambda x: get_percentile_index(compute_sum(x), 25))
df_out['prc50_i'] = df_out['mean_vector'].apply(lambda x: get_percentile_index(compute_sum(x), 50))
df_out['prc75_i'] = df_out['mean_vector'].apply(lambda x: get_percentile_index(compute_sum(x), 75))
print(df_out.head())
df_out_aux = df_out[extra_feature_cols].reset_index()
df_out_aux_2 = pd.DataFrame(df_out['mean_vector'].to_list(), columns=[f'avg_col_{i}' for i in range(df_out['mean_vector'].iloc[0].shape[0])])
# df_out_aux_3 = pd.DataFrame(df_out['s_avg'].to_list(), columns=[f'avg_s_{i}' for i in range(df_out['s_avg'].iloc[0].shape[0])])
print(df_out.head())
df_out = pd.concat([df_out_aux,df_out_aux_2],axis=1).dropna()
df_out = pd.concat([df_out_aux,df_out_aux_2],axis=1).dropna()
print(df_out.shape)
print(df_out_aux_2.columns)

# Step 1: Group by unique locations
grouped = df_out.groupby(['lon_d', 'lat_d'])

# Step 2: Define Multiple Test Regions (Modify these as needed)
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
train_df.to_pickle(os.path.join(base_path,'Xtrain'))
val_df.to_pickle(os.path.join(base_path,'Xval'))
test_df.to_pickle(os.path.join(base_path,'Xtest'))