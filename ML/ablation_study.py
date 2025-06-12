# === Import all utilities from the specified module (e.g., for metrics, plotting, etc.) ===
from utilities_ML import *

# === Define paths and experiment-level parameters ===
pickles_path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES'  # Folder where pickled datasets are stored
method = 'rf'                  # Chosen ML method (here: Random Forest)
param = False                  # Placeholder for parameter tuning (unused in this script)
random_flag = False           # Placeholder for randomization control (unused here)

# === Load datasets (already preprocessed and pickled) ===
train_df = pd.read_pickle(os.path.join(pickles_path, 'Xtrain')).drop(columns=['index'])
val_df = pd.read_pickle(os.path.join(pickles_path, 'Xval')).drop(columns=['index'])
train_df = pd.concat([train_df, val_df])  # Combine training and validation sets into one
test_df = pd.read_pickle(os.path.join(pickles_path, 'Xtest')).drop(columns=['index'])

# === Geographical filtering: remove points below latitude 30 and outside [-120, 120] longitude range ===
train_df = train_df.drop(train_df[(train_df['lon_d'] < -120) & (train_df['lat_d'] < 30)].index)
train_df = train_df.drop(train_df[(train_df['lon_d'] > 120) & (train_df['lat_d'] < 30)].index)
test_df = test_df.drop(test_df[(test_df['lon_d'] < -120) & (test_df['lat_d'] < 30)].index)
test_df = test_df.drop(test_df[(test_df['lon_d'] > 120) & (test_df['lat_d'] < 30)].index)

# === Adjust target variable: subtract minimum altitude for relative height prediction ===
train_df['y'] = train_df['y'] - train_df['min_altitude_rs']
test_df['y'] = test_df['y'] - test_df['min_altitude_rs']

# === Create group labels based on unique spatial coordinates (for potential grouping/CV) ===
train_df['group'] = train_df.groupby(['lon_d', 'lat_d']).ngroup()
groups = train_df['group']

# === Separate labels (target values) from feature sets ===
y_train = train_df['y']
y_val = val_df['y']  # Note: y_val is loaded but not used later
y_test = test_df['y']

# === Drop unused columns ===
train_df = train_df.drop(columns=['time_RS', 'y', 'group'])  # Drop target, time, and group columns from train
val_df = val_df.drop(columns=['time_RS', 'y'])               # Same for val
test_df = test_df.drop(columns=['time_RS', 'y'])             # Same for test
test_min_altitude = test_df['min_altitude_rs']               # Save for optional later use (in evaluation)

# === Define ablation study: 5 different sets of features to test model sensitivity ===
columns_list = [
    ['angle', 'month', 'power', 'skew', 'kurt',
     'avg_col_0','avg_col_10','avg_col_20','avg_col_30',
     'avg_col_40','avg_col_50','avg_col_60','avg_col_70',
     'avg_col_80','avg_col_90','avg_col_100'],  # No geographical info

    ['lat_d','lon_d','power','skew','kurt','distance_to_coast','min_altitude_rs',
     'avg_col_0','avg_col_10','avg_col_20','avg_col_30',
     'avg_col_40','avg_col_50','avg_col_60','avg_col_70',
     'avg_col_80','avg_col_90','avg_col_100'],  # No temporal info

    ['power','skew','kurt',
     'avg_col_0','avg_col_10','avg_col_20','avg_col_30',
     'avg_col_40','avg_col_50','avg_col_60','avg_col_70',
     'avg_col_80','avg_col_90','avg_col_100'],  # No contextual info

    ['angle','month','lat_d','lon_d','distance_to_coast','min_altitude_rs',
     'avg_col_0','avg_col_10','avg_col_20','avg_col_30',
     'avg_col_40','avg_col_50','avg_col_60','avg_col_70',
     'avg_col_80','avg_col_90','avg_col_100'],  # No statistical info

    ['avg_col_0','avg_col_10','avg_col_20','avg_col_30',
     'avg_col_40','avg_col_50','avg_col_60','avg_col_70',
     'avg_col_80','avg_col_90','avg_col_100']  # Only vertical column info (baseline)
]

description_list = ['no geographical', 'no temporal', 'no contextual', 'no statistical', 'nothing']

# === Loop over each feature subset to train, evaluate, and store results ===
results = []
for i, columns in enumerate(columns_list):
    # === Extract feature subsets for training, validation, and test ===
    train_df_subset = train_df[columns].copy()
    val_df_subset = val_df[columns].copy()
    test_df_subset = test_df[columns].copy()

    # === Optionally add Gaussian noise to simulate uncertainty ===
    noise_std = 1.0
    if 'lat_d' in train_df_subset.columns:
        train_df_subset['lat_d'] += np.random.normal(0, noise_std, train_df_subset.shape[0])
    if 'lon_d' in train_df_subset.columns:
        train_df_subset['lon_d'] += np.random.normal(0, noise_std, train_df_subset.shape[0])
    if 'min_altitude_rs' in train_df_subset.columns:
        train_df_subset['min_altitude_rs'] += np.random.normal(0, 0.1, train_df_subset.shape[0])

    print(f"\n▶️ Description: {description_list[i]}")
    print('Train set size:', train_df_subset.shape)

    # === Initialize and train Random Forest model ===
    model = RandomForestRegressor(
        random_state=42,
        max_features=1.0,
        verbose=1,
        max_depth=43,
        n_estimators=850,
        min_samples_leaf=4,
        min_samples_split=2,
        n_jobs=-1,
        bootstrap=True
    )
    model.fit(train_df_subset, y_train)

    # === Evaluate model on test set ===
    y_pred_test = model.predict(test_df_subset)
    mse_test = mean_squared_error(y_test - test_min_altitude, y_pred_test - test_min_altitude)
    r2_test = r2_score(y_test - test_min_altitude, y_pred_test - test_min_altitude)
    rmse_test = math.sqrt(mse_test)

    # === Evaluate model on training set ===
    y_pred_train = model.predict(train_df_subset)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = math.sqrt(mse_train)

    # === Save trained model to disk ===
    with open(f'model_{i}.pkl', 'wb') as f:
        pickle.dump(model, f)

    # === Store evaluation results ===
    results.append({
        'description': description_list[i],
        'R2_test': r2_test,
        'RMSE_test': rmse_test,
        'R2_train': r2_train,
        'RMSE_train': rmse_train
    })

    print(f"Test - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}")
    print(f"Train - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}")

    # === Save prediction results on test set ===
    test_df_eval = pd.read_pickle(os.path.join(pickles_path, 'Xtest'))  # Reload raw test set
    test_df_eval = test_df_eval.drop(test_df_eval[(test_df_eval['lon_d'] < -120) & (test_df_eval['lat_d'] < 30)].index)
    test_df_eval = test_df_eval.drop(test_df_eval[(test_df_eval['lon_d'] > 120) & (test_df_eval['lat_d'] < 30)].index)
    test_df_eval['pred'] = y_pred_test
    test_df_eval['og'] = y_test
    test_df_eval.to_pickle(f'predictions_noclouds_{i}.pkl')

    # === Save prediction results on train set ===
    train_df_subset['pred'] = y_pred_train
    train_df_subset['og'] = y_train
    train_df_subset.to_pickle(f'predictions_train_{i}.pkl')

# === Write all results to a summary text file ===
with open('ablation_results.txt', 'w') as f:
    f.write("Ablation Study Results (Random Forest)\n")
    f.write("="*50 + "\n\n")
    for res in results:
        f.write(f"Description: {res['description']}\n")
        f.write(f"  ▶ R² (Test):  {res['R2_test']:.4f}\n")
        f.write(f"  ▶ RMSE (Test): {res['RMSE_test']:.4f}\n")
        f.write(f"  ▶ R² (Train): {res['R2_train']:.4f}\n")
        f.write(f"  ▶ RMSE (Train): {res['RMSE_train']:.4f}\n")
        f.write("-" * 50 + "\n")