from utilities import *

method = 'rf'
param = False
random_flag = False
scale_flag = False

g_columns = ['lon_d', 'lat_d']
drop_columns= ['index']

train_df = pd.read_pickle(os.path.join(base_path,'Xtrain')).drop(columns=drop_columns)
val_df = pd.read_pickle(os.path.join(base_path,'Xval')).drop(columns=drop_columns)
train_df = pd.concat([train_df,val_df])

train_df = train_df.drop(train_df[(train_df['lon_d']<-120)&(train_df['lat_d']<30)].index)
train_df = train_df.drop(train_df[(train_df['lon_d']>120)&(train_df['lat_d']<30)].index)
test_df = pd.read_pickle(os.path.join(base_path,'Xtest')).drop(columns=drop_columns)
test_df = test_df.drop(test_df[(test_df['lon_d']<-120)&(test_df['lat_d']<30)].index)
test_df = test_df.drop(test_df[(test_df['lon_d']>120)&(test_df['lat_d']<30)].index)

train_df['y'] = train_df['y'] - train_df['min_altitude_rs']
test_df['y'] = test_df['y'] - test_df['min_altitude_rs']

train_df['group'] = train_df.groupby(g_columns).ngroup()
groups = train_df['group']

y_train = train_df['y']
y_val = val_df['y']
y_test = test_df['y']

train_df = train_df.drop(columns=['time_RS','y']).drop(columns=['group'])
val_df = val_df.drop(columns=['time_RS','y'])
test_df = test_df.drop(columns=['time_RS','y'])
test_min_altitude = test_df['min_altitude_rs']

columns_list =  [['angle','month','power','skew','kurt',\
           'avg_col_0','avg_col_10','avg_col_20','avg_col_30'\
            ,'avg_col_40','avg_col_50',\
        'avg_col_60','avg_col_70','avg_col_80','avg_col_90'\
            ,'avg_col_100'],['lat_d','lon_d','power','skew','kurt','distance_to_coast','min_altitude_rs',\
           'avg_col_0','avg_col_10','avg_col_20','avg_col_30'\
            ,'avg_col_40','avg_col_50',\
        'avg_col_60','avg_col_70','avg_col_80','avg_col_90'\
            ,'avg_col_100'],['power','skew','kurt',\
           'avg_col_0','avg_col_10','avg_col_20','avg_col_30'\
            ,'avg_col_40','avg_col_50',\
        'avg_col_60','avg_col_70','avg_col_80','avg_col_90'\
            ,'avg_col_100'],['angle','month','lat_d','lon_d','distance_to_coast','min_altitude_rs',\
           'avg_col_0','avg_col_10','avg_col_20','avg_col_30'\
            ,'avg_col_40','avg_col_50',\
        'avg_col_60','avg_col_70','avg_col_80','avg_col_90'\
            ,'avg_col_100'],[
           'avg_col_0','avg_col_10','avg_col_20','avg_col_30'\
            ,'avg_col_40','avg_col_50',\
        'avg_col_60','avg_col_70','avg_col_80','avg_col_90'\
            ,'avg_col_100']]

description_list = ['no geographical','no temporal','no contextual','no statistical','nothing']

results = []
for i, columns in enumerate(columns_list):
    # Slice input data
    train_df_subset = train_df[columns].copy()
    val_df_subset = val_df[columns].copy()
    test_df_subset = test_df[columns].copy()

    # Add noise if columns exist
    noise_std = 1.0
    if 'lat_d' in train_df_subset.columns:
        train_df_subset['lat_d'] += np.random.normal(0, noise_std, train_df_subset.shape[0])
    if 'lon_d' in train_df_subset.columns:
        train_df_subset['lon_d'] += np.random.normal(0, noise_std, train_df_subset.shape[0])
    if 'min_altitude_rs' in train_df_subset.columns:
        train_df_subset['min_altitude_rs'] += np.random.normal(0, 0.1, train_df_subset.shape[0])

    print(f"\n▶️ Description: {description_list[i]}")
    print('Train set size:', train_df_subset.shape)

    # Train model
    model = RandomForestRegressor(
        random_state=42, max_features=1.0, verbose=1,
        max_depth=43, n_estimators=850,
        min_samples_leaf=4, min_samples_split=2,
        n_jobs=-1, bootstrap=True
    )

    model.fit(train_df_subset, y_train)

    # --- Evaluate on TEST set ---
    y_pred_test = model.predict(test_df_subset)
    mse_test = mean_squared_error(y_test - test_min_altitude, y_pred_test - test_min_altitude)
    r2_test = r2_score(y_test - test_min_altitude, y_pred_test - test_min_altitude)
    rmse_test = math.sqrt(mse_test)

    # --- Evaluate on TRAIN set ---
    y_pred_train = model.predict(train_df_subset)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = math.sqrt(mse_train)

    # Save model
    with open(f'model_{i}.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Store result
    results.append({
        'description': description_list[i],
        'R2_test': r2_test,
        'RMSE_test': rmse_test,
        'R2_train': r2_train,
        'RMSE_train': rmse_train
    })

    print(f"Test - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}")
    print(f"Train - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}")

    # Optional: save predictions
    test_df_eval = pd.read_pickle(os.path.join(base_path, 'Xtest'))
    test_df_eval = test_df_eval.drop(test_df_eval[(test_df_eval['lon_d'] < -120) & (test_df_eval['lat_d'] < 30)].index)
    test_df_eval = test_df_eval.drop(test_df_eval[(test_df_eval['lon_d'] > 120) & (test_df_eval['lat_d'] < 30)].index)
    test_df_eval['pred'] = y_pred_test
    test_df_eval['og'] = y_test
    test_df_eval.to_pickle(f'predictions_noclouds_{i}.pkl')

    train_df_subset['pred'] = y_pred_train
    train_df_subset['og'] = y_train
    train_df_subset.to_pickle(f'predictions_train_{i}.pkl')

# Save results to a text file
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