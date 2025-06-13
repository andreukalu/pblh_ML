from utilities_ML import *

# === Define paths and parameters ===
pickles_path = ''  # Path to data used for training and test

# Define flags for model configuration
method = 'rf'           # Machine Learning model type: 'rf' (Random Forest) or 'gb' (Gradient Boosting)
param = False           # If True, enables hyperparameter tuning
random_flag = False     # If True, uses RandomizedSearchCV instead of GridSearchCV

# === Load training, validation, and test data ===
train_df = pd.read_pickle(os.path.join(pickles_path, 'Xtrain')).drop(columns=['index'])
val_df = pd.read_pickle(os.path.join(pickles_path, 'Xval')).drop(columns=['index'])
train_df = pd.concat([train_df, val_df])  # Combine train and validation sets
test_df = pd.read_pickle(os.path.join(pickles_path, 'Xtest')).drop(columns=['index'])

# === Remove outlier data based on geographical coordinates ===
# (Filtering out data in specific longitude-latitude zones)
train_df = train_df.drop(train_df[(train_df['lon_d'] < -120) & (train_df['lat_d'] < 30)].index)
train_df = train_df.drop(train_df[(train_df['lon_d'] > 120) & (train_df['lat_d'] < 30)].index)
test_df = test_df.drop(test_df[(test_df['lon_d'] < -120) & (test_df['lat_d'] < 30)].index)
test_df = test_df.drop(test_df[(test_df['lon_d'] > 120) & (test_df['lat_d'] < 30)].index)

# === Group data to prevent data leakage between train and validation folds ===
train_df['group'] = train_df.groupby(['lon_d', 'lat_d']).ngroup()
groups = train_df['group']

# === Extract labels (target variable) ===
y_train = train_df['y']
y_test = test_df['y']

# === Remove non-feature columns ===
train_df = train_df.drop(columns=['time_RS', 'y', 'group'])
test_df = test_df.drop(columns=['time_RS', 'y'])

# === Define feature columns ===
columns = ['angle', 'month', 'lat_d', 'lon_d', 'power', 'skew', 'kurt', 'distance_to_coast', 'min_altitude_rs',
           'avg_col_0', 'avg_col_10', 'avg_col_15', 'avg_col_20', 'avg_col_30', 'avg_col_40', 'avg_col_50',
           'avg_col_60', 'avg_col_70', 'avg_col_80', 'avg_col_90', 'avg_col_100', 'avg_col_110', 'avg_col_120',
           'avg_col_130', 'avg_col_140', 'avg_col_150', 'avg_col_160', 'avg_col_170']

# Select only feature columns
train_df = train_df[columns]
test_df = test_df[columns]

# === Define and configure ML model ===
if not param:  # No hyperparameter tuning
    if method == 'rf':
        # Random Forest with predefined hyperparameters
        model = RandomForestRegressor(
            max_features=1.0, verbose=1, max_depth=43, n_estimators=850,
            min_samples_leaf=4, min_samples_split=2, n_jobs=-1, bootstrap=True
        )
    elif method == 'gb':
        # Gradient Boosting with predefined hyperparameters
        model = GradientBoostingRegressor(
            loss='huber', learning_rate=0.01, max_features=0.45, n_estimators=440,
            verbose=1, max_depth=40, min_samples_leaf=5, min_samples_split=2
        )

# === Hyperparameter tuning ===
elif param:
    # Base model definition (default settings)
    if method == 'rf':
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': np.arange(50, 1000, 10).tolist(),
            'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'max_depth': np.arange(2, 50, 2).tolist(),
            'min_samples_split': np.arange(2, 20, 1).tolist(),
            'min_samples_leaf': np.arange(2, 20, 1).tolist(),
        }

    elif method == 'gb':
        model = GradientBoostingRegressor()
        param_grid = {
            'loss': ['huber'],
            'max_features': [0.4, 0.75, 1.0],
            'learning_rate': [0.03, 0.01, 0.05, 0.1],
            'n_estimators': np.arange(50, 200, 50).tolist(),
            'min_samples_split': [2],
            'max_depth': np.arange(5, 100, 10).tolist(),
            'min_samples_leaf': np.arange(2, 50, 10).tolist(),
        }

    # Define evaluation metrics and cross-validation strategy
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    group_kfold = GroupKFold(n_splits=5)

    # Use GridSearchCV or RandomizedSearchCV
    if random_flag:
        grid_search = RandomizedSearchCV(
            estimator=model, param_distributions=param_grid, n_iter=100,
            scoring=scoring, return_train_score=True, cv=group_kfold,
            refit='neg_mean_squared_error', verbose=3, random_state=42, n_jobs=-1
        )
    else:
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=group_kfold,
            n_jobs=-1, verbose=3, scoring=scoring, return_train_score=True,
            refit='neg_mean_squared_error'
        )

    # Perform hyperparameter search
    grid_search.fit(train_df, y_train, groups=groups)
    results = grid_search.cv_results_

    # Output best parameters
    print("Best Parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_

    # Save search results to file
    with open(f"{method}_grid_search_iterations.txt", "w") as f:
        for i in range(len(results["params"])):
            params = results["params"][i]
            mean_test_rmse = np.sqrt(-results["mean_test_neg_mean_squared_error"][i])
            std_test_rmse = np.sqrt(results["std_test_neg_mean_squared_error"][i])
            mean_test_r2 = results["mean_test_r2"][i]
            std_test_r2 = results["std_test_r2"][i]
            mean_train_rmse = -results["mean_train_neg_mean_squared_error"][i]
            mean_train_r2 = results["mean_train_r2"][i]

            result_block = (
                f"Iteration {i+1}:\n"
                f"Parameters: {json.dumps(params)}\n"
                f"Mean Test RMSE: {mean_test_rmse:.4f} ± {std_test_rmse:.4f}\n"
                f"Mean Test R²: {mean_test_r2:.3f} ± {std_test_r2:.3f}\n"
                f"Mean Train RMSE: {mean_train_rmse:.4f}\n"
                f"Mean Train R²: {mean_train_r2:.3f}\n"
                + "-"*50 + "\n"
            )

            print(result_block)
            f.write(result_block)

# === Train final model on full training data ===
model.fit(train_df, y_train)

# === Evaluate model on test set ===
y_pred = model.predict(test_df) - test_df['min_altitude_rs']
y_test = y_test - test_df['min_altitude_rs']
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# === Save predictions for test data ===
test_df = pd.read_pickle(os.path.join(pickles_path, 'Xtest'))
test_df = test_df.drop(test_df[(test_df['lon_d'] < -120) & (test_df['lat_d'] < 30)].index)
test_df = test_df.drop(test_df[(test_df['lon_d'] > 120) & (test_df['lat_d'] < 30)].index)
test_df['pred'] = y_pred
test_df['og'] = y_test
test_df.to_pickle('predictions')

# === Evaluate and save predictions for training data ===
y_pred = model.predict(train_df)
train_df['pred'] = y_pred
train_df['og'] = y_train
train_df.to_pickle('predictions_train')

mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# === Save the final trained model ===
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)