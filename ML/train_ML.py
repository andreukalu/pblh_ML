from utilities import *
print('hello')
method = 'rf'
param = False
random_flag = False
scale_flag = False

g_columns = ['lon_d', 'lat_d']
drop_columns= ['index']

train_df = pd.read_pickle(os.path.join(base_path,'Xtrain')).drop(columns=drop_columns)
val_df = pd.read_pickle(os.path.join(base_path,'Xval')).drop(columns=drop_columns)
train_df = pd.concat([train_df,val_df])
# train_df = train_df[train_df['cloud_flag']==0]

train_df = train_df.drop(train_df[(train_df['lon_d']<-120)&(train_df['lat_d']<30)].index)
train_df = train_df.drop(train_df[(train_df['lon_d']>120)&(train_df['lat_d']<30)].index)

test_df = pd.read_pickle(os.path.join(base_path,'Xtest')).drop(columns=drop_columns)
test_df = test_df.drop(test_df[(test_df['lon_d']<-120)&(test_df['lat_d']<30)].index)
test_df = test_df.drop(test_df[(test_df['lon_d']>120)&(test_df['lat_d']<30)].index)

# train_df['y'] = train_df['y'] - train_df['min_altitude_rs']
# test_df['y'] = test_df['y'] - test_df['min_altitude_rs']
# test_df = test_df[test_df['cloud_flag']==0]
train_df['group'] = train_df.groupby(g_columns).ngroup()
groups = train_df['group']

# threshold = 0.7
# high_values = train_df[train_df['y'] > threshold]  # Select high-value samples
# low_values = train_df[train_df['y'] <= threshold]

# high_values_upsampled = resample(high_values, replace=True, n_samples=len(low_values))
# train_df = pd.concat([low_values, high_values_upsampled])

y_train = train_df['y']
y_val = val_df['y']
y_test = test_df['y']

train_df = train_df.drop(columns=['time_RS','y']).drop(columns=['group'])
val_df = val_df.drop(columns=['time_RS','y'])
test_df = test_df.drop(columns=['time_RS','y'])

columns = ['angle','month','lat_d','lon_d','power','skew','kurt','distance_to_coast','min_altitude_rs',\
           'avg_col_0',\
            'avg_col_1','avg_col_2','avg_col_3','avg_col_4','avg_col_5',\
            'avg_col_6','avg_col_7','avg_col_8','avg_col_9','avg_col_11',\
            'avg_col_12','avg_col_13','avg_col_14','avg_col_16','avg_col_17',\
            'avg_col_18','avg_col_19',\
            'avg_col_10','avg_col_15','avg_col_20','avg_col_30'\
            ,'avg_col_40','avg_col_50',\
        'avg_col_60','avg_col_70','avg_col_80','avg_col_90'\
            ,'avg_col_100','avg_col_110','avg_col_120','avg_col_130'\
             ,'avg_col_140'\
             ,'avg_col_150','avg_col_160','avg_col_170']

# columns = ['avg_col_0','avg_col_10','avg_col_20','avg_col_30'\
#             ,'avg_col_40','avg_col_50',\
#         'avg_col_60','avg_col_70','avg_col_80','avg_col_90'\
#             ,'avg_col_100','avg_col_110','avg_col_120','avg_col_130'\
#             ,'avg_col_140'\
#             ,'avg_col_150','avg_col_160','avg_col_170']
# columns = train_df.columns
train_df = train_df[columns]
val_df = val_df[columns]
test_df = test_df[columns]

noise_std = 1.0
# train_df['angle_1'] = train_df['angle_1'] + np.random.normal(0, noise_std, train_df.shape[0])
# train_df['angle_2'] = train_df['angle_2'] + np.random.normal(0, noise_std, train_df.shape[0])
train_df['angle'] = train_df['angle'] + np.random.normal(0, noise_std, train_df.shape[0])
train_df['lat_d'] = train_df['lat_d'] + np.random.normal(0, noise_std, train_df.shape[0])
train_df['lon_d'] = train_df['lon_d'] + np.random.normal(0, noise_std, train_df.shape[0])
# train_df['min_altitude_rs'] = train_df['min_altitude_rs'] + np.random.normal(0, 0.1, train_df.shape[0])
print('train set size: ' + str(train_df.shape))
print(train_df.columns)
print(train_df.head(5))
# train_df['lon_d'] = train_df['lon_d'] + np.random.normal(0, noise_std, train_df.shape[0])
# train_df = train_df[['angle','lat_d','lon_d','month']]
# val_df = val_df[['angle','lat_d','lon_d','month']]
# test_df = test_df[['angle','lat_d','lon_d','month']]

if scale_flag == True:
    # 1. Initialize the StandardScaler
    scaler = StandardScaler()

    # 2. Fit the scaler on the training data and transform the training data
    train_df = scaler.fit_transform(train_df)

    # 3. Transform the test data using the same scaler (without fitting it again)
    test_df = scaler.transform(test_df)

if param == False:
    if method == 'lr':
        model = LinearRegression()
    elif method == 'ridge':
        model = Ridge()
    elif method == 'rf':
        model = RandomForestRegressor(random_state=42,max_features=1.0,verbose=1,max_depth=43,n_estimators=850,min_samples_leaf=4,min_samples_split=2,n_jobs=-1,bootstrap=True)
    elif method == 'gb':
        model = GradientBoostingRegressor(random_state=42,loss='huber',learning_rate=0.01,max_features=0.45,n_estimators=440,verbose=1,max_depth=40,min_samples_leaf=5,min_samples_split=2)
    elif method == 'lr':
        model = LinearRegression()
    elif method == 'sr':
        rf = RandomForestRegressor(random_state=42,max_features=None,max_depth=11,n_estimators=90,min_samples_leaf=5,min_samples_split=2,n_jobs=-1,bootstrap=True)
        gb = GradientBoostingRegressor(random_state=42,loss='squared_error',learning_rate=0.05,max_features=None,n_estimators=68,verbose=1,max_depth=11,min_samples_leaf=28,min_samples_split=2)
        nn = MLPRegressor(max_iter=100000,verbose=False,early_stopping=True,tol=1e-5,n_iter_no_change=500,activation='tanh',alpha=0.1,hidden_layer_sizes=(32,16))
        estimators = [
            ('rf', rf),
            ('gb', gb)
        ]
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=gb)
        
elif param == True:
    if method == 'rf':
        model = RandomForestRegressor(random_state=42)

        # Set up the parameter grid. R^2=0.5812
        param_grid = { # BEST PARAMETERS FOUND {'bootstrap': True, 'criterion': 'friedman_mse', 'max_depth': 19, 'max_features': 0.4, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 950}
            #{'n_estimators': 640, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 0.1, 'max_depth': 37, 'criterion': 'friedman_mse', 'bootstrap': True}
            'n_estimators': np.arange(50,1000,10).tolist(),        
            'max_features': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
            'max_depth':  np.arange(2,50,2).tolist(),          
            'min_samples_split': np.arange(2,20,1).tolist(),         
            'min_samples_leaf': np.arange(2,20,1).tolist(),    
        }

    if method == 'gb': 
        model = GradientBoostingRegressor(random_state=42)

        param_grid = { #Best Parameters:  {'learning_rate': 0.03, 'loss': 'huber', 'max_depth': 16, 'max_features': 0.4, 'min_samples_leaf': 6, 'min_samples_split': 2, 'n_estimators': 390}
            'loss': ['huber'],
            'max_features': [0.4,0.75,1.0],
            'learning_rate': [0.03,0.01,0.05,0.1],            
            'n_estimators': np.arange(50,200,50).tolist(),           
            'min_samples_split': [2],           
            'max_depth': np.arange(5,100,10).tolist(),#27,             
            'min_samples_leaf': np.arange(2,50,10).tolist(),             
        }

    if method == 'sr':
        rf = RandomForestRegressor(random_state=42,max_features=None,max_depth=28,n_estimators=280,min_samples_leaf=5,min_samples_split=2,n_jobs=-1,bootstrap=True)
        gb = GradientBoostingRegressor(random_state=42,learning_rate=0.05,max_features=None,n_estimators=68,verbose=1,max_depth=11,min_samples_leaf=28)
        estimators = [
            ('rf', rf),
            ('gb', gb)
        ]
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=GradientBoostingRegressor(n_estimators=100,random_state=42))
        
        param_grid = { #Best Parameters: {'learning_rate': 0.05, 'loss': 'squared_error', 'max_depth': 11, 'max_features': None, 'min_samples_leaf': 28, 'min_samples_split': 2, 'n_estimators': 68}
                'final_estimator__loss': ['squared_error'],
                'final_estimator__max_features': [None],
                'final_estimator__learning_rate': [0.05],            
                'final_estimator__n_estimators': [40,60],           
                'final_estimator__min_samples_split': [2],           
                'final_estimator__max_depth': [5,7],             
                'final_estimator__min_samples_leaf': [10,30],             
            }
    elif method == 'NN':
        model = MLPRegressor(max_iter=100000,verbose=False,early_stopping=True,tol=1e-5,n_iter_no_change=500)
        param_grid = {
            'activation': ['tanh','relu'],
            'solver': ['adam'],
            'learning_rate': ['adaptive'],
            'hidden_layer_sizes': [(16,16),(16,32),(32,16),(32,32),(64,64),(64,32),(32,64),(128,32),(128,64),(256,128),(32,32,16),(64,32,16)],
            'alpha': [0.1,0.01,0.001,0.0001,0.00001,0.0000,1],
        }

    scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']
    group_kfold = GroupKFold(n_splits=5)

    # Set up the grid search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=group_kfold, n_jobs=-1, verbose=3, scoring=scoring, return_train_score=True, refit='neg_mean_squared_error')
    if random_flag == 1:
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=100,  
            scoring=scoring, 
            return_train_score=True,
            cv=group_kfold, 
            refit='neg_mean_squared_error',
            verbose=3,  
            random_state=42,
            n_jobs=-1  
        )
    
    grid_search.fit(train_df, y_train, groups = groups)

    results = grid_search.cv_results_

    # Get the best parameters and the best estimator from the grid search
    print("Best Parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_

    # Open file for writing
    with open(f"{method}_grid_search_iterations.txt", "w") as f:
        for i in range(len(results["params"])):
            params = results["params"][i]
            mean_test_rmse = np.sqrt(-results["mean_test_neg_mean_squared_error"][i])
            std_test_rmse = np.sqrt(results["std_test_neg_mean_squared_error"][i])
            mean_test_r2 = results["mean_test_r2"][i]
            std_test_r2 = results["std_test_r2"][i]

            mean_train_rmse = -results["mean_train_neg_mean_squared_error"][i]
            mean_train_r2 = results["mean_train_r2"][i]

            # Create a readable block of results
            result_block = (
                f"Iteration {i+1}:\n"
                f"Parameters: {json.dumps(params)}\n"
                f"Mean Test RMSE: {mean_test_rmse:.4f} ± {std_test_rmse:.4f}\n"
                f"Mean Test R²: {mean_test_r2:.3f} ± {std_test_r2:.3f}\n"
                f"Mean Train RMSE: {mean_train_rmse:.4f}\n"
                f"Mean Train R²: {mean_train_r2:.3f}\n"
                + "-"*50 + "\n"
            )

            # Print to console
            print(result_block)

            # Save to file
            f.write(result_block)

model.fit(train_df,y_train)


# Make predictions on the test set
y_pred = model.predict(test_df)

# Evaluate the performance using Mean Squared Error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

x = y_test
y = y_pred
test_df = pd.read_pickle(os.path.join(base_path,'Xtest'))
test_df = test_df.drop(test_df[(test_df['lon_d']<-120)&(test_df['lat_d']<30)].index)
test_df = test_df.drop(test_df[(test_df['lon_d']>120)&(test_df['lat_d']<30)].index)
# test_df = test_df[test_df['cloud_flag']==0]
# test_df = test_df[test_df['angle']<90]
# train_df = train_df[['angle','lat_d','lon_d','month']]
# val_df = val_df[['angle','lat_d','lon_d','month']]
# test_df = test_df[['angle','lat_d','lon_d','month']]

test_df['pred'] = y_pred
test_df['og'] = y_test
test_df.to_pickle('predictions')
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# # Make predictions on the test set
y_pred = model.predict(train_df)

train_df['pred'] = y_pred
train_df['og'] = y_train
train_df.to_pickle('predictions_train')
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

if method == 'rf':
    # Retrieve and sort feature importances
    feature_importance = model.feature_importances_

    print(feature_importance.shape)
    print(test_df.columns.shape)
    # Combine column names and importance into a DataFrame
    importance_df = pd.DataFrame({
        'Feature': test_df[columns].columns,
        'Importance': feature_importance
    })

    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Display sorted feature importances
    print(importance_df.head(20))

    importance_df.to_pickle('importances')
