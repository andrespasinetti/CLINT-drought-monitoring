import numpy as np
import pandas as pd
import os
import multiprocessing
import argparse
import json
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping
import logging

from data_utils import *
from model_utils import *
from CMI_firsttime import *

def process_data(args):
    fold, test_size, clustering, linkage, threshold, target, features, time, use_coordinates = args
    clusters = load_clusters(fold, test_size, clustering, linkage, threshold)
    aggregated_target = get_aggregated_target(target, clusters)
    aggregated_features = get_aggregated_features(features, clusters)

    if time is not None:
        feature_time = get_feature_time(aggregated_target, time)
        aggregated_features = aggregated_features + feature_time

    if use_coordinates:
        features_coordinates = get_aggregated_coordinates(clusters, aggregated_features[0])
        aggregated_features = aggregated_features + features_coordinates

    return fold, clusters, aggregated_target, aggregated_features

def get_aggregated_target_fold_cluster(aggregated_target, cluster):
    target_cluster = aggregated_target[aggregated_target['CLUSTER']==cluster].reset_index(drop=True)
    return target_cluster

def get_aggregated_features_fold_cluster(aggregated_features, selected_features, cluster, new_feature_ID=None):
    features_cluster = []
    for i in range(len(aggregated_features)):
        if i in selected_features:
            feature = aggregated_features[i]
            features_cluster.append(feature[feature['CLUSTER']==cluster].reset_index(drop=True))
    
    if new_feature_ID is not None:
        new_feature = aggregated_features[new_feature_ID][aggregated_features[new_feature_ID]['CLUSTER']==cluster]
        return features_cluster + [new_feature]
    else:
        return features_cluster
    
def process_inner_loop(target, features, model_type, n_splits, fold, new_feature_ID):
    target_pred = pd.DataFrame(np.nan, index=range(len(target)), columns=target.columns)
    target_pred['CLUSTER'] = target['CLUSTER']
    
    all_months = list(target.columns[1:])
    # Exclude data in fold
    fold_size = 6
    start_excluded = fold * fold_size
    end_excluded = start_excluded + fold_size 
    inner_loop_months = all_months[:start_excluded] + all_months[end_excluded:]

    kf = KFold(n_splits=n_splits-1)
    for train_index, test_index in kf.split(inner_loop_months):
        train_months = [inner_loop_months[i] for i in train_index]
        test_months = [inner_loop_months[i] for i in test_index]

        target_train = target[train_months]
        target_test = target[test_months]

        features_train = [feature[train_months] for feature in features]
        features_test = [feature[test_months] for feature in features]

        y_train = target_train.values.flatten()
        y_test = target_test.values.flatten()

        X_train_list = [feature.values.flatten() for feature in features_train] 
        X_test_list = [feature.values.flatten() for feature in features_test] 

        X_train = np.column_stack(X_train_list)
        X_test = np.column_stack(X_test_list)

        def drop_nan(X, y):
            valid_indices = ~pd.isna(y) & ~np.any(pd.isna(X), axis=1)
            y = y[valid_indices]
            X = X[valid_indices]
            return X, y, valid_indices

        X_train, y_train, _ = drop_nan(X_train, y_train)

        X_train, y_train = shuffle(X_train, y_train, random_state=1)
        X_test, y_test, valid_indices_test = drop_nan(X_test, y_test)
        
        pattern = r'\((.*?)\)(d\d*\.*\d*)?'  # Updated pattern to capture dropout rate info
        tuple_match = re.search(pattern, model_type)

        if tuple_match:
            tuple_str = tuple_match.group(1)
            dropout_str = tuple_match.group(2)
            hidden_layer_sizes = ast.literal_eval(tuple_str)
            dropout_rate = float(dropout_str[1:]) if dropout_str else 0.0  # Extract dropout rate if available
        else:
            raise ValueError("Invalid MLPR architecture.")
        
        model = build_NN(hidden_layer_sizes=hidden_layer_sizes, dropout_rate=dropout_rate, input_dim=X_train.shape[1])
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])

        # Configure Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Fit the model with Early Stopping
        pipeline.fit(X_train, y_train, model__epochs=30, model__batch_size=32, model__verbose=0, model__validation_split=0.2, model__callbacks=[early_stopping])
        
        # Predict without NaN values
        y_pred_nonan = pipeline.predict(X_test, verbose=0).flatten()
        y_pred = np.full(len(valid_indices_test), np.nan)
        y_pred[valid_indices_test] = y_pred_nonan

    #print(f"process_inner_loop: fold: {fold}, new_feature: {new_feature_ID}", flush=True)
    return fold, new_feature_ID, y_pred, test_months

# 2)For each fold, select the best next feature by averaging inner_val MSE scores 
def process_feature_combination(args):
    outer_fold, new_feature_ID, aggregated_target, results = args
    prediction_inner = aggregated_target.copy()
    prediction_inner.iloc[:, 1:] = np.nan
    
    # Filter results specific to this outer_fold and new_feature_ID
    new_feature_results = [(y_pred, test_months) for outer_f, new_f, y_pred, test_months in results 
                        if outer_fold == outer_f and new_feature_ID == new_f]
    
    # Populate the prediction_inner DataFrame
    for y_pred, test_months in new_feature_results:
        for index, row in prediction_inner.iterrows():
            prediction_inner.loc[index, test_months] = y_pred[index * len(test_months): (index + 1) * len(test_months)]
    
    return outer_fold, new_feature_ID, prediction_inner

def process_eval_inner_loop(aggregated_target, predictions, clusters, outer_fold, new_feature_ID): 
    MSE_values = []
    for cluster in range(len(clusters)):
        cluster_pred = predictions[predictions['CLUSTER']==cluster].iloc[:, 1:].reset_index(drop=True)
        cluster_target = aggregated_target[aggregated_target['CLUSTER']==cluster].iloc[:, 1:].reset_index(drop=True)
        
        # Check if cluster_pred or cluster_target have all NaN values in any column
        if cluster_pred.iloc[0].isna().all() or cluster_target[cluster_pred.columns].iloc[0].isna().all():
            MSE = np.nan
        else:
            # Proceed with calculations if there are not all NaN values
            MSE = np.nanmean((cluster_pred - cluster_target[cluster_pred.columns])**2)

        MSE_values.append(MSE)
    MSE_avg = np.nanmean(MSE_values)
    return outer_fold, new_feature_ID, MSE_avg        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--n_splits', type=int, default=1, help='Description of parameter')
    parser.add_argument('--test_size', type=int, default=0, help='Description of parameter')    
    parser.add_argument('--clustering', type=str, default='none', help='Description of parameter')
    parser.add_argument('--linkage', type=str, help='Description of parameter')
    parser.add_argument('--threshold', type=float, help='Description of parameter')
    parser.add_argument('--feature_lag', type=int, choices=[0, 1, 2], default=0, help='Description of parameter')
    parser.add_argument('--time', type=str, choices=['month', 'year', 'month_year'], help='Description of parameter')
    parser.add_argument('--use_coordinates', type=bool, default=False)
    parser.add_argument('--starting_features_dict', type=str)
    parser.add_argument('--use_only_indexes', type=bool, default=False)
    parser.add_argument('--max_num_features', type=int)
    parser.add_argument('--model_type', type=str, help='Description of parameter')
    parser.add_argument('--num_processes', type=int, default=multiprocessing.cpu_count(),
                            help='Number of processes to use. Default is the maximum available processors.')

    args = parser.parse_args()

    n_splits = args.n_splits
    test_size = args.test_size
    clustering = args.clustering
    linkage = args.linkage
    threshold = args.threshold
    feature_lag = args.feature_lag
    time = args.time
    use_coordinates = args.use_coordinates
    use_only_indexes = args.use_only_indexes
    starting_features_dict = args.starting_features_dict
    max_num_features = args.max_num_features
    model_type = args.model_type
    num_processes = args.num_processes

    target = get_target()
    features = get_features(feature_lag, target, only_indexes=use_only_indexes)
    if test_size > 0:
        target = target.iloc[:, :-test_size*6]
        for i in range(len(features)):
            features[i] = features[i].iloc[:, :-test_size*6]

    outer_folds = range(n_splits)
    clusters_dict = {}
    aggregated_target_dict = {}
    aggregated_features_dict = {}

    args = [(fold, test_size, clustering, linkage, threshold, target, features, time, use_coordinates) for fold in outer_folds]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_data, args)

    for result in results:
        fold, clusters, aggregated_target, aggregated_features = result
        clusters_dict[fold] = clusters
        aggregated_target_dict[fold] = aggregated_target
        aggregated_features_dict[fold] = aggregated_features

    lag_text = '' if feature_lag == 0 else f'_lag{feature_lag}'
    indexes_text = '_indexes' if use_only_indexes else ''
    time_text = '' if time is None else '_' + time
    coord_text = '_coord' if use_coordinates else ''

    if clustering != 'none':
        filename = f'feature_selection_nested_wrapper_single_model_{clustering}_{linkage}_{threshold}{lag_text}{indexes_text}{time_text}{coord_text}'        
    else:
        filename = f'feature_selection_nested_wrapper_single_model{lag_text}{indexes_text}{time_text}{coord_text}'        

    print(f'Computing {filename}', flush=True)

    folder_path = f'../results/training_{17-test_size}/selected_features'
    os.makedirs(folder_path, exist_ok=True)

    # each outer fold and each cluster has its own selected features
    available_features_dict = {}
    selected_features_dict = {}
    
    if starting_features_dict:
        with open(f'{folder_path}/{starting_features_dict}.json', 'r') as f:
            starting_features =  json.load(f)

        for fold in outer_folds:
                available_features_dict[fold] = [feature_id for feature_id in list(range(len(aggregated_features_dict[0]))) if feature_id not in starting_features[str(fold)]]
                selected_features_dict[fold] = starting_features[str(fold)]
                print(f'initial features for fold {fold}: {selected_features_dict[fold]}', flush=True)
    else:
        num_constant_feature = 0 
        if time == 'year':
            num_constant_feature = 1
        elif time == 'month':
            num_constant_feature = 2
        elif time == 'month_year':
            num_constant_feature = 3
        
        if use_coordinates:
            num_constant_feature += 3

        if num_constant_feature>0:
            constant_feature_IDs = list(range(len(aggregated_features_dict[0])))[-num_constant_feature:]
            starting_available_features = list(range(len(aggregated_features_dict[0])))[:-num_constant_feature]
        else:
            constant_feature_IDs = []
            starting_available_features = list(range(len(aggregated_features_dict[0])))

        for fold in outer_folds:
            available_features_dict[fold] = starting_available_features.copy()
            selected_features_dict[fold] = constant_feature_IDs.copy()
            print(f'initial features for fold {fold}: {selected_features_dict[fold]}', flush=True)

    # Parallel forward wrapper among all folds 
    for i in range(max_num_features):
        print(f"Starting iteration {i+1} of feature selection", flush=True)
        # 1) For each fold, for each available feature, train on inner_train and predict on inner_val
        #   results variable has all the predictions on inner loop

        # You could parallelize also the inner folds
        with multiprocessing.Pool(processes=num_processes//2) as pool:        
            results_async = pool.starmap_async(
                process_inner_loop,
                [(aggregated_target_dict[fold], 
                  [feature for i, feature in enumerate(aggregated_features_dict[fold]) if i in selected_features_dict[fold] + [new_feature_ID]], 
                  model_type, 
                  n_splits,
                  fold,
                  new_feature_ID) 
                  for fold in outer_folds
                  for new_feature_ID in available_features_dict[fold]])

            results_async.wait()
            results = results_async.get()
        
        print(f"Completed iteration {i+1} of process_inner_loop", flush=True)

        # Create list of arguments for each combination of outer_fold and new_feature_ID
        task_args = [(outer_fold, new_feature_ID, aggregated_target_dict[outer_fold], results)
                    for outer_fold in outer_folds
                    for new_feature_ID in available_features_dict[outer_fold]]
        
        # Create a pool of processes and map the function to the arguments
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_feature_combination, task_args)

        print(f"Completed iteration {i+1} of process_feature_combination", flush=True)

        # Organize results back into a dictionary
        predictions_inner_dict = {}
        for outer_fold, new_feature_ID, prediction_inner in results:
            if outer_fold not in predictions_inner_dict:
                predictions_inner_dict[outer_fold] = {}
            predictions_inner_dict[outer_fold][new_feature_ID] = prediction_inner

        with multiprocessing.Pool(processes=num_processes) as pool: 
            results_async = pool.starmap_async(
                process_eval_inner_loop,
                [(aggregated_target_dict[outer_fold], 
                  predictions_inner_dict[outer_fold][new_feature_ID], 
                  clusters_dict[outer_fold], 
                  outer_fold, 
                  new_feature_ID) 
                 for outer_fold in outer_folds
                 for new_feature_ID in available_features_dict[outer_fold]])

            results_async.wait()
            results = results_async.get()

        print(f"Completed iteration {i+1} of process_eval_inner_loop", flush=True)

        MSEs_inner_loop = {}
        for outer_fold in outer_folds:
            MSEs_inner_loop[outer_fold] = {}
        
        for outer_fold, new_feature_ID, MSE_avg in results:
            MSEs_inner_loop[outer_fold][new_feature_ID] = MSE_avg

        for outer_fold in outer_folds:
            best_feature = min(MSEs_inner_loop[outer_fold], key=MSEs_inner_loop[outer_fold].get)
            selected_features_dict[outer_fold].append(best_feature)
            available_features_dict[outer_fold].remove(best_feature)

        print(f'No. selected features: {i+1}', flush=True)
        print(selected_features_dict, flush=True)
        print()



    with open(f'{folder_path}/{filename}.json', 'w') as f:
        json.dump(selected_features_dict, f)