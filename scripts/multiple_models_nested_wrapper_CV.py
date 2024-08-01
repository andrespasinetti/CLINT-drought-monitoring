import numpy as np
import pandas as pd
import os
import multiprocessing
import argparse
import json
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from data_utils import *
from model_utils import *
from CMI import *

def process_data(args):
    fold, test_size, clustering, linkage, threshold, target, features, time = args
    clusters = load_clusters(fold, test_size, clustering, linkage, threshold)
    aggregated_target = get_aggregated_target(target, clusters)
    aggregated_features = get_aggregated_features(features, clusters)

    if time is not None:
        feature_time = get_feature_time(aggregated_target, time)
        aggregated_features = aggregated_features + feature_time

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
    
def process_inner_loop(target, features, model_type, n_splits, fold, cluster, new_feature_ID):
    #print(f"cluster: {cluster}", flush=True)
    #print(f"outer val: {fold}", flush=True)
    #print(f"new feature ID: {new_feature_ID}", flush=True)

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
        #print("train months", flush=True)
        #print(train_months, flush=True)
        #print("test months", flush=True)
        #print(test_months, flush=True)

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

        if X_train.shape[0]==0 or X_test.shape[0]==0:
            continue    
        
        model = get_model(model_type)
        clf = make_pipeline(StandardScaler(), model)
        clf.fit(X_train, y_train)        

        # Predict without NaN values
        y_pred_nonan = clf.predict(X_test)
        y_pred = np.full(len(valid_indices_test), np.nan)
        y_pred[valid_indices_test] = y_pred_nonan

        for index, row in target_pred.iterrows():
            target_pred.loc[index, test_months] = y_pred[index * len(test_months): (index + 1) * len(test_months)]
            
    return fold, cluster, new_feature_ID, target_pred

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--n_splits', type=int, default=1, help='Description of parameter')
    parser.add_argument('--test_size', type=int, default=0, help='Description of parameter')    
    parser.add_argument('--clustering', type=str, default='none', help='Description of parameter')
    parser.add_argument('--linkage', type=str, help='Description of parameter')
    parser.add_argument('--threshold', type=float, help='Description of parameter')
    parser.add_argument('--feature_lag', type=int, choices=[0, 1, 2], default=0, help='Description of parameter')
    parser.add_argument('--time', type=str, choices=['month', 'year', 'month_year'], help='Description of parameter')
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
    use_only_indexes = args.use_only_indexes
    max_num_features = args.max_num_features
    model_type = args.model_type
    num_processes = args.num_processes

    target = get_target()
    features = get_features(feature_lag, target, only_indexes=use_only_indexes)
    if test_size > 0:
        target = target.iloc[:, :-test_size*6]
        for i in range(len(features)):
            features[i] = features[i].iloc[:, :-test_size*6]

    clusters_dict = {}
    aggregated_target_dict = {}
    aggregated_features_dict = {}

    args = [(fold, test_size, clustering, linkage, threshold, target, features, time) for fold in range(n_splits)]

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

    if clustering != 'none':
        filename = f'feature_selection_nested_wrapper_multiple_models_{clustering}_{linkage}_{threshold}{lag_text}{indexes_text}{time_text}'        
    else:
        filename = f'feature_selection_nested_wrapper_multiple_models{lag_text}{indexes_text}{time_text}'        

    print(f'Computing {filename}', flush=True)
    
    num_constant_feature = 0 
    if time == 'year':
        num_constant_feature = 1
    elif time == 'month':
        num_constant_feature = 2
    elif time == 'month_year':
        num_constant_feature = 3
    
    if num_constant_feature>0:
        constant_feature_IDs = list(range(len(aggregated_features_dict[0])))[-num_constant_feature:]
        starting_available_features = list(range(len(aggregated_features_dict[0])))[:-num_constant_feature]
    else:
        constant_feature_IDs = []
        starting_available_features = list(range(len(aggregated_features_dict[0])))
        
    # each outer fold and each cluster has its own selected features
    available_features_dict = {}
    selected_features_dict = {}
    for fold in range(n_splits):
        available_features_dict[fold] = {}
        selected_features_dict[fold] = {}
        for cluster in range(len(clusters_dict[fold])):
            available_features_dict[fold][cluster] = starting_available_features.copy()
            selected_features_dict[fold][cluster] = constant_feature_IDs.copy()

    # Parallel forward wrapper among all folds and all clusters 
    for i in range(max_num_features):
        # 1) For each fold, for each cluster, for each available feature, for each inner_fold, train on inner_train and predict on inner_val
        #   results variable has all the predictions on inner loop
        with multiprocessing.Pool(processes=num_processes) as pool:        
            results_async = pool.starmap_async(
                process_inner_loop,
                [(get_aggregated_target_fold_cluster(aggregated_target_dict[fold], cluster), 
                  get_aggregated_features_fold_cluster(aggregated_features_dict[fold], selected_features_dict[fold][cluster], cluster, new_feature_ID), 
                  model_type, 
                  n_splits,
                  fold,
                  cluster,  
                  new_feature_ID) 
                  for fold in range(n_splits)
                  for cluster in range(len(clusters_dict[fold])) 
                  for new_feature_ID in available_features_dict[fold][cluster]])

            results_async.wait()
            results = results_async.get()

        # 2)For each fold, for each cluster, select the best next feature by averaging inner_val MSE scores 
        MSEs_inner_loop = {}
        for fold in range(n_splits):
            MSEs_inner_loop[fold] = {}
            for cluster in range(len(clusters_dict[fold])):
                MSEs_inner_loop[fold][cluster] = {}
                target_cluster = get_aggregated_target_fold_cluster(aggregated_target_dict[fold], cluster)
                target_cluster = target_cluster.iloc[:, 1:].reset_index(drop=True)
                for new_feature_ID in available_features_dict[fold][cluster]:
                    prediction_inner = next((pred for f, c, n, pred in results if f==fold and c == cluster and n == new_feature_ID), None)
                    prediction_inner = prediction_inner.iloc[:, 1:].reset_index(drop=True)
                    # Check if prediction_inner or target_cluster have all NaN values in any column
                    if prediction_inner.iloc[0].isna().all() or target_cluster[prediction_inner.columns].iloc[0].isna().all():
                        MSE_inner = np.nan
                    else:
                        MSE_inner = np.nanmean((prediction_inner - target_cluster[prediction_inner.columns])**2)
                    MSEs_inner_loop[fold][cluster][new_feature_ID] = MSE_inner

                best_feature = min(MSEs_inner_loop[fold][cluster], key=MSEs_inner_loop[fold][cluster].get)
                selected_features_dict[fold][cluster].append(best_feature)
                available_features_dict[fold][cluster].remove(best_feature)

        print(f'No. selected features: {i+1}', flush=True)


    folder_path = f'../results/training_{17-test_size}/selected_features'
    os.makedirs(folder_path, exist_ok=True)

    with open(f'{folder_path}/{filename}.json', 'w') as f:
        json.dump(selected_features_dict, f)