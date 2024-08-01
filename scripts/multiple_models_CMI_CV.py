import numpy as np
import pandas as pd
import os
import argparse
import multiprocessing
from sklearn.model_selection import KFold
import json

from data_utils import *
from model_utils import *
from scripts.CMI_firsttime import *

def process_data(args):
    fold, test_size, clustering, linkage, threshold, target, features, time = args
    clusters = load_clusters(fold, test_size, clustering, linkage, threshold)
    aggregated_target = get_aggregated_target(target, clusters)
    aggregated_features = get_aggregated_features(features, clusters)

    if time is not None:
        feature_time = get_feature_time(aggregated_target, time)
        aggregated_features = aggregated_features + feature_time

    return fold, clusters, aggregated_target, aggregated_features

def process_fold_cluster_CMI(fold, n_splits, cluster, features, target, k, CMI_threshold, CMI_n_features, time):        
    all_months = list(target.columns[1:])
    kf = KFold(n_splits=n_splits)
    train_index, test_index = list(kf.split(all_months))[fold]
    train_months = [all_months[i] for i in train_index]
    target_train = target[train_months].iloc[0, :]
    features_train = [feature[train_months].iloc[0, :] for feature in features]

    target_train = target_train.dropna()

    # Some features might have all NA:
    # we give as input to the algorithm only aggregated features which have at least 1 observation, but we have also aggregated features with all nan 
    # we track them in order to correctly define the dictionary of selected features wrt the aggregated features
    features_temp = []
    mapping = {}
    no_nan_count = 0
    feature_count = 0
    for i, feature in enumerate(features_train):
        if not feature.isna().all().all():
            features_temp.append(feature.dropna())
            mapping[no_nan_count] = feature_count
            no_nan_count += 1
        feature_count += 1    
    
    features_train = features_temp
    features_target = features_train + [target_train]
    indexes_list = [s.index for s in features_target]
    common_indexes = set(indexes_list[0]).intersection(*indexes_list[1:])
    
    if len(common_indexes)==0:
        return fold, cluster, []
    
    common_indexes_list = sorted(list(common_indexes))
    #print(common_indexes_list, flush=True)
    
    features_train = [feature[common_indexes_list] for feature in features_train]
    features_train_arr = np.array(features_train).T
    target_train = target_train[common_indexes_list]
    target_train_arr = np.array(target_train) 

    selected_features_train = forwardFeatureSelection(features_train_arr, target_train_arr, k, CMI_threshold, CMI_n_features, time, nproc=1, verbose=False)
    selected_features = [mapping[sel] for sel in selected_features_train]

    print(f'Selected features for fold {fold}, cluster {cluster}: {selected_features}', flush=True)
    return fold, cluster, selected_features

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
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--CMI_threshold', type=float, default=0.0)
    parser.add_argument('--CMI_n_features', type=int, default=-1)
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
    k = args.k
    CMI_threshold = args.CMI_threshold
    CMI_n_features = args.CMI_n_features
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
    threshold_text = f'_t{CMI_threshold}' if CMI_n_features==-1 else ''
    if clustering != 'none':
        filename = f'feature_selection_CMIk{k}{threshold_text}_multiple_models_{clustering}_{linkage}_{threshold}{lag_text}{indexes_text}{time_text}'        
    else:
        filename = f'feature_selection_CMIk{k}{threshold_text}_multiple_models{lag_text}{indexes_text}{time_text}'        

    print(f'Computing {filename}', flush=True)

    # Use map_async to run processes asynchronously and get results
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_async = pool.starmap_async(
            process_fold_cluster_CMI,
            [(fold, 
              n_splits,
              cluster, 
              [feature[feature['CLUSTER'] == cluster] for feature in aggregated_features_dict[fold]],
              aggregated_target_dict[fold][aggregated_target_dict[fold]['CLUSTER'] == cluster],
              k, CMI_threshold, CMI_n_features,
              time
              ) 
              for fold in range(n_splits) 
              for cluster in range(len(clusters_dict[fold]))])

        # Wait for all processes to complete
        results_async.wait()

        # Retrieve the results from each process (if needed)
        results = results_async.get()

    selected_features_dict = {}
    for fold in range(n_splits):
        selected_features_dict[fold] = {}
    for fold, cluster, selected_features in results:
        selected_features_dict[fold][cluster] = selected_features

    folder_path = f'../results/training_{17-test_size}/selected_features'
    os.makedirs(folder_path, exist_ok=True)

    with open(f'{folder_path}/{filename}.json', 'w') as f:
        json.dump(selected_features_dict, f)


