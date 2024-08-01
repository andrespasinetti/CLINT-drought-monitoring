import numpy as np
import pandas as pd
import os
import argparse
import multiprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import random

from data_utils import *
from model_utils import *

def process_cluster(fold, n_splits, drop_years, cluster, features, target, model_type):        
    all_months = list(target.columns[1:])
    kf = KFold(n_splits=n_splits)
    train_index, test_index = list(kf.split(all_months))[fold]
    random.seed(fold)
    years_to_drop = random.sample(range(0, n_splits-1), drop_years)
    months_to_drop = []
    for year in years_to_drop:
        months_to_drop.extend(train_index[year*6:(year+1)*6])
    train_months = [all_months[i] for i in train_index if not i in months_to_drop]
    test_months = [all_months[i] for i in test_index]

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
        y_pred = np.full(len(test_months), np.nan)
        return fold, cluster, test_months, y_pred

    model = get_model(model_type, input_dim=X_train.shape[1])
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(X_train, y_train)        
    
    # Predict without NaN values
    if model_type.startswith('MLPR'):
        y_pred_nonan = clf.predict(X_test).flatten()
    else:
        y_pred_nonan = clf.predict(X_test)
    y_pred = np.full(len(valid_indices_test), np.nan)
    y_pred[valid_indices_test] = y_pred_nonan

    print(f'Computed predictions for fold {fold}, cluster {cluster}', flush=True)
    return fold, cluster, test_months, y_pred
    
def process_data(args):
    fold, test_size, clustering, linkage, threshold, target, features, time = args
    clusters = load_clusters(fold, test_size, clustering, linkage, threshold)
    aggregated_target = get_aggregated_target(target, clusters)
    aggregated_features = get_aggregated_features(features, clusters)

    if time is not None:
        feature_time = get_feature_time(aggregated_target, time)
        aggregated_features = aggregated_features + feature_time

    return fold, clusters, aggregated_target, aggregated_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--n_splits', type=int, default=1, help='Description of parameter')
    parser.add_argument('--test_size', type=int, default=0, help='Description of parameter')
    parser.add_argument('--drop_years', type=int, default=0, help='Description of parameter')
    parser.add_argument('--clustering', type=str, default='none', help='Description of parameter')
    parser.add_argument('--linkage', type=str, help='Description of parameter')
    parser.add_argument('--threshold', type=float, help='Description of parameter')
    parser.add_argument('--feature_lag', type=int, choices=[0, 1, 2], default=0, help='Description of parameter')
    parser.add_argument('--time', type=str, choices=['month', 'year', 'month_year'], help='Description of parameter')
    parser.add_argument('--detrend', type=bool, default=False)
    parser.add_argument('--use_only_indexes', type=bool, default=False)
    parser.add_argument('--selected_features', type=int, nargs='+')
    parser.add_argument('--feature_selection', type=str)
    parser.add_argument('--max_num_features', type=int)
    parser.add_argument('--model_type', type=str, help='Description of parameter')
    parser.add_argument('--num_processes', type=int, default=multiprocessing.cpu_count(),
                            help='Number of processes to use. Default is the maximum available processors.')

    args = parser.parse_args()

    n_splits = args.n_splits
    test_size = args.test_size
    drop_years = args.drop_years
    clustering = args.clustering
    linkage = args.linkage
    threshold = args.threshold
    feature_lag = args.feature_lag
    time = args.time
    detrend = args.detrend
    use_only_indexes = args.use_only_indexes
    selected_features = args.selected_features
    feature_selection = args.feature_selection
    max_num_features = args.max_num_features
    model_type = args.model_type
    num_processes = args.num_processes

    target = get_target()
    features = get_features(feature_lag, target, only_indexes=use_only_indexes)
    if test_size > 0:
        target = target.iloc[:, :-test_size*6]
        for i in range(len(features)):
            features[i] = features[i].iloc[:, :-test_size*6]
    
    if selected_features is not None:
        features = [features[i] for i in selected_features]

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
    time_text = '' if time is None else '_' + time
    detrend_text = '_detrended' if detrend else ''
    indexes_text = '_indexes' if use_only_indexes else ''
    sel_text = '' if selected_features is None else '_' + '_'.join(map(str, selected_features))
    sel_text = sel_text if feature_selection is None else '_' + feature_selection
    max_text = '' if max_num_features is None else f'_max{max_num_features}'
    drop_text = '' if drop_years==0 else f'_drop{drop_years}'

    if clustering != 'none':
        filename = f'multiple_models_{clustering}_{linkage}{drop_text}_{threshold}{lag_text}{indexes_text}{time_text}{sel_text}{detrend_text}{max_text}_{model_type}'        
    else:
        filename = f'multiple_models{drop_text}{lag_text}{indexes_text}{time_text}{sel_text}{detrend_text}{max_text}_{model_type}'        

    if feature_selection is not None:
        #fix the detrend stuff, not necessary anyway... delete it
        path = f'../results/training_{n_splits}/selected_features/feature_selection_{feature_selection}_multiple_models_{clustering}_{linkage}_{threshold}{lag_text}{indexes_text}{detrend_text}{time_text}.json'
        with open(path, 'r') as f:
            selected_features_dict = json.load(f)

        if max_num_features is not None:
            for fold in range(n_splits):
                for cluster in range(len(clusters_dict[fold])):
                    selected_features_dict[str(fold)][str(cluster)] = selected_features_dict[str(fold)][str(cluster)][:max_num_features]

    else:
        selected_features_dict = {}
        for fold in range(n_splits):
            selected_features_dict[str(fold)] = {}
            for cluster in range(len(clusters_dict[fold])):
                selected_features_dict[str(fold)][str(cluster)] = list(range(len(aggregated_features_dict[fold])))

    print(f'Computing {filename}', flush=True)

    # Use map_async to run processes asynchronously and get results
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_async = pool.starmap_async(
            process_cluster,
            [(fold, 
              n_splits,
              drop_years,
              cluster, 
              [feature[feature['CLUSTER'] == cluster] for i, feature in enumerate(aggregated_features_dict[fold]) if i in selected_features_dict[str(fold)][str(cluster)]],
              aggregated_target_dict[fold][aggregated_target_dict[fold]['CLUSTER'] == cluster],
              model_type) 
              for fold in range(n_splits) 
              for cluster in range(len(clusters_dict[fold]))])

        # Wait for all processes to complete
        results_async.wait()

        # Retrieve the results from each process (if needed)
        results = results_async.get()

    # We map each fold-cluster prediction to the subbasins, this is necessary since each fold has its own clustering
    predicted_target = pd.DataFrame(np.nan, index=range(len(target)), columns=target.columns)
    predicted_target['SUBID'] = target['SUBID']
    
    folder_path = f'../results/training_{17-test_size}/predictions'
    os.makedirs(folder_path, exist_ok=True)

    from collections import defaultdict

    # Initialize a dictionary to hold data for each SUBID
    results_dict = defaultdict(lambda: defaultdict(float))

    # Loop through the results
    for fold, cluster, test_months, y_pred in results:
        for subid in clusters_dict[fold][cluster]:
            for month, prediction in zip(test_months, y_pred):
                # Ensure each SUBID's month prediction is updated in the dictionary
                # Here we assume the latest prediction should be used,
                # or you could aggregate them (e.g., average, max, etc.)
                results_dict[subid][month] = prediction

    # Convert to DataFrame
    updates_df = pd.DataFrame.from_dict(results_dict, orient='index')
    updates_df.index.name = 'SUBID'

    # Merge updates into the original DataFrame
    predicted_target.set_index('SUBID', inplace=True)
    predicted_target.update(updates_df)
    predicted_target.reset_index(inplace=True)

    predicted_target.to_csv(f'{folder_path}/{filename}.csv', index=False)

"""
    for fold, cluster, test_months, y_pred in results:
        for subid in clusters_dict[fold][cluster]:
            # Select the row corresponding to the subid
            row_index = predicted_target.index[predicted_target['SUBID'] == subid][0]

            # Assign y_pred to the appropriate columns and row
            predicted_target.loc[row_index, test_months] = y_pred

    predicted_target.to_csv(f'{folder_path}/{filename}.csv', index=False)
"""




