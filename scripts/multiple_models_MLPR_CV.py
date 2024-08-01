import numpy as np
import pandas as pd
import os
import argparse
import multiprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from data_utils import *
from model_utils import *

def process_cluster(fold, n_splits, cluster, features, target, model_type):        
    all_months = list(target.columns[1:])
    kf = KFold(n_splits=n_splits)
    train_index, test_index = list(kf.split(all_months))[fold]
    train_months = [all_months[i] for i in train_index]
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

    #early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Fit the model
    pipeline.fit(X_train, y_train, model__epochs=10, model__batch_size=32, model__verbose=0, model__validation_split=0.0)#, model__callbacks=[early_stopping])    
    
    # Predict without NaN values
    y_pred_nonan = pipeline.predict(X_test).flatten()
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
    parser.add_argument('--clustering', type=str, default='none', help='Description of parameter')
    parser.add_argument('--linkage', type=str, help='Description of parameter')
    parser.add_argument('--threshold', type=float, help='Description of parameter')
    parser.add_argument('--feature_lag', type=int, choices=[0, 1, 2], default=0, help='Description of parameter')
    parser.add_argument('--time', type=str, choices=['month', 'year', 'month_year'], help='Description of parameter')
    parser.add_argument('--use_only_indexes', type=bool, default=False)
    parser.add_argument('--selected_features', type=int, nargs='+')
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
    selected_features = args.selected_features
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

    lag_text = '' if feature_lag == 0 else f'lag{feature_lag}_'
    time_text = '' if time is None else time + '_'
    indexes_text = 'indexes_' if use_only_indexes else ''
    sel_text = '' if selected_features is None else '_'.join(map(str, selected_features)) + '_'

    if clustering != 'none':
        filename = f'multiple_models_{clustering}_{linkage}_{threshold}_{lag_text}{indexes_text}{sel_text}{time_text}{model_type}'        
    else:
        filename = f'multiple_models_{lag_text}{indexes_text}{sel_text}{time_text}{model_type}'        

    print(f'Computing {filename}', flush=True)

    # Use map_async to run processes asynchronously and get results
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_async = pool.starmap_async(
            process_cluster,
            [(fold, 
              n_splits,
              cluster, 
              [feature[feature['CLUSTER'] == cluster] for feature in aggregated_features_dict[fold]],
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

    for fold, cluster, test_months, y_pred in results:
        for subid in clusters_dict[fold][cluster]:
            # Select the row corresponding to the subid
            row_index = predicted_target.index[predicted_target['SUBID'] == subid][0]

            # Assign y_pred to the appropriate columns and row
            predicted_target.loc[row_index, test_months] = y_pred

    predicted_target.to_csv(f'{folder_path}/{filename}.csv', index=False)


