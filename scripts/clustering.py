import sys
import argparse
from sklearn.model_selection import KFold
from multiprocessing import Pool
import os
import pickle
import json
import random 

# Increase recursion limit
sys.setrecursionlimit(100000)

from data_utils import *
from HierarchicalClustering import HierarchicalClustering
from NonLinCTFA_estimate import NonLinCTFA_estimate
from NonLinCTFA_random import NonLinCTFA_random
from NonLinCTFA import NonLinCTFA
from NonLinCTFA_sizesort import NonLinCTFA_sizesort
from NonLinCTFA_strengthsort import NonLinCTFA_strengthsort
def get_folder_path(fold, drop_years, test_size, clustering, linkage, threshold, shp_area, use_only_indexes=False, selected_features=None, time=None):
    fold_dir = "full_training_set" if fold == -1 else f"fold_{fold:02d}"
    weight = 'weighted_' if shp_area is not None else '' 
    indexes = 'indexes' if use_only_indexes else ''
    sel_text = '' if selected_features is None else '_'.join(map(str, selected_features))
    time_text = '' if time is None else time
    drop_text = '' if drop_years==0 else f'_drop{drop_years}'
    if linkage is not None:
        if threshold is not None:
            folder_path = f"../results/training_{17-test_size}/{fold_dir}/clustering_{clustering}{indexes}{sel_text}{time_text}_{weight}{linkage}{drop_text}/clustering_{threshold}"
        else:
            folder_path = f"../results/training_{17-test_size}/{fold_dir}/clustering_{clustering}{indexes}{sel_text}{time_text}_{weight}{linkage}{drop_text}"
    else:
        if threshold is not None:
            folder_path = f"../results/training_{17-test_size}/{fold_dir}/clustering_{clustering}{indexes}{sel_text}{time_text}{drop_text}/clustering_{threshold}"
        else:
            folder_path = f"../results/training_{17-test_size}/{fold_dir}/clustering_{clustering}{indexes}{sel_text}{time_text}{drop_text}"

    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def process_split_hierarchical(args):
    n_splits, fold, test_size, train_index, test_index, drop_years, target, neighbour_pairs, clustering, linkage, initial_threshold, threshold_list, shp_area, save_pkl = args
    all_months = list(target.index)
    random.seed(fold)
    years_to_drop = random.sample(range(0, n_splits-1), drop_years)
    months_to_drop = []
    for year in years_to_drop:
        months_to_drop.extend(train_index[year*6:(year+1)*6])
    train_months = [all_months[i] for i in train_index if not i in months_to_drop]
    target_train_months = target.loc[train_months].reset_index(drop=True)

    for i, threshold in enumerate(threshold_list):
        folder_path = get_folder_path(fold, drop_years, test_size, clustering, linkage, threshold, shp_area)
        
        if i == 0:
            if initial_threshold is None:
                hierarchicalClustering = HierarchicalClustering(target_train_months, neighbour_pairs, 
                                                linkage=linkage, threshold=threshold,
                                                weights=shp_area)
            else:
                initial_clustering_path = get_folder_path(fold, drop_years, test_size, clustering, linkage, initial_threshold, shp_area)
                with open(f'{initial_clustering_path}/clustering.pkl', 'rb') as f:
                    hierarchicalClustering = pickle.load(f)
                hierarchicalClustering.set_threshold(threshold)

        else:
            hierarchicalClustering.set_initial_threshold(threshold_list[i-1])
            hierarchicalClustering.set_threshold(threshold)
        
        hierarchicalClustering.compute_clusters()

        if save_pkl:
            with open(f'{folder_path}/clustering.pkl', 'wb') as f:
                pickle.dump(hierarchicalClustering, f)
        
        with open(f'{folder_path}/clustering.json', 'w') as f:
            clusters = hierarchicalClustering.get_clusters(threshold)
            json.dump(clusters, f)

        print(f'Computed {folder_path}', flush=True)

def process_split_NonLinCTFA(args):
    n_splits, fold, test_size, train_index, test_index, drop_years, features, target, neighbour_pairs, clustering, linkage, epsilon, shp_area, use_only_indexes, selected_features, time = args
    all_months = list(target.index)
    random.seed(fold)
    years_to_drop = random.sample(range(0, n_splits-1), drop_years)
    months_to_drop = []
    for year in years_to_drop:
        months_to_drop.extend(train_index[year*6:(year+1)*6])
    train_months = [all_months[i] for i in train_index if not i in months_to_drop]
    features_train_months = [feature.loc[train_months].reset_index(drop=True) for feature in features]
    target_train_months = target.loc[train_months].reset_index(drop=True)

    folder_path = get_folder_path(fold, drop_years, test_size, clustering, linkage, epsilon, shp_area, use_only_indexes, selected_features, time)
    
    if clustering=='NonLinCTFA_estimate':
        nonLinCTFA = NonLinCTFA_estimate(features_train_months, target_train_months, neighbour_pairs, 
                                            linkage=linkage, epsilon=epsilon,
                                            weights=shp_area)
        
    elif clustering=='NonLinCTFA':
        nonLinCTFA = NonLinCTFA(features_train_months, target_train_months, neighbour_pairs, 
                                    linkage=linkage, epsilon=epsilon,
                                    weights=shp_area)

    elif clustering=='NonLinCTFA_random':
        nonLinCTFA = NonLinCTFA_random(features_train_months, target_train_months, neighbour_pairs, epsilon=epsilon,
                                            weights=shp_area)
    
    elif clustering =='NonLinCTFA_sizesort':
        nonLinCTFA = NonLinCTFA_sizesort(features_train_months, target_train_months, neighbour_pairs, epsilon=epsilon,
                                            weights=shp_area)
    
    elif clustering =='NonLinCTFA_strengthsort':
        nonLinCTFA = NonLinCTFA_strengthsort(features_train_months, target_train_months, neighbour_pairs, epsilon=epsilon,
                                            weights=shp_area)
    
    else:
        raise ValueError("clustering parameter must be 'NonLinCTFA', 'NonLinCTFA_estimate', 'NonLinCTFA_sizesort' or 'NonLinCTFA_random'.")

    nonLinCTFA.compute_clusters()
    clusters = nonLinCTFA.get_clusters()
    clusters_list = [values for values in clusters.values()]
    clusters_list = sorted(clusters_list, key=len, reverse=True)

    with open(f'{folder_path}/clustering.json', 'w') as f:
        json.dump(clusters_list, f)

    print(f'Computed {folder_path}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--n_splits', type=int, default=1, help='Description of parameter')
    parser.add_argument('--drop_years', type=int, default=0, help='Description of parameter')
    parser.add_argument('--test_size', type=int, default=0, help='Description of parameter')
    parser.add_argument('--clustering', type=str, help='Description of parameter')
    parser.add_argument('--linkage', type=str, help='Description of parameter')
    parser.add_argument('--initial_threshold', type=float)
    parser.add_argument('--threshold_list', type=float, nargs='+', help='List of clustering thresholds in descending order.')
    parser.add_argument('--epsilon_list', type=float, nargs='+')
    parser.add_argument('--use_area_weight', type=bool, default=False)
    parser.add_argument('--use_only_indexes', type=bool, default=False)
    parser.add_argument('--selected_features', type=int, nargs='+')    
    parser.add_argument('--time', type=str, choices=['month', 'year', 'month_year'], help='Description of parameter')
    parser.add_argument('--no_pkl', type=bool, default=False)

    args = parser.parse_args()
    n_splits = args.n_splits
    drop_years = args.drop_years
    test_size = args.test_size
    clustering = args.clustering
    linkage = args.linkage
    initial_threshold = args.initial_threshold
    threshold_list = args.threshold_list    
    epsilon_list = args.epsilon_list
    use_area_weight = args.use_area_weight  
    use_only_indexes = args.use_only_indexes
    selected_features = args.selected_features
    time = args.time
    save_pkl = not args.no_pkl

    target = get_target_for_clustering()
    neighbour_pairs = get_neighbour_pairs(subids=target.columns)

    if clustering.startswith('NonLinCTFA'):
        target_for_features = get_target()
        features = get_features_for_clustering(target=target_for_features, only_indexes=use_only_indexes)
        if selected_features is not None:
            features = [features[i] for i in selected_features]
        if time is not None:
            feature_time = get_feature_time(target_for_features, time)
            for i in range(len(feature_time)):
                feature_time[i] = select_columns_by_months(feature_time[i], [4,5,6,7,8,9])
                # Transpose the DataFrame and set the "SUBID" column as the column names
                feature_time[i] = feature_time[i].set_index('SUBID').T
                feature_time[i] = feature_time[i].reset_index(drop=True)
                feature_time[i] = feature_time[i].rename_axis(None, axis=1)
            features = features + feature_time

    if use_area_weight and linkage!='complete':
        shp = get_shapefile()
        shp_area = shp[['SUBID', 'AREA']] 
    else:
        shp_area = None

    if test_size > 0:
        test_start = target.shape[0]-test_size*6
        target = target.iloc[:test_start, :]
        if clustering.startswith('NonLinCTFA'):
            for i in range(len(features)):
                features[i] = features[i].iloc[:test_start, :]

    all_months = list(target.index)

    if clustering=='hierarchical':
        if n_splits == 1:
            train_index = test_index = list(range(len(all_months)))  # Use the entire dataset
            args_list = [(n_splits, -1, test_size, train_index, test_index, drop_years, target, neighbour_pairs, clustering, linkage, initial_threshold, threshold_list, shp_area, save_pkl)]
        else:
            kf = KFold(n_splits=n_splits)
            args_list = [(n_splits, i, test_size, train_index, test_index, drop_years, target, neighbour_pairs, clustering, linkage, initial_threshold, threshold_list, shp_area, save_pkl) 
                        for i, (train_index, test_index) in enumerate(kf.split(all_months))]
            
        with Pool(processes=os.cpu_count()) as pool:
            pool.map(process_split_hierarchical, args_list)
    

    elif clustering.startswith('NonLinCTFA'):
        if n_splits == 1:
            train_index = test_index = list(range(len(all_months)))  # Use the entire dataset
            args_list = [(n_splits, -1, test_size, train_index, test_index, drop_years, features, target, neighbour_pairs, clustering, linkage, epsilon, shp_area, use_only_indexes, selected_features, time)
                        for epsilon in epsilon_list]
        else:
            kf = KFold(n_splits=n_splits)
            args_list = [(n_splits, i, test_size, train_index, test_index, drop_years, features, target, neighbour_pairs, clustering, linkage, epsilon, shp_area, use_only_indexes, selected_features, time) 
                        for i, (train_index, test_index) in enumerate(kf.split(all_months))
                        for epsilon in epsilon_list]

        with Pool(processes=os.cpu_count()) as pool:
            pool.map(process_split_NonLinCTFA, args_list)
