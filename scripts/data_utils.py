import numpy as np
import pandas as pd
import ast
import json

def select_columns_by_months(dataframe, selected_months, start_column=1):
    selected_columns = []
    all_columns = list(map(int, dataframe.columns[start_column:]))

    for col in all_columns:
        if col%12 != 0:
            month = col%12
        else: 
            month = 12

        if month in selected_months:
            selected_columns.append(col)    

    columns_to_drop = np.setdiff1d(all_columns, np.array(selected_columns))
    columns_to_drop = [str(i) for i in columns_to_drop]
    filtered_dataframe = dataframe.drop(columns=columns_to_drop, axis=1)

    return filtered_dataframe

def get_target(max_nan=4):
    target = pd.read_csv('../data/target/FAPAN_50_25_2001_2018.csv')
    
    # Keep data from 2002 to 2018
    columns = ['SUBID'] + list(map(str, np.arange(109, 313)))
    target = target[columns]
    # Keep data from April to September
    target = select_columns_by_months(target, [4,5,6,7,8,9])
    selected_columns = target.iloc[:, 1:]
    # Keep subbasins with at most 4 na observations
    target = target[selected_columns.isnull().sum(axis=1)<=max_nan]
    target = target.reset_index(drop=True)
    
    target['SUBID'] = target['SUBID'].astype(int).astype(str)
    target[target.columns[1:]] = target[target.columns[1:]].astype('float32')
    return target

def get_target_for_clustering():
    target = get_target()
    target_for_clustering = target.set_index('SUBID').T
    target_for_clustering = target_for_clustering.reset_index(drop=True)
    target_for_clustering = target_for_clustering.rename_axis(None, axis=1)
    return target_for_clustering

def get_shapefile_from_csv():
    target = get_target()
    shp = pd.read_csv('../data/shp.csv')
    shp['SUBID'] = shp['SUBID'].astype('int32').astype(str)
    shp['AREA'] = shp['AREA'].astype('int32')
    shp[['LATITUDE', 'LONGITUDE', 'ELEV_MEAN']] = shp[['LATITUDE', 'LONGITUDE', 'ELEV_MEAN']].astype('float32')
    shp = shp[shp['SUBID'].isin(target['SUBID'])].reset_index(drop=True)
    return shp

def get_shapefile(target=get_target()):
    import geopandas as gpd
    shapefile = gpd.read_file('../data/shapefile/SUBID_TotalDomain_WGS84_20140428_2_repaired_geometry.shp')
    shapefile['SUBID'] = shapefile['SUBID'].astype(int).astype(str)
    shapefile = shapefile[shapefile['SUBID'].isin(target['SUBID'])].reset_index(drop=True)
    return shapefile

def get_neighbour_pairs(subids):
    neighbours_csv = pd.read_csv("../data/neighbours.csv")

    # Create a set to store unique pairs
    neighbour_pairs = set()

    # Iterate over each row and generate pairs
    for index, row in neighbours_csv.iterrows():
        subid = str(int(row['SUBID']))
        neighbours = map(str, map(int, ast.literal_eval(row['NEIGHBOURS'])))
        
        # Generate pairs and add to the set
        for neighbour in neighbours:
            pair = tuple(sorted([subid, neighbour]))
            neighbour_pairs.add(pair)

    # Eliminate neighbours pairs for subbasins not in the data anymore
    neighbour_pairs = {pair for pair in neighbour_pairs if ((pair[0] in subids) and (pair[1] in subids))}

    return neighbour_pairs

def get_features(feature_lag=0, target=None, only_indexes=False):
    features_folder = '../data/features'
    if only_indexes:
        feature_names = ["SPEI_1", "SPEI_3", "SPI_1", "SPI_3", "SRI_6", "SSI_1", "SSI_3", "SSI_6"]
    else:         
        feature_names = ["meanCOUT1", "meanCOUT3", "meanCOUT6", "meanSRFF1", "meanSRFF3", "meanSRFF6",
                        "SPEI_1", "SPEI_3", "SPI_1", "SPI_3", "SRI_6", "SSI_1", "SSI_3", "SSI_6",
                        "sumCPRC_sumEPOT1", "sumCPRC_sumEPOT3", "sumCPRC1", "sumCPRC3"
                        ]
    features = [pd.read_csv(f"{features_folder}/{file}.csv", sep=',') for file in feature_names]
    
    for i in range(len(features)):
        features[i]['SUBID'] = features[i]['SUBID'].astype(str)
        #features[i] = features[i][['SUBID'] + list(map(str, range(109, 313)))]        
        features[i][features[i].columns[1:]] = features[i][features[i].columns[1:]].astype('float32')

    column_mappings = [{str(i): str(int(i) + lag) for i in list(features[0].columns[1:])} for lag in range(1, feature_lag + 1)]

    all_features = features.copy()
    for column_mapping in column_mappings:
        all_features.extend([features[i].rename(columns=column_mapping) for i in range(len(features))])

    if target is not None:
        for i in range(len(all_features)):
            all_features[i] = all_features[i][target.columns]
            all_features[i] = all_features[i][all_features[i]['SUBID'].isin(target['SUBID'])]
            all_features[i] = all_features[i].reset_index(drop=True)
    return all_features

def get_features_for_clustering(target, only_indexes=False):
    features = get_features(target=target, only_indexes=only_indexes)
    
    for i in range(len(features)):
        features[i] = select_columns_by_months(features[i], [4,5,6,7,8,9])
        # Transpose the DataFrame and set the "SUBID" column as the column names
        features[i] = features[i].set_index('SUBID').T
        features[i] = features[i].reset_index(drop=True)
        features[i] = features[i].rename_axis(None, axis=1)

    return features

def load_clusters(fold=None, test_size=None, clustering='none', linkage=None, threshold=None, clustering_file=None):
    if clustering_file:
        with open(clustering_file, 'r') as json_file:
            clusters = json.load(json_file)
    
    else:
        if clustering == 'none':
            folder_path = f'../results'
            file_name = 'clustering_none.json'
        else:
            fold_dir = "full_training_set" if fold == -1 else f"fold_{fold:02d}"
            file_name = 'clustering.json'
            if type(threshold)==float:
                folder_path = f'../results/training_{17-test_size}/{fold_dir}/clustering_{clustering}_{linkage}/clustering_{threshold}'
            else:
                folder_path = f'../results/training_{17-test_size}/{fold_dir}/clustering_{clustering}_{linkage}'

        with open(f'{folder_path}/{file_name}', 'r') as json_file:
            clusters = json.load(json_file)
    
    return clusters


def get_aggregated_target(target, clusters):
    if len(clusters) == target.shape[0]:
        aggregated_target = target.iloc[:, 1:].copy()
        aggregated_target.insert(0, 'CLUSTER', np.array(range(len(clusters))))
        return aggregated_target
    
    aggregated_target = pd.DataFrame(np.nan, index=range(len(clusters)), columns=target.columns[1:])
    aggregated_target.insert(0, 'CLUSTER', np.array(range(len(clusters))))

    for i in range(len(clusters)):
        cluster_target = target[target['SUBID'].isin(clusters[i])].iloc[:, 1:].reset_index(drop=True)
        
        # for the moment do not consider the area of each subbasin
        cluster_target = cluster_target.mean(axis=0)
        aggregated_target.iloc[i, 1:] = cluster_target

    return aggregated_target


def get_feature_time(feature, time):
    # Pre-calculate constants for transformation
    pi_transform = 2 * np.pi / 12
    
    # Calculate month and year directly using vectorized operations
    columns = feature.columns[1:].astype(int)
    months = np.where(columns % 12 == 0, 12, columns % 12)
    years = 1993 + (columns - 1) // 12

    # Calculate cosine and sine transformations for months
    feature_month_cos = np.cos(months * pi_transform)
    feature_month_sin = np.sin(months * pi_transform)

    # Construct the new DataFrames
    feature_month_cos_df = feature.copy()
    feature_month_sin_df = feature.copy()
    feature_year_df = feature.copy()
    
    feature_month_cos_df.iloc[:, 1:] = feature_month_cos
    feature_month_sin_df.iloc[:, 1:] = feature_month_sin
    feature_year_df.iloc[:, 1:] = years

    # Mapping from time parameter to corresponding data features
    feature_time = {
        'year': [feature_year_df],
        'month': [feature_month_sin_df, feature_month_cos_df],
        'month_year': [feature_month_sin_df, feature_month_cos_df, feature_year_df]
    }

    if time not in feature_time:
        raise ValueError(f'Feature {time} is not available, select feature time from "year", "month", and "month_year".')

    return feature_time[time]

def get_aggregated_features(features, clusters):
    aggregated_features = []

    if len(clusters) == features[0].shape[0]:
        for i in range(len(features)):
            aggregated_features.append(features[i].iloc[:, 1:].copy())
            aggregated_features[i].insert(0, 'CLUSTER', np.array(range(len(clusters))))

        return aggregated_features
    
    for feature in features:
        cluster_means = []
        for cluster in clusters:
            cluster_feature = feature[feature['SUBID'].isin(cluster)].copy()
            cluster_mean = cluster_feature.iloc[:, 1:].mean()
            cluster_means.append(cluster_mean)

        aggregated_feature = pd.concat(cluster_means, axis=1).transpose().reset_index(drop=True)
        aggregated_feature.insert(0, 'CLUSTER', range(len(clusters)))

        aggregated_features.append(aggregated_feature)

    return aggregated_features

def get_selected_features_from_json(clustering_linkage, clustering_threshold, feature_aggregation, feature_lag, feature_lag_only, time, k, CMI_threshold=0.0, CMI_n_features=-1, aug=-1):
    aug_string = ''
    if aug != -1:
        aug_string = f'_aug{aug}' 

    only = ''
    if feature_lag_only:
        only = 'only'

    if CMI_n_features != -1:
        filename = f'selected_features_{only}lag{feature_lag}_{time}k{k}_n{CMI_n_features}{aug_string}.json'
    else:
        filename = f'selected_features_{only}lag{feature_lag}_{time}k{k}_t{CMI_threshold}{aug_string}.json'

    print(f'Loading selected features from {filename}')

    if type(clustering_threshold)==float:
        folder_path = f'../data/results/clustering_{clustering_linkage}/clustering_{clustering_threshold}/{feature_aggregation}/selected_features'
    else:
        folder_path = f'../data/results/clustering_{clustering_linkage}/{feature_aggregation}/selected_features'

    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r') as json_file:
        selected_features = json.load(json_file)

    new_dict = {int(key): value for key, value in selected_features.items()}
    selected_features = dict(sorted(new_dict.items()))

    return selected_features

def get_selected_features_all(aggregated_features, clusters):
    print('All features are considered.')
    selected_features = {}
    
    #progress_bar = tqdm(total=len(clusters), position=0, leave=True, smoothing=0)
    for i in range(len(clusters)):
        num_features = 0
        for feature in aggregated_features:
            num_features += feature[feature['CLUSTER'] == i].shape[0]

        selected_features[i] = list(range(num_features))   
        #progress_bar.update(1)

    #progress_bar.close()

    return selected_features


def get_aggregated_coordinates(clusters, feature):
    shp = get_shapefile_from_csv()
    
    coord_aggr_list = [
        pd.DataFrame(np.nan, index=range(feature.shape[0]), columns=feature.columns)
        .assign(CLUSTER=feature['CLUSTER'])
        for _ in range(3)
    ]

    for i in range(len(clusters)):
        coord_aggr_list[0].iloc[i, 1:] = shp[shp['SUBID'].isin(clusters[i])]['LATITUDE'].mean(axis=0)
        coord_aggr_list[1].iloc[i, 1:] = shp[shp['SUBID'].isin(clusters[i])]['LONGITUDE'].mean(axis=0)
        coord_aggr_list[2].iloc[i, 1:] = shp[shp['SUBID'].isin(clusters[i])]['ELEV_MEAN'].mean(axis=0)

    return coord_aggr_list