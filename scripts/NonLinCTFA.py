import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
from itertools import combinations
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


class NonLinCTFA():
    """
    Attributes:
    - features: List of DataFrames containing features.
    - targets_df: DataFrame containing target values.
    - clusters: Dictionary mapping cluster IDs to a list of elements.
    - neighbors: List of neighbor pairs.
    - neighbor_strengths: Dictionary storing strengths for each neighbor pair.
    - active_neighbor_strengths: Dictionary storing strengths only for pairs that could be aggregated in the next iteration (to optimize code)
    """
        
    def __init__(self, features, targets_df, neighbors=None, linkage='centroid', epsilon=0, weights=None):

        self.features = copy.deepcopy(features)
        """
        # Features preprocessing
        for i in range(len(self.features)):
            scaler = StandardScaler() #it deals with nan values
            features[i] = pd.DataFrame(scaler.fit_transform(features[i]), columns=features[i].columns)
        """

        if type(targets_df) == str:
            pd.read_csv(targets_df)
        else: self.targets_df = targets_df.copy(deep=True)

        # Initialize clusters: each element as a cluster
        self.clusters = {subid:[subid] for subid in targets_df.columns}
        
        self.linkage = linkage

        # If neighbors are not provided, generate all possible neighbor pairs
        if neighbors:
            self.neighbors = copy.deepcopy(neighbors)
        else:
            self.neighbors = set(combinations(list(self.targets_df.columns), 2))

        # Get strengths of each element
        self.neighbor_strengths = self.initialize_neighbor_strengths()
        self.active_neighbor_strengths = copy.deepcopy(self.neighbor_strengths)
        self.epsilon = epsilon
        self.weights = weights

    def get_clusters(self):
        return self.clusters

    def compute_strength(self, cluster1, cluster2):
        """
        Compute the strength between cluster1 and cluster2. 
        In this case we consider as strength the correlation between the two vectors.
        """
        cluster1_value = self.targets_df[self.clusters[cluster1]].mean(axis=1)
        cluster2_value = self.targets_df[self.clusters[cluster2]].mean(axis=1)

        # Create a mask to filter nan values
        cluster1_mask = ~np.isnan(cluster1_value)
        cluster2_mask = ~np.isnan(cluster2_value)
        mask = cluster1_mask & cluster2_mask
        
        strength = np.corrcoef(cluster1_value[mask], cluster2_value[mask])[0, 1]
        return strength

    def initialize_neighbor_strengths(self):
        neighbor_strengths = {}

        for neighbor_pair in self.neighbors:
            neighbor1, neighbor2 = neighbor_pair
            strength = self.compute_strength(neighbor1, neighbor2)
            neighbor_strengths[neighbor_pair] = strength

        return  neighbor_strengths 


    """
    def average_with_nan(self, vec1, vec2):
    averaged_vec = np.where(np.isnan(vec1), vec2, np.where(np.isnan(vec2), vec1, (vec1 + vec2) / 2))
    return averaged_vec
    """

    def prepare_features(self, cluster1, cluster2):       
        x1 = pd.concat([feature[self.clusters[cluster1]].mean(axis=1) for feature in self.features], axis=1)
        x2 = pd.concat([feature[self.clusters[cluster2]].mean(axis=1) for feature in self.features], axis=1)
        x1.columns = range(len(x1.columns))
        x2.columns = range(len(x2.columns))
        
        # Identify features with all NaN values in either x1 or x2
        nan_features_x1 = x1.columns[x1.isna().all()]
        nan_features_x2 = x2.columns[x2.isna().all()]
        nan_features_to_drop = set(nan_features_x1) | set(nan_features_x2)

        # Eliminate features with all NaN values in either x1 or x2
        x1 = x1.drop(columns=nan_features_to_drop)
        x2 = x2.drop(columns=nan_features_to_drop)

        x_aggr = pd.concat([feature[self.clusters[cluster1] + self.clusters[cluster2]].mean(axis=1) for feature in self.features], axis=1)
        x_aggr = x_aggr.drop(columns=nan_features_to_drop)

        return x1, x2, x_aggr 
    
    def prepare_target(self, cluster1, cluster2):
        y1 = self.targets_df[self.clusters[cluster1]].mean(axis=1)
        y2 = self.targets_df[self.clusters[cluster2]].mean(axis=1)
        y1 = y1 - np.nanmean(y1)
        y2 = y2 - np.nanmean(y2)

        y_aggr = self.targets_df[self.clusters[cluster1] + self.clusters[cluster2]].mean(axis=1)
        y_aggr = y_aggr - np.nanmean(y_aggr)
        return y1, y2, y_aggr

    def drop_observations_with_nan(self, x1, x2, x_aggr, y1, y2, y_aggr):
        # Drop rows with NaN values for each variable separately
        x1 = x1.dropna()
        x2 = x2.dropna()
        x_aggr = x_aggr.dropna()
        y1 = y1.dropna()
        y2 = y2.dropna()
        y_aggr = y_aggr.dropna()

        # Find the common indices across all variables
        common_indices = x1.index.intersection(x2.index).intersection(x_aggr.index).intersection(y1.index).intersection(y2.index).intersection(y_aggr.index)

        # Filter variables to keep only rows present in all variables
        x1 = x1.loc[common_indices].values
        x2 = x2.loc[common_indices].values
        x_aggr = x_aggr.loc[common_indices].values
        y1 = y1.loc[common_indices].values
        y2 = y2.loc[common_indices].values
        y_aggr = y_aggr.loc[common_indices].values

        return x1, x2, x_aggr, y1, y2, y_aggr, common_indices

    def compute_scores(self, cluster1, cluster2):
        x1, x2, x_aggr = self.prepare_features(cluster1, cluster2)
        y1, y2, y_aggr = self.prepare_target(cluster1, cluster2)
        x1, x2, x_aggr, y1, y2, y_aggr, _ = self.drop_observations_with_nan(x1, x2, x_aggr, y1, y2, y_aggr)

        if (x1.shape[1] == 0) or (x2.shape[1] == 0):
            return None, None, None, None, None, None

        target1_regr = LinearRegression()
        target2_regr = LinearRegression()
        aggr_regr = LinearRegression()

        # no need to standardize for a linear regression
        target1_regr.fit(x1, y1)
        target2_regr.fit(x2, y2)
        aggr_regr.fit(x_aggr, y_aggr)
        preds1 = target1_regr.predict(x1).astype('float32')
        preds2 = target2_regr.predict(x2).astype('float32')
        preds_aggr = aggr_regr.predict(x_aggr).astype('float32')
        
        ### variance ###
        D = x1.shape[1] 
        n = x1.shape[0]

        residuals1 = y1 - preds1
        residuals2 = y2 - preds2
        residuals_aggr = y_aggr - preds_aggr

        s_squared1 = np.dot(residuals1.reshape(1, n), residuals1) / (n-D-1)
        s_squared2 = np.dot(residuals2.reshape(1, n), residuals2) / (n-D-1)
        s_squared_aggr = np.dot(residuals_aggr.reshape(1, n), residuals_aggr) / (n-D-1)

        var1 = s_squared1*D / (n-1)
        var2 = s_squared2*D / (n-1)
        var_aggr = s_squared_aggr*D / (n-1)

        ### bias ### 
        r2_1 = r2_score(y1, preds1)
        r2_2 = r2_score(y2, preds2)
        r2_aggr = r2_score(y_aggr, preds_aggr)

        s_squaredF1 = (np.var(y1, ddof=1) - s_squared1)
        s_squaredF2 = (np.var(y2, ddof=1) - s_squared2)
        s_squaredFaggr = (np.var(y_aggr, ddof=1) - s_squared_aggr)

        r2_1_weighted = r2_1 * s_squaredF1
        r2_2_weighted = r2_2 * s_squaredF2
        r2_aggr_weighted = r2_aggr * s_squaredFaggr

        return var1, var2, var_aggr, r2_1_weighted, r2_2_weighted, r2_aggr_weighted

    def check_aggregation(self, cluster1, cluster2):
        var1, var2, var_aggr, r2_1_weighted, r2_2_weighted, r2_aggr_weighted = self.compute_scores(cluster1, cluster2)
        if var1 == None:
            return False
        
        if (var1 - var_aggr + r2_aggr_weighted - (r2_1_weighted + r2_2_weighted)/2 >= self.epsilon) & (var2 - var_aggr + r2_aggr_weighted - (r2_1_weighted + r2_2_weighted)/2 >= self.epsilon): 
            return True
        else:
            return False
        
    def get_new_cluster_key(self):
        cluster_ID = 0
        new_key = f'cluster_{cluster_ID}'

        while new_key in self.clusters:
            cluster_ID += 1
            new_key = f'cluster_{cluster_ID}'
        
        return new_key
    
    def update_clusters(self, cluster1, cluster2):
        """
        Update clusters dictionary
        """
        if not cluster1.startswith('cluster') and not cluster2.startswith('cluster'):
            key = self.get_new_cluster_key()
            self.clusters[key] = [cluster1, cluster2]
            del self.clusters[cluster1]
            del self.clusters[cluster2]

        elif cluster1.startswith('cluster') and not cluster2.startswith('cluster'):    
            key = cluster1
            self.clusters[key].append(cluster2)
            del self.clusters[cluster2]

        elif not cluster1.startswith('cluster') and cluster2.startswith('cluster'):    
            key = cluster2
            self.clusters[key].append(cluster1)
            del self.clusters[cluster1]

        else: #cluster1.startswith('cluster') and cluster2.startswith('cluster'):    
            cluster1_ID = int(cluster1.split('_')[1]) 
            cluster2_ID = int(cluster2.split('_')[1]) 
            cluster_ID = min(cluster1_ID, cluster2_ID)
            del_cluster_ID = max(cluster1_ID, cluster2_ID)
            key = f'cluster_{cluster_ID}'
            del_key = f'cluster_{del_cluster_ID}'
            self.clusters[key].extend(self.clusters[del_key])
            del self.clusters[del_key]

        return key

    def update_neighbors(self, cluster1, cluster2, cluster1_elements, cluster2_elements, key):
        """
        Repopulate active_neighbor_strengths with pairs that had their strength updated 
        or simply update them if they are already present (same code for both cases)
        """

        # Delete neighbor relation between the merged clusters
        if (cluster1, cluster2) in self.neighbor_strengths:
            del self.neighbor_strengths[(cluster1, cluster2)]
            del self.active_neighbor_strengths[(cluster1, cluster2)]
        else:
            del self.neighbor_strengths[(cluster2, cluster1)]
            del self.active_neighbor_strengths[(cluster2, cluster1)]

        # Update neighbors strenghts
        for key_temp in list(self.neighbor_strengths.keys()):
            neighbor1, neighbor2 = key_temp

            if neighbor1 in (cluster1, cluster2) or neighbor2 in (cluster1, cluster2):
                if neighbor1 in (cluster1, cluster2):
                    neighbor_outside_cluster = neighbor2
                    neighbor_inside_cluster = neighbor1
                    new_key = (key, neighbor_outside_cluster)
                    inverse_new_key = (neighbor_outside_cluster, key)
                else:
                    neighbor_outside_cluster = neighbor1
                    neighbor_inside_cluster = neighbor2
                    new_key = (neighbor_outside_cluster, key)
                    inverse_new_key = (key, neighbor_outside_cluster)

                neighbor_outside_cluster_col = self.clusters[neighbor_outside_cluster]
                neighbor_inside_cluster_col = cluster1_elements if neighbor_inside_cluster == cluster1 else cluster2_elements
                new_neighbor_col = cluster1_elements if neighbor_outside_cluster == cluster1 else cluster2_elements
                
                # Check if 'neighbor_inside_cluster' was previously a cluster,
                # and its index matches the index of the new cluster.
                # note: inverse_new_key != key_temp always
                if (new_key == key_temp): 
                    strength = self.get_neighbor_strength(neighbor_outside_cluster_col, neighbor_inside_cluster_col, new_neighbor_col, key_temp)
                    if ~np.isnan(strength):
                        self.neighbor_strengths[key_temp] = strength 
                        self.active_neighbor_strengths[key_temp] = strength # whether key_temp was there or not
                else:
                    # Check that the same strength is not already updated (because of a common neighbor) 
                    # or that it won't be updated later (by entering condition (new_key == key_temp))
                    if inverse_new_key not in self.neighbor_strengths and new_key not in self.neighbor_strengths:
                        strength = self.get_neighbor_strength(neighbor_outside_cluster_col, neighbor_inside_cluster_col, new_neighbor_col, key_temp)
                        if ~np.isnan(strength):
                            self.neighbor_strengths[new_key] = strength
                            self.active_neighbor_strengths[new_key] = strength

                    del self.neighbor_strengths[key_temp]
                    if key_temp in self.active_neighbor_strengths:
                        del self.active_neighbor_strengths[key_temp]

    def get_neighbor_strength(self, neighbor_outside_cluster_col, neighbor_inside_cluster_col, new_neighbor_col, key_temp):
        if self.linkage == 'centroid':
            centroid1 = self.targets_df[neighbor_inside_cluster_col + new_neighbor_col].mean(axis=1)
            centroid2 = self.targets_df[neighbor_outside_cluster_col].mean(axis=1) 
            
            mask1 = ~np.isnan(centroid1)
            mask2 = ~np.isnan(centroid2)
            mask = mask1 & mask2
            strength = np.corrcoef(centroid1[mask], centroid2[mask])[0, 1]

        elif self.linkage == 'average':
            strength_1 = self.neighbor_strengths[key_temp]
            e1_l = len(neighbor_outside_cluster_col)
            e2_l = len(neighbor_inside_cluster_col)
            l1 = e1_l * e2_l

            strength_2 = []
            for e1 in neighbor_outside_cluster_col:
                e1_mask = ~np.isnan(self.targets_df[e1])
                for e2 in new_neighbor_col:
                    e2_mask = ~np.isnan(self.targets_df[e2])
                    mask = e1_mask & e2_mask
                    strength_temp = np.corrcoef(self.targets_df[e1][mask], self.targets_df[e2][mask])[0, 1]
                    strength_2.append(strength_temp)
                    
            strength_2 = np.nanmean(strength_2)      
            l2 = len(neighbor_outside_cluster_col) * len(new_neighbor_col) 
            strength = (l1 * strength_1 + l2 * strength_2) / (l1 + l2)

        elif self.linkage == 'complete':
            strength = self.neighbor_strengths[key_temp]

            for e1 in neighbor_outside_cluster_col:
                e1_mask = ~np.isnan(self.targets_df[e1])
                for e2 in new_neighbor_col:
                    e2_mask = ~np.isnan(self.targets_df[e2])
                    mask = e1_mask & e2_mask
                    strength_temp = np.corrcoef(self.targets_df[e1][mask], self.targets_df[e2][mask])[0, 1]

                    if strength_temp < strength:
                        strength = strength_temp

        else:
            raise ValueError("linkage parameter must be 'centroid', 'average' or 'complete'.")

        return strength.astype('float32')

    def compute_clusters(self):
        print("Computing clusters...")
        total = len(self.targets_df.columns)
        progress_bar = tqdm(total=total, miniters=int(total/10), maxinterval=60*60*4, position=0, smoothing=0.1)
        
        # Get neighbor pairs sorted by their strengths  
        # handle np.nan values from np.corrcoef
        get_value = lambda key: self.active_neighbor_strengths.get(key)
        custom_sort_key = lambda key: float('-inf') if np.isnan(get_value(key)) else get_value(key)
        sorted_strengths = sorted(self.active_neighbor_strengths.keys(), key=custom_sort_key, reverse=True)
        
        # Iterate until there are no pairs that can be merged into a cluster 
        while len(self.active_neighbor_strengths) != 0:
    
            for pair in sorted_strengths:
                (cluster1, cluster2) = pair

                # Check if the pair can be merged 
                if self.check_aggregation(cluster1, cluster2):  
                    cluster1_elements = self.clusters[cluster1]
                    cluster2_elements = self.clusters[cluster2]              
                    # Update clusters dictionary with the new cluster and return its key
                    cluster_key = self.update_clusters(cluster1, cluster2)
                    
                    # Update neighbor_strengths and active_neighbor_strengths dictionaries with new strengths and neighbors
                    self.update_neighbors(cluster1, cluster2, cluster1_elements, cluster2_elements, cluster_key) #eventually add new pairs to active_neighbor_strengths
                    
                    # Get sorted pairs only of clusters which could be merged
                    get_value = lambda key: self.active_neighbor_strengths.get(key)
                    custom_sort_key = lambda key: float('-inf') if np.isnan(get_value(key)) else get_value(key)
                    sorted_strengths = sorted(self.active_neighbor_strengths.keys(), key=custom_sort_key, reverse=True)        

                    progress_bar.update()
                    break
                
                else: 
                    # Remove the pair from active_neighbor_strengths so that, if also its neighbors do not change,
                    # in the next iteration we don't need to consider it during sorting and to run check_aggregation  
                    del self.active_neighbor_strengths[pair]

            else:
                break

        progress_bar.close()




        """
                if self.cross_validation:
            preds1 = preds2 = preds_aggr = np.full(len(y1), np.nan)

            kf = KFold(n_splits=16)
            for train_index, test_index in kf.split(all_indices):
                x1_train = x1.loc[train_index].values
                y1_train = y1.loc[train_index].values
                x1_test = x1.loc[test_index].values
                y1_test = y1.loc[test_index].values
                
                x2_train = x2.loc[train_index].values
                y2_train = y2.loc[train_index].values
                x2_test = x2.loc[test_index].values
                y2_test = y2.loc[test_index].values

                x_aggr_train = x_aggr.loc[train_index].values
                y_aggr_train = y_aggr.loc[train_index].values       
                x_aggr_test = x_aggr.loc[test_index].values
                y_aggr_test = y_aggr.loc[test_index].values        

                def drop_nan(X, y):
                    valid_indices = ~pd.isna(y) & ~np.any(pd.isna(X), axis=1)
                    y = y[valid_indices]
                    X = X[valid_indices]
                    return X, y, valid_indices

                x1_train, y1_train, _ = drop_nan(x1_train, y1_train)
                x1_test, y1_test, valid_indices_test1 = drop_nan(x1_test, y1_test)
                x2_train, y2_train, _ = drop_nan(x2_train, y2_train)
                x2_test, y2_test, valid_indices_test2 = drop_nan(x2_test, y2_test)
                x_aggr_train, y_aggr_train, _ = drop_nan(x_aggr_train, y_aggr_train)
                x_aggr_test, y_aggr_test, valid_indices_test_aggr = drop_nan(x_aggr_test, y_aggr_test)                

                target1_regr = LinearRegression()
                target2_regr = LinearRegression()
                aggr_regr = LinearRegression()

                # no need to standardize for a linear regression
                target1_regr.fit(x1_train, y1_train)
                target2_regr.fit(x2_train, y2_train)
                aggr_regr.fit(x_aggr_train, y_aggr_train)

                # Predict without NaN values
                y_pred1_nonan = target1_regr.predict(x1_test)
                y_pred1 = np.full(len(valid_indices_test1), np.nan)
                y_pred1[valid_indices_test1] = y_pred1_nonan
                preds1[test_index] = y_pred1.astype('float32')

                y_pred2_nonan = target2_regr.predict(x2_test)
                y_pred2 = np.full(len(valid_indices_test2), np.nan)
                y_pred2[valid_indices_test2] = y_pred2_nonan
                preds2[test_index] = y_pred2.astype('float32')

                y_pred_aggr_nonan = aggr_regr.predict(x_aggr_test)
                y_pred_aggr = np.full(len(valid_indices_test_aggr), np.nan)
                y_pred_aggr[valid_indices_test_aggr] = y_pred_aggr_nonan
                preds_aggr[test_index] = y_pred_aggr.astype('float32')

            x1, x2, x_aggr, y1, y2, y_aggr, common_indices = self.drop_observations_with_nan(x1, x2, x_aggr, y1, y2, y_aggr)
            preds1 = np.array([preds1[i] for i in common_indices])
            preds2 = np.array([preds2[i] for i in common_indices])
            preds_aggr = np.array([preds_aggr[i] for i in common_indices])
        """