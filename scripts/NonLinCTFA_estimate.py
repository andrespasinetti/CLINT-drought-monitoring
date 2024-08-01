import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
from itertools import combinations
import copy
from sklearn.preprocessing import StandardScaler
import multiprocessing
import random
import math

class NonLinCTFA_estimate():
    """
    Attributes:
    - features: List of DataFrames containing features.
    - target: DataFrame containing target values.
    - clusters: Dictionary mapping cluster IDs to a list of elements.
    - neighbours: List of neighbour pairs.
    - neighbour_strengths: Dictionary storing strengths for each neighbour pair.
    - active_neighbour_strengths: Dictionary storing strengths only for pairs that could be aggregated in the next iteration (to optimize code)
    """
        
    def __init__(self, features, target, neighbours, linkage, epsilon=0, weights=None):

        self.features = copy.deepcopy(features)
        """
        # Features preprocessing
        for i in range(len(self.features)):
            scaler = StandardScaler() #it deals with nan values
            features[i] = pd.DataFrame(scaler.fit_transform(features[i]), columns=features[i].columns)
        """
        
        if type(target)==str:
            pd.read_csv(target)
        else: self.target = target.copy(deep=True)

        # Initialize clusters: each element as a cluster
        self.clusters = {subid:[subid] for subid in target.columns}

        # If neighbours are not provided, generate all possible neighbour pairs
        if neighbours:
            self.neighbours = copy.deepcopy(neighbours)
        else:
            self.neighbours = set(combinations(list(self.target.columns), 2))

        # Get strengths of each element
        self.neighbour_strengths = self.initialize_neighbour_strengths()
        self.active_neighbour_strengths = copy.deepcopy(self.neighbour_strengths)
        self.linkage = linkage
        self.epsilon = epsilon
        self.weights = weights

    def get_clusters(self):
        return self.clusters


    def compute_strength(self, cluster1, cluster2):
        """
        Compute the strength between cluster1 and cluster2. 
        In this case we consider as strength the correlation between the two vectors.
        """
        cluster1_value = self.target[self.clusters[cluster1]].mean(axis=1)
        cluster2_value = self.target[self.clusters[cluster2]].mean(axis=1)

        # Create a mask to filter nan values
        cluster1_mask = ~np.isnan(cluster1_value)
        cluster2_mask = ~np.isnan(cluster2_value)
        mask = cluster1_mask & cluster2_mask
        
        strength = np.corrcoef(cluster1_value[mask], cluster2_value[mask])[0, 1]
        return strength


    def initialize_neighbour_strengths(self):
        neighbour_strengths = {}

        print("Computing initial neighbours strengths...")
        for neighbour_pair in tqdm(self.neighbours, miniters=int(len(self.neighbours)/10), maxinterval=60*10, position=0, smoothing=0.01):
            neighbour1, neighbour2 = neighbour_pair
            strength = self.compute_strength(neighbour1, neighbour2)
            neighbour_strengths[neighbour_pair] = strength

        print()
        return  neighbour_strengths 


    """
    def average_with_nan(self, vec1, vec2):
    averaged_vec = np.where(np.isnan(vec1), vec2, np.where(np.isnan(vec2), vec1, (vec1 + vec2) / 2))
    return averaged_vec
    """

    def weighted_average(self, data, weights):
        # Create a masked array where the data is masked where NaNs are present
        masked_data = np.ma.array(data, mask=np.isnan(data))
        result = np.ma.average(masked_data, axis=1, weights=weights)

        # Fill masked results with NaN for any segments where all values were NaN
        return result.filled(np.nan).astype('float32')  # Converts masked values back to NaN
    

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

        if self.weights is not None:
            weighted_averages = []
            for feature in self.features:
                feature_cluster = feature[self.clusters[cluster1] + self.clusters[cluster2]]
                weights_cluster = self.weights.loc[self.weights['SUBID'].isin(self.last_clusters[cluster1] + self.clusters[cluster2]), 'AREA'].values
                weighted_average = self.weighted_average(feature_cluster, weights_cluster)
                weighted_averages.append(weighted_average)
            
            x_aggr = pd.concat(weighted_averages, axis=1)

        else:
            x_aggr = pd.concat([feature[self.clusters[cluster1] + self.clusters[cluster2]].mean(axis=1) for feature in self.features], axis=1)
        
        x_aggr = x_aggr.drop(columns=nan_features_to_drop)

        return x1, x2, x_aggr 
    

    def prepare_target(self, cluster1, cluster2):
        y1 = self.target[self.clusters[cluster1]].mean(axis=1)
        y2 = self.target[self.clusters[cluster2]].mean(axis=1)
        y1 = y1 - np.nanmean(y1)
        y2 = y2 - np.nanmean(y2)

        if self.weights is not None:
            target_cluster = self.target[self.clusters[cluster1] + self.clusters[cluster2]]
            weights_cluster = self.weights.loc[self.weights['SUBID'].isin(self.last_clusters[cluster1] + self.clusters[cluster2]), 'AREA'].values
            y_aggr = self.weighted_average(target_cluster, weights_cluster)
        else:
            y_aggr = self.target[self.clusters[cluster1] + self.clusters[cluster2]].mean(axis=1)
        
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

        return x1, x2, x_aggr, y1, y2, y_aggr


    def compute_VALscores(self, cluster1, cluster2):
        x1, x2, x_aggr = self.prepare_features(cluster1, cluster2)
        y1, y2, y_aggr = self.prepare_target(cluster1, cluster2)

        x1, x2, x_aggr, y1, y2, y_aggr = self.drop_observations_with_nan(x1, x2, x_aggr, y1, y2, y_aggr)
        
        # features "x" already standardized when initializing the class
        target1_regr = LinearRegression()
        target2_regr = LinearRegression()
        aggr_regr = LinearRegression()

        target1_regr.fit(x1,y1)
        target2_regr.fit(x2,y2)
        aggr_regr.fit(x_aggr,y_aggr)

        # we are now ready to perform the three linear regressions: the two individual ones and the one with aggregated targets
        # if for both it is convenient to aggregate, we do so 

        ### variance ###
        D = x1.shape[1] 
        n = x1.shape[0]
        preds1 = target1_regr.predict(x1)
        preds2 = target2_regr.predict(x2)
        preds_aggr = aggr_regr.predict(x_aggr)
        residuals1 = y1 - preds1
        residuals2 = y2 - preds2
        residuals_aggr = y_aggr - preds_aggr
        s_squared1 = np.dot(residuals1.reshape(1,n),residuals1)/(n-D-1)
        s_squared2 = np.dot(residuals2.reshape(1,n),residuals2)/(n-D-1)
        s_squared_aggr = np.dot(residuals_aggr.reshape(1,n),residuals_aggr)/(n-D-1)

        var1 = s_squared1*D/(n-1)
        var2 = s_squared2*D/(n-1)
        var_aggr = s_squared_aggr*D/(n-1)

        #aggr_r2 = cross_val_score(aggr_regr, x_aggr, y, cv=TimeSeriesSplit(-self.n_val), scoring='r2')
        #bivariate_r2 = cross_val_score(bivariate_regr, x, y, cv=TimeSeriesSplit(-self.n_val), scoring='r2')

        ### bias ### 
        r2_1 = r2_score(y1,preds1)
        r2_2 = r2_score(y2,preds2)
        r2_aggr = r2_score(y_aggr,preds_aggr)
        
        ### the following two are not needed but they can help to monitor the performances
        #r2_aggr_1 = r2_score(y1,preds_aggr)
        #r2_aggr_2 = r2_score(y2,preds_aggr)

        ### all equations of biases, not needed for the final threshold
        #bias1 = (np.var(y1,ddof=1)-s_squared1)*(1-r2_1)
        #bias2 = (np.var(y2,ddof=1)-s_squared2)*(1-r2_2)

        s_squaredF1 = (np.var(y1,ddof=1)-s_squared1)
        s_squaredF2 = (np.var(y2,ddof=1)-s_squared2)
        s_squaredFaggr = (np.var(y_aggr,ddof=1)-s_squared_aggr)

        #bias_aggr1 = s_squaredF1 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2))
        #bias_aggr2 = s_squaredF2 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF2*(r2_2) - s_squaredF1*(r2_1))

        ### these are the needed ones
        r2_1_weighted = r2_1*s_squaredF1
        r2_2_weighted = r2_2*s_squaredF2
        r2_aggr_weighted = r2_aggr*s_squaredFaggr

        #print(var1,var2,var_aggr,bias1,bias2,bias_aggr1,bias_aggr2)
        #print(s_squaredF1,s_squaredFaggr*r2_aggr, 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2)), s_squaredF2, 0.5*(s_squaredF2*(r2_2) - s_squaredF1*(r2_1)))

        #print(var1-var_aggr,var2-var_aggr,r2_aggr_weighted-0.5*r2_1_weighted-0.5*r2_2_weighted)
        #print(f'Basins: {cluster1, cluster2}, \nR2 coefficients: {r2_1,r2_2}, \naggregating: {r2_aggr,r2_aggr_1,r2_aggr_2}\n',flush=True)

        '''
        var1 - var_aggr + bias1 - bias_aggr1 
        var1 - var_aggr + (np.var(y1,ddof=1)-s_squared1)*(1-r2_1) - s_squaredF1 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2)) 
        var1 - var_aggr + (np.var(y1,ddof=1)-s_squared1)*(1-r2_1) - (np.var(y1,ddof=1)-s_squared1) - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2)) 
        var1 - var_aggr - (np.var(y1,ddof=1)-s_squared1)*r2_1 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2)) 


        var1 - var_aggr + r2_aggr_weighted - 0.5*(r2_1_weighted+r2_2_weighted)
        var1 + r2_aggr_weighted - var_aggr - 0.5*(r2_1_weighted+r2_2_weighted)
        '''
        return var1,var2,var_aggr,r2_1_weighted,r2_2_weighted,r2_aggr_weighted



    def check_aggregation(self, cluster1, cluster2):
        var1,var2,var_aggr,r2_1_weighted,r2_2_weighted,r2_aggr_weighted = self.compute_VALscores(cluster1, cluster2)
        '''
        if self.weights is not None:
            cluster1_elements = self.clusters[cluster1]
            cluster2_elements = self.clusters[cluster2]    
            cluster1_weight = np.sum(self.weights.loc[self.weights['SUBID'].isin(cluster1_elements), 'AREA'])
            cluster2_weight = np.sum(self.weights.loc[self.weights['SUBID'].isin(cluster2_elements), 'AREA'])
        else:
            cluster1_weight = cluster2_weight = 0.5

        delta_var1 = var1 - var_aggr
        delta_var2 = var2 - var_aggr
        delta_bias1 = 
        '''
        
        if (var1+r2_aggr_weighted - (var_aggr+0.5*(r2_1_weighted+r2_2_weighted)) >= self.epsilon) & (var2+r2_aggr_weighted - (var_aggr+0.5*(r2_1_weighted+r2_2_weighted)) >= self.epsilon): 
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


    def update_neighbours(self, cluster1, cluster2, cluster1_elements, cluster2_elements, key):
        """
        Repopulate active_neighbour_strengths with pairs that had their strength updated 
        or simply update them if they are already present (same code for both cases)
        """

        # Delete neighbour relation between the merged clusters
        if (cluster1, cluster2) in self.neighbour_strengths:
            del self.neighbour_strengths[(cluster1, cluster2)]
            del self.active_neighbour_strengths[(cluster1, cluster2)]
        else:
            del self.neighbour_strengths[(cluster2, cluster1)]
            del self.active_neighbour_strengths[(cluster2, cluster1)]

        # Update neighbours strenghts
        for key_temp in list(self.neighbour_strengths.keys()):
            neighbour1, neighbour2 = key_temp

            if neighbour1 in (cluster1, cluster2) or neighbour2 in (cluster1, cluster2):
                if neighbour1 in (cluster1, cluster2):
                    neighbour_outside_cluster = neighbour2
                    neighbour_inside_cluster = neighbour1
                    new_key = (key, neighbour_outside_cluster)
                    inverse_new_key = (neighbour_outside_cluster, key)
                else:
                    neighbour_outside_cluster = neighbour1
                    neighbour_inside_cluster = neighbour2
                    new_key = (neighbour_outside_cluster, key)
                    inverse_new_key = (key, neighbour_outside_cluster)

                neighbour_outside_cluster_col = self.clusters[neighbour_outside_cluster]
                neighbour_inside_cluster_col = cluster1_elements if neighbour_inside_cluster==cluster1 else cluster2_elements
                new_neighbour_col = cluster1_elements if neighbour_outside_cluster==cluster1 else cluster2_elements
                
                # Check if 'neighbour_inside_cluster' was previously a cluster,
                # and its index matches the index of the new cluster.
                # note: inverse_new_key != key_temp always
                if (new_key == key_temp): 
                    if self.linkage=='average':
                        strength = self.get_neighbour_strength_average(neighbour_outside_cluster_col, neighbour_inside_cluster_col, new_neighbour_col)
                    if ~np.isnan(strength):
                        self.neighbour_strengths[key_temp] = strength 
                        self.active_neighbour_strengths[key_temp] = strength # whether key_temp was there or not
                else:
                    # Check that the same strength is not already updated (because of a common neighbour) 
                    # or that it won't be updated later (by entering condition (new_key == key_temp))
                    if inverse_new_key not in self.neighbour_strengths and new_key not in self.neighbour_strengths:
                        if self.linkage=='average':
                            strength = self.get_neighbour_strength_average(neighbour_outside_cluster_col, neighbour_inside_cluster_col, new_neighbour_col)
                        if ~np.isnan(strength):
                            self.neighbour_strengths[new_key] = strength
                            self.active_neighbour_strengths[new_key] = strength

                    del self.neighbour_strengths[key_temp]
                    if key_temp in self.active_neighbour_strengths:
                        del self.active_neighbour_strengths[key_temp]


    def extract_percentage_with_replacement(self, input_list):
        num_total = len(input_list)
        num_elements_to_extract = math.ceil(math.sqrt(num_total))
        extracted_elements = random.choices(input_list, k=num_elements_to_extract)
        return extracted_elements


    def get_neighbour_strength_average(self, neighbour_outside_cluster_col, neighbour_inside_cluster_col, new_neighbour_col):
        columns_outside = neighbour_outside_cluster_col
        columns_outside = self.extract_percentage_with_replacement(columns_outside)
        columns_new_cluster = neighbour_inside_cluster_col + new_neighbour_col
        columns_new_cluster = self.extract_percentage_with_replacement(columns_new_cluster)

        strength_list = []
        w_list = []
        for e1 in columns_outside:
            e1_mask = ~np.isnan(self.target[e1])

            for e2 in columns_new_cluster:
                e2_mask = ~np.isnan(self.target[e2])
                mask = e1_mask & e2_mask

                if self.weights is not None:
                    e1_w = self.weights.loc[self.weights['SUBID']==e1, 'AREA'].values
                    e2_w = self.weights.loc[self.weights['SUBID']==e2, 'AREA'].values 
                    e12_w = e1_w + e2_w 
                    strength_temp = e12_w * np.corrcoef(self.df[e1][mask], self.df[e2][mask])[0, 1]
                    w_list.append(e12_w)
                else:
                    strength_temp = np.corrcoef(self.target[e1][mask], self.target[e2][mask])[0, 1]

                strength_list.append(strength_temp)

        if self.weights is not None:
            strength = np.sum(strength_list)/np.sum(w_list)
        else:
            strength = np.nanmean(strength_list)    

        return strength


    def compute_clusters(self):
        # Get neighbour pairs sorted by their strengths  
        # handle np.nan values from np.corrcoef
        get_value = lambda key: self.active_neighbour_strengths.get(key)
        custom_sort_key = lambda key: float('-inf') if np.isnan(get_value(key)) else get_value(key)
        sorted_strengths = sorted(self.active_neighbour_strengths.keys(), key=custom_sort_key, reverse=True)
        
        print("Computing clusters...")
        total = len(self.target.columns)
        progress_bar = tqdm(total=total, miniters=int(total/10), maxinterval=60*60*4, position=0, smoothing=0.1)
        
        # Iterate until there are no pairs that can be merged into a cluster 
        while len(self.active_neighbour_strengths) != 0:
    
            for pair in sorted_strengths:
                (cluster1, cluster2) = pair

                # Check if the pair can be merged 
                if self.check_aggregation(cluster1, cluster2):  
                    cluster1_elements = self.clusters[cluster1]
                    cluster2_elements = self.clusters[cluster2]              
                    # Update clusters dictionary with the new cluster and return its key
                    cluster_key = self.update_clusters(cluster1, cluster2)
                    
                    # Update neighbour_strengths and active_neighbour_strengths dictionaries with new strengths and neighbours
                    self.update_neighbours(cluster1, cluster2, cluster1_elements, cluster2_elements, cluster_key) #eventually add new pairs to active_neighbour_strengths
                    
                    # Get sorted pairs only of clusters which could be merged
                    get_value = lambda key: self.active_neighbour_strengths.get(key)
                    custom_sort_key = lambda key: float('-inf') if np.isnan(get_value(key)) else get_value(key)
                    sorted_strengths = sorted(self.active_neighbour_strengths.keys(), key=custom_sort_key, reverse=True)        

                    progress_bar.update()
                    break
                
                else: 
                    # Remove the pair from active_neighbour_strengths so that, if also its neighbours do not change,
                    # in the next iteration we don't need to consider it during sorting and to run check_aggregation  
                    del self.active_neighbour_strengths[pair]

            else:
                break

        progress_bar.close()


