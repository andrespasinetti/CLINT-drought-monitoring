import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
import math
import random


class Cluster:
    def __init__(self, clusterID, childs, correlation):
        self.clusterID = clusterID
        self.childs = childs
        self.parent = ''
        self.correlation = correlation
        self.dim = sum(child.dim for child in self.childs) if self.childs != '' else 1

    def get_last_childs(self):
        if self.childs == '':
            return [self]
        else: 
            last_childs = []
            for child in self.childs:
                last_childs.extend(child.get_last_childs())  
            return last_childs      

    def set_parent(self, parent):
        self.parent = parent

    def get_last_parent(self):
        if self.parent == '':
            return self
        else: 
            last_parent = self.parent.get_last_parent()
            return last_parent


class HierarchicalClustering:
    '''
        There are two data structures:
        - 'clusters': a list that contains all the Cluster objects created until now, they form the tree structure
        - 'last_clusters': a dictionary that contains the last clusters in the tree, i.e. the first childs (excluding leaves)

        'cluster_mapping' is a dictionary which maps 'last_clusters' keys to 'clusters' IDs 
    '''

    def __init__(self, df, neighbours, linkage, threshold, weights=None, verbose=False):
        self.df = df
        self.neighbours = neighbours
        self.linkage = linkage
        self.threshold = threshold
        self.weights = weights
        self.verbose = verbose
        self.last_clusters = {} # contains the last (biggest) computed clusters (no singletons) until now. each cluster is a list of SUBIDs
        self.neighbours_strengths = self.initialize_neighbours_strengths()
        self.clusters = [Cluster(sub_basin, '', 1) for sub_basin in df.columns] # contains a list of all the clusters in the tree 
        self.cluster_mapping = {}
        self.initial_threshold = 1
        self.cluster_num_internal = 0
        
    def get_linkage(self):
        return self.linkage
        
    def set_initial_threshold(self, initial_threshold):
        self.initial_threshold = initial_threshold

    def set_threshold(self, new_threshold):
        self.threshold = new_threshold

    def get_clustering_count_analysis(self):
        ordered_parents, ordered_correlations, ordered_count, ordered_singletons = self.get_ordered_clusters()

        # Create a single plot with two different y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the first set of data on the first y-axis
        ax1.plot(ordered_correlations, ordered_count, label='No. Clusters', color='blue')
        ax1.plot(ordered_correlations, np.subtract(np.array(ordered_count), np.array(ordered_singletons)), label='No. Clusters - Singletons', color='green')
        ax1.plot(ordered_correlations, ordered_singletons, label='No. Singletons', color='red')
        ax1.set_xlabel('Correlation threshold')
        ax1.set_ylabel('No. Clusters')
        ax1.tick_params(axis='y')

        # Create a secondary y-axis on the same plot
        ax2 = ax1.twinx()

        # Plot the second set of data on the secondary y-axis
        ax2.plot(ordered_correlations, np.subtract(np.array(ordered_count), np.array(ordered_singletons))/np.array(ordered_singletons), label='(No. Clusters - Singletons) / No. Singletons', color='purple')
        ax2.set_ylabel('(No. Clusters - Singletons) / No. Singletons', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        # Combine the legends for both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        ax1.legend(lines, labels, loc='upper left')

        # Set titles for the combined plot
        plt.title('Complete-Linkage, Clusters Count')

        # Show the combined plot
        plt.show()    

    def get_ordered_clusters(self):
        ordered_parents = []
        ordered_correlations = []
        ordered_count = []
        ordered_singletons = []
        
        last_parents = self.get_last_parents()
        min_correlation = 0

        # if min_correlation == 1 it could mean that we have some clusters of different elements with correlation 1 and we want to stop there
        while (last_parents != [] and min_correlation != 1):
            min_correlation = 1
            min_parents = []
            
            for parent in last_parents:
                if parent.correlation < min_correlation:
                    min_correlation = parent.correlation 
                    min_parents = [parent]
                
                elif parent.correlation == min_correlation:      
                    min_parents.append(parent)

            ordered_parents.append(min_parents)
            ordered_correlations.append(min_correlation)
            ordered_count.append(len(last_parents))

            ordered_singletons.append(len([cluster for cluster in last_parents if cluster.childs=='']))

            for min_parent in min_parents:
                last_parents.extend(min_parent.childs)
                last_parents.remove(min_parent)

        return ordered_parents, ordered_correlations, ordered_count, ordered_singletons

    def get_clusters_from_parents(self, parents):
        clusters_from_parents = []
        for parent in parents:
            from_parent = []
            for last_child in parent.get_last_childs():
                from_parent.append(last_child.clusterID)

            clusters_from_parents.append(from_parent)
        return sorted(clusters_from_parents, key=lambda x: len(x), reverse=True)
    
    def get_parents_from_last_parents(self, threshold):
        parents = []
        temp_parents = self.get_last_parents()

        while (temp_parents != []):
            temp2_parents = []
            for parent in temp_parents:
                if parent.correlation >= threshold:
                    parents.append(parent)
                else: 
                    temp2_parents.extend(parent.childs)

            temp_parents = temp2_parents

        return parents

    def get_clusters(self, threshold):
        parents = self.get_parents_from_last_parents(threshold)
        clusters = self.get_clusters_from_parents(parents)
        return clusters

    def get_last_parents(self):
        last_parents = []        

        for cluster in self.clusters:
            if cluster.parent == '':
                last_parents.append(cluster)

        return last_parents    
    
    def get_cluster_by_ID(self, clusterID):
        for cluster in self.clusters:
            if cluster.clusterID == clusterID:
                return cluster

    def create_cluster(self, childs, correlation):
        cluster = Cluster(f'cluster_{len(self.clusters) - len(self.df.columns)}', childs, correlation)
        self.clusters.append(cluster)

    def growth_function(self, x, a, b, c):
        return a * x**2 + b * x + c
    
    def compute_clusters(self):
        current_score = self.initial_threshold

        if self.verbose:
            print('Computing clusters...')

        # Number of iterations
        N = current_score*1000 - int(self.threshold*1000)
        i = 0
        # Arrays to store iteration and elapsed time data
        iteration_data = np.arange(1, N + 1)
        elapsed_time_data = []
        start_time = time.time()

        max_strength = sorted(self.neighbours_strengths.keys(), key=self.neighbours_strengths.get, reverse=True)[0]

        while len(self.neighbours_strengths) != 0 and self.enough_strength(self.neighbours_strengths[max_strength]):
            #print(self.neighbours_strengths[max_strength], flush=True)
            (elem1, elem2) = max_strength
   
            # Update cluster and correlation values
            if elem1.startswith('cluster') and not elem2.startswith('cluster'):
                new_cluster_index = int(elem1.split('_')[1])
                self.last_clusters[new_cluster_index].append(elem2)

                child1 = self.get_cluster_by_ID(self.cluster_mapping[new_cluster_index])
                child2 = self.get_cluster_by_ID(elem2)
                self.create_cluster([child1, child2], self.neighbours_strengths[max_strength])
                self.cluster_mapping[new_cluster_index] = self.clusters[-1].clusterID
                child1.set_parent(self.clusters[-1])
                child2.set_parent(self.clusters[-1])

                del self.neighbours_strengths[max_strength]
                self.refresh_corr_values(elem1, elem2, new_cluster_index)
                
            elif not elem1.startswith('cluster') and elem2.startswith('cluster'):
                new_cluster_index = int(elem2.split('_')[1])
                self.last_clusters[new_cluster_index].append(elem1)

                child1 = self.get_cluster_by_ID(elem1)
                child2 = self.get_cluster_by_ID(self.cluster_mapping[new_cluster_index])
                self.create_cluster([child1, child2], self.neighbours_strengths[max_strength])
                self.cluster_mapping[new_cluster_index] = self.clusters[-1].clusterID
                child1.set_parent(self.clusters[-1])
                child2.set_parent(self.clusters[-1])

                del self.neighbours_strengths[max_strength]
                self.refresh_corr_values(elem1, elem2, new_cluster_index)

            elif not elem1.startswith('cluster') and not elem2.startswith('cluster'):
                self.cluster_mapping[self.cluster_num_internal] = f'cluster_{len(self.clusters) - len(self.df.columns)}'

                # creo un nuovo cluster nel dizionario last_clusters 
                self.last_clusters[self.cluster_num_internal] = [elem1, elem2]

                # creo un nuovo cluster nella lista clusters
                child1 = self.get_cluster_by_ID(elem1) # ID == SUBID
                child2 = self.get_cluster_by_ID(elem2)
                self.create_cluster([child1, child2], self.neighbours_strengths[max_strength])
                child1.set_parent(self.clusters[-1])
                child2.set_parent(self.clusters[-1])

                del self.neighbours_strengths[max_strength]
                self.refresh_corr_values(elem1, elem2, self.cluster_num_internal)
                self.cluster_num_internal += 1
                
            else: # both are clusters
                cluster1_index = int(elem1.split('_')[1])
                cluster2_index = int(elem2.split('_')[1])
                new_cluster_index = min(cluster1_index, cluster2_index)
                to_be_removed = max(cluster1_index, cluster2_index)
                self.last_clusters[new_cluster_index].extend(self.last_clusters[to_be_removed])           

                child1 = self.get_cluster_by_ID(self.cluster_mapping[cluster1_index])
                child2 = self.get_cluster_by_ID(self.cluster_mapping[cluster2_index])
                self.create_cluster([child1, child2], self.neighbours_strengths[max_strength])
                self.cluster_mapping[new_cluster_index] = self.clusters[-1].clusterID
                child1.set_parent(self.clusters[-1])
                child2.set_parent(self.clusters[-1])

                del self.neighbours_strengths[max_strength]   
                self.refresh_corr_values(elem1, elem2, new_cluster_index)

                # do not remove it before because it is needed to recompute the correlation values
                del self.last_clusters[to_be_removed]   


            max_strength = sorted(self.neighbours_strengths.keys(), key=self.neighbours_strengths.get, reverse=True)[0]
            
            if self.verbose:
                if (self.neighbours_strengths[max_strength] < current_score - 0.001):
                    current_score = current_score - 0.001
                    formatted = '%.3f' % current_score

                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    elapsed_time_data.append(elapsed_time)

                    # Fit a curve to the current data
                    try:
                        params, _ = curve_fit(self.growth_function, iteration_data[:len(elapsed_time_data)], elapsed_time_data)
                        remaining_time = self.growth_function(N + 1, *params) - elapsed_time
                    except Exception as e:
                        # If the fit fails, use a default estimate
                        #print(f'Curve fitting failed: {e}')
                        remaining_time = elapsed_time * (N - i)

                    print(f'Iteration {i+1}/{N} - Elapsed Time: {elapsed_time/3600:.1f}h - Remaining Time: {remaining_time/3600:.1f}h')
                    print(formatted)

                    i = i+1
            
        if self.verbose:
            print('Clusters computed.')
        #self.last_clusters = sorted(self.last_clusters, key=lambda x: len(x), reverse=True)

    def enough_strength(self, strength):
        return strength >= self.threshold


    def initialize_neighbours_strengths(self):
        if self.verbose:
            print('Computing initial neighbours strengths...')
            progress_bar = tqdm(total=len(self.neighbours), position=0, leave=True, smoothing=0)

        neighbours_strengths = {}
        for neighbour_pair in self.neighbours:
            neighbour1, neighbour2 = neighbour_pair
            neighbour1_mask = ~np.isnan(self.df[neighbour1])
            neighbour2_mask = ~np.isnan(self.df[neighbour2])

            mask = neighbour1_mask & neighbour2_mask
            strength = np.corrcoef(self.df[neighbour1][mask], self.df[neighbour2][mask])[0, 1]
            neighbours_strengths[neighbour_pair] = strength
            if self.verbose:
                progress_bar.update(1)  
            
        if self.verbose:
            print('neighbours strengths computed.')
        return neighbours_strengths

    def weighted_average(self, data, weights):
        # Create a masked array where the data is masked where NaNs are present
        masked_data = np.ma.array(data, mask=np.isnan(data))
        result = np.ma.average(masked_data, axis=1, weights=weights)

        # Fill masked results with NaN for any segments where all values were NaN
        return result.filled(np.nan).astype('float32')  # Converts masked values back to NaN

    def get_neighbours_strength_centroid(self, elem, cluster_index):
        if self.weights is not None:
            cluster_weights1 = self.weights.loc[self.weights['SUBID'].isin(self.last_clusters[cluster_index]), 'AREA'].values
            cluster_weights2 = self.weights.loc[self.weights['SUBID'].isin(self.last_clusters[int(elem.split('_')[1])]), 'AREA'].values if elem.startswith('cluster') else self.weights.loc[self.weights['SUBID'] == elem, 'AREA'].values    
            centroid1 = self.weighted_average(self.df[self.last_clusters[cluster_index]], cluster_weights1)
            centroid2 = self.weighted_average(self.df[self.last_clusters[int(elem.split('_')[1])]], cluster_weights2) if elem.startswith('cluster') else self.df[elem]
        else:
            centroid1 = self.df[self.last_clusters[cluster_index]].mean(axis=1)
            centroid2 = self.df[self.last_clusters[int(elem.split('_')[1])]].mean(axis=1) if elem.startswith('cluster') else self.df[elem]
        
        mask1 = ~np.isnan(centroid1)
        mask2 = ~np.isnan(centroid2)
        mask = mask1 & mask2
        strength = np.corrcoef(centroid1[mask], centroid2[mask])[0, 1]
        
        return strength.astype('float32')
    
    def get_neighbours_strength_complete(self, elem_outside_cluster, elem_new_neighbour, key):
        strength = self.neighbours_strengths[key]
        
        if elem_new_neighbour.startswith('cluster'):
            columns_neighbour = self.last_clusters[int(elem_new_neighbour.split('_')[1])]
        else:
            columns_neighbour = [elem_new_neighbour]

        if elem_outside_cluster.startswith('cluster'):
            columns_elem = self.last_clusters[int(elem_outside_cluster.split('_')[1])]
        else:
            columns_elem = [elem_outside_cluster]

        for e1 in columns_elem:
            e1_mask = ~np.isnan(self.df[e1])
            for e2 in columns_neighbour:
                e2_mask = ~np.isnan(self.df[e2])
                mask = e1_mask & e2_mask

                strength_temp = np.corrcoef(self.df[e1][mask], self.df[e2][mask])[0, 1]
                if strength_temp < strength:
                    strength = strength_temp
                               
        return strength.astype('float32')

    def get_neighbours_strength_average(self, elem_outside_cluster, elem_new_neighbour, key):
        strength_1 = self.neighbours_strengths[key]
        e1, e2 = key
        if self.weights is not None:
            e1_subids = self.last_clusters[int(e1.split('_')[1])] if e1.startswith('cluster') else [e1]
            e2_subids = self.last_clusters[int(e2.split('_')[1])] if e2.startswith('cluster') else [e2]
            e1_w = self.weights.loc[self.weights['SUBID'].isin(e1_subids), 'AREA'].values
            e2_w = self.weights.loc[self.weights['SUBID'].isin(e2_subids), 'AREA'].values
            # Dividend for the weighted correlation between e1 and e2
            w1 = np.sum(e1_w)*len(e2_w) + np.sum(e2_w)*len(e1_w)
        else:
            e1_l = len(self.last_clusters[int(e1.split('_')[1])]) if e1.startswith('cluster') else 1
            e2_l = len(self.last_clusters[int(e2.split('_')[1])]) if e2.startswith('cluster') else 1
            l1 = e1_l * e2_l

        columns_neighbour = self.last_clusters[int(elem_new_neighbour.split('_')[1])] if elem_new_neighbour.startswith('cluster') else [elem_new_neighbour]
        columns_elem = self.last_clusters[int(elem_outside_cluster.split('_')[1])] if elem_outside_cluster.startswith('cluster') else [elem_outside_cluster]

        strength_2_list = []
        w2_list = []
        for e1 in columns_elem:
            e1_mask = ~np.isnan(self.df[e1])
            for e2 in columns_neighbour:
                e2_mask = ~np.isnan(self.df[e2])
                mask = e1_mask & e2_mask
                if self.weights is not None:
                    e1_w = self.weights.loc[self.weights['SUBID']==e1, 'AREA'].values
                    e2_w = self.weights.loc[self.weights['SUBID']==e2, 'AREA'].values 
                    e12_w = e1_w + e2_w 
                    strength_temp = e12_w * np.corrcoef(self.df[e1][mask], self.df[e2][mask])[0, 1]
                    w2_list.append(e12_w)
                else:
                    strength_temp = np.corrcoef(self.df[e1][mask], self.df[e2][mask])[0, 1]
                strength_2_list.append(strength_temp)
        strength_2_sum = np.nansum(strength_2_list)

        if self.weights is not None:
            w2 = np.nansum(w2_list)
            strength = (w1 * strength_1 + strength_2_sum) / (w1 + w2)
        else:
            l2 = len(strength_2_list) 
            strength = (l1 * strength_1 + strength_2_sum) / (l1 + l2)

        return strength.astype('float32')


    def extract_percentage_with_replacement(self, input_list):
        num_total = len(input_list)
        num_elements_to_extract = math.ceil(math.sqrt(num_total))
        extracted_elements = random.choices(input_list, k=num_elements_to_extract)
        return extracted_elements


    def get_neighbour_strength_estimated_average(self, elem_outside_cluster, cluster_index):
        columns_outside = self.last_clusters[int(elem_outside_cluster.split('_')[1])] if elem_outside_cluster.startswith('cluster') else [elem_outside_cluster]
        columns_outside = self.extract_percentage_with_replacement(columns_outside)
        columns_new_cluster = self.last_clusters[cluster_index]
        columns_new_cluster = self.extract_percentage_with_replacement(columns_new_cluster)

        strength_list = []
        w_list = []
        for e1 in columns_outside:
            e1_mask = ~np.isnan(self.df[e1])

            for e2 in columns_new_cluster:
                e2_mask = ~np.isnan(self.df[e2])
                mask = e1_mask & e2_mask

                if self.weights is not None:
                    e1_w = self.weights.loc[self.weights['SUBID']==e1, 'AREA'].values
                    e2_w = self.weights.loc[self.weights['SUBID']==e2, 'AREA'].values 
                    e12_w = e1_w + e2_w 
                    strength_temp = e12_w * np.corrcoef(self.df[e1][mask], self.df[e2][mask])[0, 1]
                    w_list.append(e12_w)
                else:
                    strength_temp = np.corrcoef(self.df[e1][mask], self.df[e2][mask])[0, 1]

                strength_list.append(strength_temp)

        if self.weights is not None:
            strength = np.sum(strength_list)/np.sum(w_list)
        else:
            strength = np.nanmean(strength_list)    

        return strength

    def refresh_corr_values(self, elem1_cluster, elem2_cluster, cluster_index):
        for key in list(self.neighbours_strengths.keys()):
            elem1, elem2 = key

            # se elem1 appartiene al nuovo cluster, allora aggiorno il legame elem2-cluster    
            if elem1 in (elem1_cluster, elem2_cluster) or elem2 in (elem1_cluster, elem2_cluster):
                if elem1 in (elem1_cluster, elem2_cluster):
                    elem_in_cluster = elem1
                    elem_outside_cluster = elem2                
                    new_key = (f'cluster_{cluster_index}', elem_outside_cluster)
                    inverse_new_key = (elem_outside_cluster, f'cluster_{cluster_index}')

                elif elem2 in (elem1_cluster, elem2_cluster):
                    elem_in_cluster = elem2
                    elem_outside_cluster = elem1
                    new_key = (elem_outside_cluster, f'cluster_{cluster_index}')
                    inverse_new_key = (f'cluster_{cluster_index}', elem_outside_cluster)

                if self.linkage == 'complete':
                    elem_new_neighbour = elem2_cluster if elem_in_cluster == elem1_cluster else elem1_cluster 
                    strength = self.get_neighbours_strength_complete(elem_outside_cluster, elem_new_neighbour, key)
                elif self.linkage == 'centroid':
                    strength = self.get_neighbours_strength_centroid(elem_outside_cluster, cluster_index)
                elif self.linkage == 'average':
                    elem_new_neighbour = elem2_cluster if elem_in_cluster == elem1_cluster else elem1_cluster 
                    strength = self.get_neighbours_strength_average(elem_outside_cluster, elem_new_neighbour, key)
                elif self.linkage == 'estimated_average':
                    strength = self.get_neighbour_strength_estimated_average(elem_outside_cluster, cluster_index)
                else:
                    raise ValueError('Unsupported linkage. Supported linkages are "complete", "centroid", "average" and "estimated_average".')
                
                # if elem_in_cluster was already a cluster and it got bigger keeping the same cluster_index
                if (new_key == key): 
                    self.neighbours_strengths[key] = strength
                else:
                    # Check if the same cluster has been considered previously (a common neighbour)
                    if inverse_new_key not in self.neighbours_strengths and new_key not in self.neighbours_strengths:
                        self.neighbours_strengths[new_key] = strength
                    del self.neighbours_strengths[key]




