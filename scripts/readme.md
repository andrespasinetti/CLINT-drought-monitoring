# Scripts Directory

This directory contains the Python scripts used to process data and generate results. Each script is documented below with its usage instructions.

### `clustering.py`
**Description**: computes clusterings on the dataset. If n_splits is greater than 1, it computes one clustering for each cross-validation split on the training folds.

**Arguments**:
- `--n_splits` (`int`): Number of splits for cross-validation. Default is 1.
- `--drop_years` (`int`): Number of years to randomly drop in the training set of each cross-validation split. Default is 0.
- `--test_size` (`int`): Size of the test set outside cross-validation in years. Default is 0.
- `--clustering` (`str`): Clustering method to use ('hierarchical', 'NonLinCTFA', 'NonLinCTFA_estimate').
- `--linkage` (`str`): Linkage criterion to use for hierarchical clustering ('centroid', 'average').
- `--initial_threshold` (`float`, optional): Initial threshold value for hierarchical clustering.
- `--threshold_list` (`float`, `nargs='+'`): List of thresholds for hierarchical clustering in descending order.
- `--epsilon_list` (`float`, `nargs='+'`): List of epsilon values for NonLinCTFA clustering.
- `--use_area_weight` (`bool`): Whether to use area weights. Default is False.
- `--use_only_indexes` (`bool`): Whether to use only indexes. Default is False.
- `--selected_features` (`int`, `nargs='+'`, optional): List of selected feature indexes.
- `--time` (`str`, `choices=['month', 'year', 'month_year']`): Time dimension for features.
- `--no_pkl` (`bool`): Whether to skip saving the clustering object as a pickle file. Default is False.

**Outputs**:
A folder containing the derived clusterings for each fold will be created under ../results/training_{n_splits}.
- `clustering.pkl`: (if --no_pkl is not set) Pickle file containing the hierarchical clustering object.
- `clustering.json`: JSON file containing the clusters.


### `multiple_models_CV.py`
**Description**: computes multiple local models.

**Arguments**:
- `--n_splits` (`int`): Number of splits for cross-validation. Default is 1.
- `--test_size` (`int`): Size of the test set in months. Default is 0.
- `--drop_years` (`int`): Number of years to drop from training data. Default is 0.
- `--clustering` (`str`): Clustering method to use ('none', 'hierarchical', 'NonLinCTFA', 'NonLinCTFA_estimate). Default is 'none'.
- `--linkage` (`str`): Linkage method ('centroid', 'average'). Required if clustering is not 'none'.
- `--threshold` (`float`): Threshold for clustering. Required if clustering is not 'none'.
- `--feature_lag` (`int`, choices=[0, 1, 2]): Lag of features. Default is 0.
- `--time` (`str`, choices=['month', 'year', 'month_year']): Time granularity. Optional.
- `--detrend` (`bool`): Whether to detrend the data. Default is False.
- `--use_only_indexes` (`bool`): Use only indexed features. Default is False.
- `--selected_features` (`int` list): List of selected feature indices. Optional.
- `--feature_selection` (`str`): Method for feature selection. Optional.
- `--max_num_features` (`int`): Maximum number of features to use. Optional.
- `--model_type` (`str`): Type of model to use ('RL', 'SVR'). Required.
- `--num_processes` (`int`): Number of processes to use. Default is the maximum available processors.

**Outputs**:
The predicted target csv is stored in ../results/training_{n_splits}/predictions.


### `multiple_models_nested_wrapper_CV.py`
**Description**: Rank features for each cluster and fold with nested wrapper feature selection.

**Arguments**:
- `--n_splits` (`int`): Number of splits for cross-validation. Default is 1.
- `--test_size` (`int`): Size of the test set in months. Default is 0.
- `--clustering` (`str`): Clustering method to use ('none', 'hierarchical', 'NonLinCTFA', 'NonLinCTFA_estimate). Default is 'none'.
- `--linkage` (`str`): Linkage criterion for clustering. Options include 'centroid', 'average'. Required if clustering is not 'none'.
- `--threshold` (`float`): Threshold for clustering. Required if clustering is not 'none'.
- `--feature_lag` (`int`, choices=[0, 1, 2]): Lag of features. Default is 0.
- `--time` (`str`, choices=['month', 'year', 'month_year']): Time granularity for feature aggregation. Optional.
- `--use_only_indexes` (`bool`): Whether to use only indexed features. Default is False.
- `--max_num_features` (`int`): Maximum number of features to select. Optional.
- `--model_type` (`str`): Type of model to use. Required. Valid values depend on the `get_model` function implementation.
- `--num_processes` (`int`): Number of processes to use for parallel computation. Default is the maximum available processors.

**Output**:
The script generates a JSON file containing the ranked features for each fold and cluster. The file is saved in ../results/training_{n_splits}.

- **File Path:** `../results/training_{17-test_size}/selected_features/feature_selection_nested_wrapper_multiple_models_{clustering}_{linkage}_{threshold}{lag_text}{indexes_text}{time_text}.json`
- **Content:** A dictionary where:
  - **Key:** Fold index
  - **Value:** Dictionary of clusters, where:
    - **Key:** Cluster index
    - **Value:** List of selected feature IDs

- `multiple_models_CMI_CV.py`: Brief description of what this script does.
- `single_model_MLPR_CV.py`: Brief description of what this script does.
- `single_model_nested_wrapper_CV.py`: Brief description of what this script does.
- `single_model_CMI_CV.py`: Brief description of what this script does.
Provide example commands to run the scripts:

```bash
python example_script.py --input data/raw/dataset1.csv --output results/output1.csv
