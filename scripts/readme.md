# Scripts Directory

This directory contains the Python scripts used to process data and generate results. Each script is documented below with its usage instructions.

### `clustering.py`
**Description**: computes clusterings on the dataset.

**Arguments**:
- `--n_splits` (int): Number of splits for cross-validation. Default is 1.
- `--drop_years` (int): Number of years to drop during training. Default is 0.
- `--test_size` (int): Size of the test set in years. Default is 0.
- `--clustering` (str): Clustering method to use ('hierarchical', 'NonLinCTFA', 'NonLinCTFA_estimate').
- `--linkage` (str): Linkage criterion to use for hierarchical clustering ('complete', 'average').
- `--initial_threshold` (float, optional): Initial threshold value for hierarchical clustering.
- `--threshold_list` (float, nargs='+'): List of thresholds for hierarchical clustering in descending order.
- `--epsilon_list` (float, nargs='+'): List of epsilon values for NonLinCTFA clustering.
- `--use_area_weight` (bool): Whether to use area weights. Default is False.
- `--use_only_indexes` (bool): Whether to use only indexes. Default is False.
- `--selected_features` (int, nargs='+', optional): List of selected feature indexes.
- `--time` (str, choices=['month', 'year', 'month_year']): Time dimension for features.
- `--no_pkl` (bool): Whether to skip saving the clustering object as a pickle file. Default is False.

**Outputs**:
A folder containing the derived clusterings for each fold will be created under ../results/training_{years}.
- `clustering.pkl`: (if --no_pkl is not set) Pickle file containing the clustering object.
- `clustering.json`: JSON file containing the clusters.

### `multiple_models_CV.py`
**Description**: computes multiple local models



- `multiple_models_nested_wrapper_CV.py`: Brief description of what this script does.
- `multiple_models_CMI_CV.py`: Brief description of what this script does.
- `single_model_MLPR_CV.py`: Brief description of what this script does.
- `single_model_nested_wrapper_CV.py`: Brief description of what this script does.
- `single_model_CMI_CV.py`: Brief description of what this script does.
Provide example commands to run the scripts:

```bash
python example_script.py --input data/raw/dataset1.csv --output results/output1.csv
