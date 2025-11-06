Project Report: Lap Time Prediction Pipeline

Author: Artificial Human

Date: 02 nov 2025

1. Executive Summary

This document details a machine learning pipeline developed to predict motorsport lap times (Lap_Time_Seconds). The solution employs an ensemble of tree-based regression models, specifically RandomForestRegressor and ExtraTreesRegressor, to achieve high predictive accuracy.

The core of the project is a robust, reproducible preprocessing and modeling pipeline built with scikit-learn. This pipeline systematically handles feature imputation, scaling, and encoding, including a custom transformer for high-cardinality categorical features.

Models are rigorously evaluated using a 5-fold cross-validation strategy, generating out-of-fold (OOF) predictions for reliable performance assessment. The final output includes serialized model files (.joblib) for inference and multiple CSV files with test set predictions. Analysis of the results indicates a superior performance from the ExtraTreesRegressor model.

2. System Requirements

The pipeline is developed in Python 3.x. The following libraries are essential for execution and can be installed via pip:

# Core data handling and numerical operations

pip install pandas numpy

# Machine learning, preprocessing, and metrics

pip install scikit-learn

# Model persistence

pip install joblib

# For running the notebook environment

pip install jupyterlab notebook

3. Project Structure & Data

The script must be executed in a directory with the following structure:

.
├── train.csv # Primary training dataset
├── test.csv # Test dataset for inference
├── best_submission.ipynb # The main python notebook
└── ...

Data Requirements

train.csv: This file is used for training and validation. It must contain:

An id column (Unique Identifier).

The target variable: Lap_Time_Seconds.

All feature columns used in the model.

test.csv: This file is used for generating final predictions. It must contain:

An id column.

All feature columns present in the training set (excluding the target).

4. Execution Instructions

This pipeline is contained within a Jupyter Notebook (best_submission.ipynb). To run the complete training and prediction pipeline, follow these steps:

Install Dependencies:
Ensure you have Python and pip installed. Open a terminal and install all required libraries:

pip install pandas numpy scikit-learn joblib jupyterlab notebook

Launch Jupyter Environment:
After installation, open a terminal, navigate to the project's root directory, and start the Jupyter server. We recommend using JupyterLab:

jupyter lab

Alternatively, you can use the classic Jupyter Notebook interface:

jupyter notebook

This command will launch a new tab in your default web browser, showing the Jupyter file explorer.

Prepare Directory:
Using the Jupyter file explorer in your browser, verify that train.csv and test.csv are present in the same directory as best_submission.ipynb.

Execute the Notebook:

In the Jupyter file explorer, click on best_submission.ipynb to open it in a new tab.

To execute the entire pipeline from start to finish, navigate to the notebook's menu bar and select:
Kernel > Restart & Run All Cells...

A dialog box will ask for confirmation. Click Restart and Run All Cells.

The notebook will execute each cell in order. All print outputs, including the introductory feature lists and the fold-by-fold RMSE results, will be displayed inline, directly beneath their corresponding code cells. The final artifacts (CSV files and .joblib models) will be saved to the project directory, as detailed in Section 8.

5. Detailed Pipeline Architecture

This section details the technical implementation of the pipeline, from data preparation to model training.

5.1. Data Ingestion & Preparation

Loading: train.csv and test.csv are loaded into pandas DataFrames.

Feature Definition: Columns are programmatically categorized based on predefined lists:

numeric_features: Continuous or ordinal data.

cat_onehot: Low-cardinality categorical data suitable for one-hot encoding.

cat_freq: High-cardinality categorical data where one-hot encoding would be inefficient.

Special Handling: The Penalty feature is pre-processed to fill missing (NaN) values with the string 'NoPenalty', ensuring it is treated as a distinct category.

5.2. Preprocessing Strategy

A ColumnTransformer is used to apply different transformations to different subsets of features. This ensures that the same preprocessing steps are consistently applied during both training and inference.

The preprocessor is constructed by the build_preprocessor function:

Numeric Features (numeric_transformer):

Imputation: SimpleImputer(strategy="median") is used to fill missing values. The median is chosen as it is robust to outliers, which are common in real-world data.

Scaling: StandardScaler() is applied to scale features by removing the mean and scaling to unit variance. This is essential for many models, though less critical (but still good practice) for tree-based ensembles.

Low-Cardinality Categorical (onehot_transformer):

Imputation: SimpleImputer(strategy="constant", fill_value="Missing") creates an explicit "Missing" category.

Encoding: OneHotEncoder(handle_unknown="ignore") converts these categories into binary columns. handle_unknown="ignore" prevents errors if unseen categories appear in the test data.

High-Cardinality Categorical (freq_transformer):

Imputation: SimpleImputer(strategy="constant", fill_value="Missing").

Encoding: A custom ValueCountMapper (detailed below) is used. This strategy replaces each category with its frequency (or percentage) in the training data. This is a powerful technique for handling features like circuit_name where one-hot encoding would create hundreds of sparse columns.

5.3. Custom Transformer: ValueCountMapper

A plain Python class, ValueCountMapper, was implemented to perform frequency encoding. It adheres to the scikit-learn transformer API (i.e., it has .fit(), .transform(), and .get_feature_names_out() methods).

fit(X, y=None):

Iterates through the provided columns.

For each column, it calculates the value_counts() (normalized to frequencies).

These frequency maps are stored in a dictionary, self.count*maps*.

transform(X):

For each column, it retrieves the learned frequency map.

It applies the map to the new data using pd.Series.map().

Any values in X that were not seen during fit (and thus are not in the map) are filled with 0.0.

5.4. Modeling Strategy

Two separate models are trained and evaluated to compare performance and provide an ensemble of results.

Model 1: RandomForestRegressor (rf)

Parameters: n_estimators=200, random_state=42, n_jobs=-1

Rationale: A robust, well-understood ensemble model that is highly resistant to overfitting and provides strong baseline performance.

Model 2: ExtraTreesRegressor (et)

Parameters: n_estimators=300, random_state=42, n_jobs=-1, bootstrap=False

Rationale: An "Extremely Randomized Trees" model. It is similar to Random Forest but introduces more randomness in the selection of split-points, which can reduce variance and computation time.

6. Validation and Evaluation

A KFold cross-validation strategy with n_splits=5 is the primary method for model evaluation.

6.1. Cross-Validation (CV) Loop

For each of the 5 folds:

Split: The training data is divided into a train_idx and val_idx.

Fit Preprocessor: A new build_preprocessor is fit only on the train_idx data. This is crucial to prevent data leakage from the validation set.

Transform: The preprocessor transforms the train_idx, val_idx, and the entire test_df data.

Train Model: The model (e.g., rf) is trained only on the transformed train_idx data.

Predict:

Predictions are made on the transformed val_idx data. These are stored in an array (oof_preds) to build a complete set of out-of-fold predictions.

Predictions are made on the transformed test_df. These are added to test_preds_cv and will be averaged at the end (by dividing by n_splits).

Score: The Root Mean Squared Error (RMSE) is calculated for the fold and printed.

6.2. Final Model Training

After the CV loop, two final steps are taken:

Full Train: A new preprocessor and a new model are fit and trained on the entirety of X_full and y_full.

Persistence: This final preprocessor and model are serialized and saved to disk using joblib (e.g., rf_final_model.joblib). This pair represents the final, deployable artifact.

7. Example Output & Results Analysis

The following is an example console output from a complete run of the pipeline, demonstrating the performance of both models.

7.1. Console Log

Loading train/test CSVs...
Numeric features (25): ['Unique ID', 'Rider_ID', 'Len_Circuit_inkm', 'Laps', 'Start_Position', 'Formula_Avg_Speed_kmh', 'Humidity_%', 'Champ_Points', 'Champ_Position', 'race_year', 'seq', 'position', 'points', 'Corners_in_Lap', 'Tire_Degradation_Factor_per_Lap', 'Pit_Stop_Duration_Seconds', 'Ambient_Temperature_Celsius', 'Track_Temperature_Celsius', 'air', 'ground', 'starts', 'finishes', 'with_points', 'podiums', 'wins']
One-hot categorical (7): ['Formula_category_x', 'Formula_Track_Condition', 'Tire_Compound', 'Penalty', 'Session', 'weather', 'track']
Freq categorical (2): ['Formula_shortname', 'circuit_name']

========================================
Running model: rf
========================================

-- Fold 1/5 --
Fold 1 RMSE: 0.1852

-- Fold 2/5 --
Fold 2 RMSE: 0.1580

-- Fold 3/5 --
Fold 3 RMSE: 0.1770

-- Fold 4/5 --
Fold 4 RMSE: 0.2090

-- Fold 5/5 --
Fold 5 RMSE: 0.1727

RF CV mean RMSE: 0.1804 ± 0.0168
RF OOF RMSE: 0.1811

Training final rf model on full data...
Saved rf_final_model.joblib and rf_preprocessor.joblib
Finished model: rf

========================================
Running model: et
========================================

-- Fold 1/5 --
Fold 1 RMSE: 0.0811

-- Fold 2/5 --
Fold 2 RMSE: 0.0461

-- Fold 3/5 --
Fold 3 RMSE: 0.0735

-- Fold 4/5 --
Fold 4 RMSE: 0.0995

-- Fold 5/5 --
Fold 5 RMSE: 0.0674

ET CV mean RMSE: 0.0735 ± 0.0174
ET OOF RMSE: 0.0755

Training final et model on full data...
Saved et_final_model.joblib and et_preprocessor.joblib
Finished model: et

All done.

7.2. Results Analysis

Based on the execution log, we can draw clear conclusions:

Random Forest (RF): Achieved a final Out-of-Fold (OOF) RMSE of 0.1811. The cross-validation mean RMSE was 0.1804, with a standard deviation of 0.0168, indicating consistent performance across folds.

Extra Trees (ET): Achieved a final OOF RMSE of 0.0755. The cross-validation mean RMSE was 0.0735, with a standard deviation of 0.0174.

Conclusion: The ExtraTreesRegressor (OOF RMSE: 0.0755) significantly outperforms the RandomForestRegressor (OOF RMSE: 0.1811) on this dataset. The more randomized nature of the Extra Trees model appears to be highly effective for this problem. The OOF RMSE is the most reliable metric for performance, as it is generated on data the model has not been trained on.

8. Generated Artifacts (Outputs)

Upon successful execution, the script will create the following 8 files in the root directory:

Submission Files (CSV)

output_cv_avg_rf.csv: Test predictions based on the averaged 5-fold CV predictions from the Random Forest model.

output_cv_avg_et.csv: Test predictions based on the averaged 5-fold CV predictions from the Extra Trees model.

output_rf.csv: Test predictions from the final Random Forest model (trained on 100% of the data).

outputET.csv: Test predictions from the final Extra Trees model (trained on 100% of the data).

Model Files (Joblib)

rf_preprocessor.joblib: The fitted ColumnTransformer corresponding to the final RF model.

rf_final_model.joblib: The final RandomForestRegressor object trained on all data.

et_preprocessor.joblib: The fitted ColumnTransformer corresponding to the final ET model.

et_final_model.joblib: The final ExtraTreesRegressor object trained on all data.

(To load and use a final model for inference, one would load both the preprocessor and the model)
