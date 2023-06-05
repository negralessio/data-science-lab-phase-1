""" Main file for the project. This file is used to run the whole pipeline. """
import argparse
import time

import numpy as np
import pandas as pd
import mlflow

import modelling
from data_cleaning import drop_correlated_features
from data_cleaning import group_categorical_features
from data_cleaning import prepare_data
from feature_engineering import dimensionality_reduction
from feature_engineering import (encode_train_data, encode_test_data, normalize_train_data, normalize_test_data,
                                 get_quality_of_superstructure, get_risk_status_based_on_geo_level,
                                 get_geocoded_districts)
from feature_selection import (get_top_k_features_using_rfe_cv, get_top_k_features_using_rfe,
                               get_top_k_features_using_mi)
from utils import load_config, check_file_exists


def parse_args():
    """
    Function that parses the config.yaml
    :return: Parsed config file
    """
    # Parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file")
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)
    return cfg


def load_data(cfg):
    """
    Function to load the data using the paths provided in the config file cfg
    :param cfg: Parsed config file
    :return: Train Data, Label of Train Data, Test Values, Result Path to save results
    """

    # Check if data in config exists and read it
    if cfg["modelling"]["verbosity"] >= 1:
        print("Searching for data paths in config ...")
    train_values_path = cfg["paths"]["train_values"]
    train_labels_path = cfg["paths"]["train_labels"]
    test_values_path = cfg["paths"]["test_values"]
    result_path = cfg["paths"]["result"]

    check_file_exists(train_values_path)
    check_file_exists(train_labels_path)
    check_file_exists(test_values_path)

    # Load data
    if cfg["modelling"]["verbosity"] >= 1:
        print("Loading Data ...")
    train_values = pd.read_csv(train_values_path).set_index("building_id")
    train_labels = pd.read_csv(train_labels_path).set_index("building_id")
    test_values = pd.read_csv(test_values_path).set_index("building_id")

    """ Make Sample Size smaller for experimenting and testing; Keep commented! """
    train_values = train_values.iloc[:1000]
    train_labels = train_labels.iloc[:1000]
    test_values = test_values.iloc[:1000]

    return train_values, train_labels, test_values, result_path


def clean_data(cfg, train_values, test_values):
    """
    Function to clean the data, i.e. grouping, dropping correlated features, etc...
    :param cfg: Parsed config file
    :param train_values: Train data X
    :param test_values: Test data
    :return: Cleaned train data and cleaned test data
    """

    # Prepare raw data
    binary_encoded_cols = [x for x in train_values.columns if x.startswith("has_")]
    columns_to_ignore = cfg.get("data_cleaning", "NO DATA CLEANING DEFINED!").get("columns_to_ignore", [])

    if cfg["modelling"]["verbosity"] >= 1:
        print("Cleaning Train Data ...")
    train_data_cleaned = prepare_data(df=train_values, config=cfg, ignore_cols=columns_to_ignore+binary_encoded_cols,
                                      outlier_method="replace", verbose=cfg["modelling"]["verbosity"])

    if cfg["modelling"]["verbosity"] >= 1:
        print("Cleaning Test Data ...")
    test_data_cleaned = prepare_data(df=test_values, config=cfg, ignore_cols=columns_to_ignore+binary_encoded_cols,
                                     outlier_method="replace", verbose=cfg["modelling"]["verbosity"])

    if cfg["modelling"]["verbosity"] >= 1:
        print("Drop correlated features...")
    train_data_cleaned = drop_correlated_features(data=train_data_cleaned, config=cfg["data_cleaning"]["correlations"])
    test_data_cleaned = drop_correlated_features(data=test_data_cleaned, config=cfg["data_cleaning"]["correlations"])

    return train_data_cleaned, test_data_cleaned


def feature_engineer_data(cfg, train_data_cleaned, test_data_cleaned, train_values, test_values, train_labels):
    """
    Function to engineer new features from the provided data
    :param cfg: Parsed config file
    :param train_data_cleaned: Train data cleaned
    :param test_data_cleaned: Test data cleaned
    :param train_values: Train data
    :param test_values: Test data
    :param train_labels: Labels of train data
    :return: Train and Test Data with engineered features
    """

    # Group categorical features with rarely occurring realizations
    if not cfg["feature_engineering"]["group_categorical"].get("skip", False):
        if cfg["modelling"]["verbosity"] >= 1:
            print("Grouping categorical features ...")
        train_data_cleaned = group_categorical_features(df=train_data_cleaned, default_val="others",
                                                        verbose=cfg["modelling"]["verbosity"])
        test_data_cleaned = group_categorical_features(df=test_data_cleaned, default_val="others",
                                                       verbose=cfg["modelling"]["verbosity"])

    # Add new features for risk status
    if not cfg["feature_engineering"]["risk_status"].get("skip", False):
        if cfg["modelling"]["verbosity"] >= 1:
            print("Add risk status features...")
        test_data_cleaned = get_risk_status_based_on_geo_level(data=train_values,
                                                               df_to_add_info=test_data_cleaned,
                                                               labels=train_labels,
                                                               geo_level=cfg["feature_engineering"]["risk_status"]["geo_level"])
        train_data_cleaned = get_risk_status_based_on_geo_level(data=train_values,
                                                                df_to_add_info=train_data_cleaned,
                                                                labels=train_labels,
                                                                geo_level=cfg["feature_engineering"]["risk_status"]["geo_level"])

    # Add geocoded districts features
    if not cfg["feature_engineering"]["geocode_districts"].get("skip", False):
        if cfg["modelling"]["verbosity"] >= 1:
            print("Add geocoded districts features (lat, long, district name, min distance to epicenter, "
                  "max distance to epicenter) ...")
        train_data_cleaned = get_geocoded_districts(df=train_values,
                                                    geo_path=cfg["feature_engineering"]["geocode_districts"]["path"],
                                                    drop_key=cfg["feature_engineering"]["geocode_districts"]["drop_key"])
        test_data_cleaned = get_geocoded_districts(df=test_values,
                                                   geo_path=cfg["feature_engineering"]["geocode_districts"]["path"],
                                                   drop_key=cfg["feature_engineering"]["geocode_districts"]["drop_key"])

    # Add superstructure quality
    if not cfg["feature_engineering"]["superstructure_quality"].get("skip", False):
        if cfg["modelling"]["verbosity"] >= 1:
            print("Add superstructure quality feature...")
        train_data_cleaned = get_quality_of_superstructure(raw_data=train_values, df_to_add_info=train_data_cleaned)
        test_data_cleaned = get_quality_of_superstructure(raw_data=test_values, df_to_add_info=test_data_cleaned)

    # Apply One Hot Encoding on categorical features
    if not cfg["feature_engineering"]["categorical_encoding"].get("skip", False):
        if cfg["feature_engineering"]["categorical_encoding"]["method"] == "One-Hot":
            if cfg["modelling"]["verbosity"] >= 1:
                print("One Hot Encoding features ...")
            train_data_cleaned, ohe = encode_train_data(x_train=train_data_cleaned)
            test_data_cleaned = encode_test_data(x_test=test_data_cleaned, ohe=ohe)

    # Apply StandardScaler (method="standard") or MinMax Scaler (method="minmax") on Features
    if not cfg["feature_engineering"]["normalize"].get("skip", False):
        if cfg["modelling"]["verbosity"] >= 1:
            print("Normalizing Data ...")
        train_data_cleaned, scaler = normalize_train_data(x_train=train_data_cleaned,
                                                          method=cfg["feature_engineering"]["normalize"]["method"])
        test_data_cleaned = normalize_test_data(x_test=test_data_cleaned, scaler=scaler)

    return train_data_cleaned, test_data_cleaned


def feature_selection(cfg, train_data_cleaned, train_labels, test_data_cleaned):
    """
    Function to perform feature selection (default: RFECV)
    :param cfg: Parsed config file
    :param train_data_cleaned: Train data cleaned
    :param train_labels: Labels of train data
    :param test_data_cleaned: Test data cleaned
    :return: Feature selected train and test data as well as list of the selected features
    """

    # Store list of best features here
    best_feats = []

    # Feature Selection: Get top k features using RFE, RFECV, or use MI
    feature_selection_config = cfg["feature_engineering"]["feature_selection"]
    method = feature_selection_config.get("method")

    if not feature_selection_config.get("skip", False) and method:
        if cfg["modelling"]["verbosity"] >= 1:
            print(f"Selecting best features using {method}...")

        if method == "RFECV":
            best_feats, rfecv = get_top_k_features_using_rfe_cv(x_train=train_data_cleaned,
                                                                y_train=train_labels,
                                                                min_features_to_select=5,
                                                                k_folds=5,
                                                                scoring="matthews_corrcoef",
                                                                step=feature_selection_config.get("step", 1),
                                                                verbose=0)

        elif method == "RFE":
            best_feats, rfe = get_top_k_features_using_rfe(x_train=train_data_cleaned,
                                                           y_train=train_labels,
                                                           k=feature_selection_config.get("k", 10),
                                                           step=feature_selection_config.get("step", 1),
                                                           verbose=0)

        elif method == "MI":
            best_feats, mi_scores = get_top_k_features_using_mi(x_train=train_data_cleaned,
                                                                y_train=train_labels,
                                                                k=feature_selection_config.get("k", 10))

        if cfg["modelling"]["verbosity"] >= 1:
            print(f"\nSelected feature set: {best_feats}\n")

        # Keep best columns
        train_data_cleaned = train_data_cleaned[best_feats]
        test_data_cleaned = test_data_cleaned[best_feats]

    return train_data_cleaned, test_data_cleaned, best_feats


def reduce_dimensionality(cfg, best_feats, train_data_cleaned, test_data_cleaned):
    """
    Function to perform PCA dimensionality reduction.
    :param cfg: Parsed config file
    :param best_feats: List of best features yielded from feature_selection() function
    :param train_data_cleaned: Train data cleaned
    :param test_data_cleaned: Test data cleaned
    :return: Dim. reduced Train and Test data
    """

    # Dimensionality Reduction (only if number of features is larger than threshold)
    dimensionality_reduction_config = cfg["feature_engineering"]["dimensionality_reduction"]

    if not dimensionality_reduction_config.get("skip", False) and len(best_feats) > dimensionality_reduction_config.get(
            "feature_threshold", 0):
        if cfg["modelling"]["verbosity"] >= 1:
            print("Performing dimensionality reduction...")
        train_data_cleaned, test_data_cleaned = dimensionality_reduction(train_data=train_data_cleaned,
                                                                         test_data=test_data_cleaned,
                                                                         method=dimensionality_reduction_config.get(
                                                                             "method"),
                                                                         n_components=dimensionality_reduction_config.get(
                                                                             "n_components"))

    return train_data_cleaned, test_data_cleaned


def train_model(cfg, train_data_cleaned, train_labels):
    """
    Function to perform CV and to train the final estimator
    :param cfg: Parsed config file
    :param train_data_cleaned: Train data cleaned
    :param train_labels: Labels of the train data
    :return: Fitted model and CV results
    """

    # Model training
    if cfg["modelling"]["verbosity"] >= 1:
        print("Modelling ...")

    # Convert to float64 for computation purposes
    train_data_cleaned = train_data_cleaned.astype(np.float64)

    # Return fitted model
    model, cv_results = modelling.train_model(model="XGBoost",
                                              hyperparameter_grid=cfg["modelling"]["params_xgb"],
                                              train_data=train_data_cleaned,
                                              train_labels=train_labels,
                                              scoring=cfg["modelling"]["scoring"],
                                              verbose=cfg["modelling"]["verbosity"])
    return model, cv_results


def make_prediction(cfg, model, test_data_cleaned, result_path):
    """
    Function to make final prediction on the test set
    :param cfg: Parsed config file
    :param model: Fitted model yielded from train_model() function
    :param test_data_cleaned: Test data cleaned
    :param result_path: Path to save results
    :return: None
    """

    # Make prediction
    if cfg["modelling"]["verbosity"] >= 1:
        print("Make predictions ...")

    modelling.make_prediction(model=model,
                              test_data=test_data_cleaned,
                              result_path=result_path,
                              verbose=cfg["modelling"]["verbosity"])


def log_model_eval(model, cv_results):
    """
    Function that logs the model evaluation to the MLflow run registry
    :param model: Fitted model
    :param cv_results: Cross validation results
    :return: None
    """
    print("Logging model results to mlflow registry ...")

    #mlflow.log_params(model.get_xgb_params())
    mlflow.log_metric(key="CV Test MCC MEAN", value=round(cv_results['test_matthews_corrcoef'].mean(), 4))
    mlflow.log_metric(key="CV Test MCC STD", value=round(cv_results['test_matthews_corrcoef'].std(), 4))
    mlflow.log_metric(key="CV Test ACC MEAN", value=round(cv_results['test_accuracy'].mean(), 4))
    mlflow.log_metric(key="CV Test ACC STD", value=round(cv_results['test_accuracy'].std(), 4))


def main(cfg):
    """
    Main function that executes the pipeline, i.e. ...
    - Load data
    - Clean Data
    - Perform Feature Engineering
    - Perform Feature Selection (Default: RFECV)
    - Reduce Dimensionality using PCA (Default: skipped)
    - Train Model
    - Log model results using mlflow
    - Make Prediction
    :param cfg: Parsed config file
    :return: None
    """

    # Track time for execution
    start_time = time.time()
    print(80 * "=")
    print("Starting pipeline ...")

    # Load Data
    train_values, train_labels, test_values, result_path = load_data(cfg)
    # Clean Data
    train_data_cleaned, test_data_cleaned = clean_data(cfg, train_values, test_values)
    # Perform Feature Engineering
    train_data_cleaned, test_data_cleaned = feature_engineer_data(cfg, train_data_cleaned, test_data_cleaned,
                                                                  train_values, test_values, train_labels)
    # Perform Feature Selection
    train_data_cleaned, test_data_cleaned, best_feats = feature_selection(cfg, train_data_cleaned,
                                                                          train_labels, test_data_cleaned)
    # Reduce Dimensionality
    train_data_cleaned, test_data_cleaned = reduce_dimensionality(cfg, best_feats,
                                                                  train_data_cleaned, test_data_cleaned)

    # Log model results
    mlflow.xgboost.autolog(log_models=False)
    with mlflow.start_run(tags={"train_set_size": len(train_data_cleaned)}):
        # Perform CV and train final estimator
        model, cv_results = train_model(cfg, train_data_cleaned, train_labels)
        log_model_eval(model, cv_results)

    # Make Prediction on Test Set and save data
    make_prediction(cfg, model, test_data_cleaned, result_path)

    # Track time for execution and print
    print(f"Finished -- Pipeline took {(time.time() - start_time):.2f} seconds --")
    print(80 * "=")


if __name__== "__main__":
    # Parse config file
    cfg = parse_args()

    # Call Pipeline
    main(cfg)

