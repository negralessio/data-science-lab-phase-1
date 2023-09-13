"""
This python file is used to perform bayesian hyperparameter optimization on the processed data.
It is not intended to call this file on the pipeline. Instead, execute this file using the follow commands
while in the root directory, i.e. phase-1:
- python3 src/bayesian_hyeropt.py
This will perform the Bayesian Hyperparameter search on 100% of the train data with all default params.
You can also use:
- python3 src/bayesian_hyeropt.py --use_test_split --subsample 100000 --n_iter 100
To specify whether to split into train test, use subsample or the number of iterations of the BayesianSearch
"""

import pandas as pd
import argparse
import time
import os
import xgboost
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef, accuracy_score

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


def load_data_from_processed(path_xtrain, path_ytrain, subsample = None):
    """
    Function to load the already preprocessed data from the data/processed folder

    :param path_xtrain: Path to X_train
    :param path_ytrain: Path to y_train
    :param subsample: Integer optional. Number of subsamples to take
    :return: X_train and y_train -- pd.DataFrames
    """
    print("Loading Data ...")
    X_train = pd.read_csv(path_xtrain, index_col="building_id")
    y_train = pd.read_csv(path_ytrain, index_col="building_id",
                          usecols=["building_id", "damage_grade"])

    # If subsample size is provided, take first :subsamples of the df, e.g. first 100k rows
    if subsample is not None:
        print(f"Taking subsample: First {subsample} rows")
        X_train = X_train.iloc[:subsample]
        y_train = y_train.iloc[:subsample]

    return X_train, y_train


def split_into_train_test(X_train, y_train, test_size=0.2):
    """
    Function to split the train data into train and test set

    :param X_train: Train Data
    :param y_train: Labels of the Train Data
    :param test_size: Float -- Size of the test size, e.g. 0.2
    :return: X_train, X_test, y_train, y_test
    """
    print(f"Data split into train test set using test_size of {test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                        random_state=42,
                                                        stratify=y_train,
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test


def get_prediction_score(model, X_test, y_test, rename=True):
    """
    Function to get the prediction on a hold out set of a model

    :param model: Fitted model, e.g. XGBClassifier
    :param X_test: Hold Out Set to evaluate
    :param y_test: Labels of the Hold Out Set to evaluate
    :param rename: Bool whether to allow renaming of the prediction
    :return: Accuracy acc and mcc
    """

    # Predict on unseen data
    y_pred = pd.DataFrame(model.predict(X_test))

    # Since XGBoost predicts [0, 1, 2] classes, we have to transform it to original class labels
    if isinstance(model, xgboost.XGBClassifier) and rename:
        print("XGBoost model was used. Renaming predictions ...")
        y_pred = y_pred.replace({0: 1, 1: 2, 2: 3})

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    mcc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
    print(f"ACC on test set: {acc:.4f}")
    print(f"MCC on test set: {mcc:.4f}")

    return acc, mcc


def encode_labels(y_train):
    """
    Encodes the labels using LabelEncoder. Only relevant if XGBClassifier is used.
    :param y_train: Labels of the train data {1, 2, 3}
    :return: Encoded Labels of the train data {0, 1, 2}
    """
    print("Encoding labels ...")

    # XGBoost expects [0, 1, 2] class labels instead of [1, 2, 3]
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train["damage_grade"].to_numpy())

    return encoder, y_train


def perform_bayessearch(X_train, y_train, n_iter=100):
    """
    Performs BayesSearchCV with given parameter grid for XGBClassifier.

    :param X_train: Train Data
    :param y_train: Labels of Train Data
    :param n_iter: Number of iterations of BayesSearch
    :return: Fitted bayessearch object and CV_results as pd.DataFrame
    """

    # Define base model to optimize
    model_xgb = xgboost.XGBClassifier(random_state=42, n_jobs=-1)

    # Define param grid for XGB model
    param_grid_xgb = {'learning_rate': Real(0.01, 0.7, 'uniform'),
                      'max_depth': Integer(3, 25),
                      'subsample': Real(0.1, 1.0, 'uniform'),
                      'colsample_bytree': Real(0.1, 1.0, 'uniform'),  # subsample ratio of columns by tree
                      'reg_lambda': Real(1e-9, 20., 'uniform'),  # L2 regularization
                      'reg_alpha': Real(1e-9, 20., 'uniform'),  # L1 regularization
                      'n_estimators': Integer(50, 300)
                      }

    start_time = time.time()
    print("Performing BayesSearchCV on XGBClassifier ...")
    bayes_opt = BayesSearchCV(estimator=model_xgb, n_iter=n_iter,
                              search_spaces=param_grid_xgb,
                              n_jobs=-1, cv=5, random_state=42,
                              scoring="matthews_corrcoef",
                              return_train_score=True, verbose=1, refit="matthews_corrcoef")
    bayes_opt.fit(X_train, y_train)

    # Print information
    print(f"BayesSearchCV took {(time.time() - start_time):.2f} seconds")
    print(80 * "=")
    print(f"Best Score: {(bayes_opt.best_score_):.4f} MCC")
    print(f"Best Params: {dict(bayes_opt.best_params_)}")
    print(80 * "=")

    # Transform CV Result to pandas for returning
    bayes_opt_cv_results = pd.DataFrame(bayes_opt.cv_results_)
    # Print information of best candidate
    bayes_opt_cv_results = bayes_opt_cv_results.sort_values(by="rank_test_score", ascending=True)
    # Get mean and std of test scores of each fold and print it
    mean = bayes_opt_cv_results['mean_test_score'].values[0]
    std = bayes_opt_cv_results['std_test_score'].values[0]
    print(f"Mean Test Score: {mean:.4f} +/- {std:.4f}")
    print(80 * "=")

    return bayes_opt, bayes_opt_cv_results


def save_cv_results(cv_results: pd.DataFrame, path: str):
    """
    Function to save csv file to specified folder
    :param cv_results: DataFrame to save
    :param path: Path to save cv results to
    :return: None
    """
    # Check if results folder exists, if not create one
    if not os.path.exists(path):
        print(f"{path} dir does not exist yet. Create one.")
        os.makedirs(path)

    # Save df as csv with unique name
    path = f"{path}bayesopt_cv_results_of_{str(datetime.datetime.now()).replace(':', '_')}.csv"
    cv_results.to_csv(path_or_buf=path)
    print(f"Saved CV Results as .csv as: '{path}'")


def main(split_data: bool = False, subsample = None, n_iter=100):
    """
    Main function to execute all necessary steps for the BayesSearch i.e.
    - Data Loading (supports subsampling)
    - Optional: Split into Train-Test Sets
    - Encoding
    - BayesSearchCV
    - Saving of the Results
    - Optional: Scoring on the Test Set

    :param split_data: bool, whether to create hold-out-set (0.2 frac)
    :param subsample: int, number of subsamples to take
    :param n_iter: int, number of iterations of the BayesSearchCV
    :return: None
    """
    # Load Data from processed folder
    X_train, y_train = load_data_from_processed(path_xtrain="data/processed/train_data_cleaned.csv",
                                                path_ytrain="data/processed/train_labels.csv",
                                                subsample=subsample)
    # Split data if provided arg is True
    if split_data:
        encoder, y_train = encode_labels(y_train=y_train)
        X_train, X_test, y_train, y_test = split_into_train_test(X_train, y_train)
    else:
        # Encode labels
        encoder, y_train = encode_labels(y_train=y_train)

    # Perform BayesSearchCV
    bayes_opt, bayes_opt_cv_results = perform_bayessearch(X_train, y_train, n_iter=n_iter)

    # Save Dataframe
    save_cv_results(cv_results=bayes_opt_cv_results,
                    path="data/hyperparamopt_results/")

    # If Hold Out set was created, then test model on that set
    if split_data:
        print("Testing fitted model on Hold-Out-Set ...")
        get_prediction_score(model=bayes_opt, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    # Should be executed from dir phase-1 with the command
    # python3 src/bayesian_hyeropt.py     (default !!!)
    # python3 src/bayesian_hyeropt.py --use_test_split
    # python3 src/bayesian_hyeropt.py --use_test_split --subsample 100000
    # python3 src/bayesian_hyeropt.py --use_test_split --subsample 100000 --n_iter 100

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_test_split",
                        default=False,
                        action="store_true",
                        help="Bool -- Whether to use test split or not")
    parser.add_argument("--subsample", default=None, type=int, help="Number of samples to use")
    parser.add_argument("--n_iter", default=100, type=int, help="Number of iterations of BayesSearchCV")
    args = parser.parse_args()

    # Parse whether to use split_data
    split_data = bool(args.use_test_split)
    subsample = args.subsample
    n_iter = args.n_iter

    # Call main with parsed args
    main(split_data=split_data, subsample=subsample, n_iter=n_iter)

