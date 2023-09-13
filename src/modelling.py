""" This module contains functions to train a model and make a prediction."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate

import xgboost


def train_model(model=None, hyperparameter_grid=None, train_data=None, train_labels=None, scoring=None, verbose=True):
    """
    Trains a model on the given data and returns the trained model and the cross validation results. If no model is
    given, a dummy classifier is used. If no hyperparameter grid is given, the default parameters are used.

    :param model: Model to train
    :type model: str
    :param hyperparameter_grid: Hyperparameter grid for the model
    :type hyperparameter_grid: dict
    :param train_data: Training data
    :type train_data: pandas DataFrame
    :param train_labels: Training labels
    :type train_labels: pandas DataFrame
    :param scoring: Scoring function to use for cross validation (see sklearn.metrics).
    :type scoring: str
    :param verbose: Verbosity level
    :type verbose: bool

    :return: Trained model and cross validation results
    :rtype: sklearn model, dict
    """

    # Set up default parameters for each model available
    default_parameters = {
        "Dummy": {"strategy": "most_frequent"},
        "RandomForest": {"random_state": 42},
        "DecisionTree": {"random_state": 42},
        "XGBoost": {"n_estimators": 100,
                    "max_depth": 20,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1}
    }

    # Initialize model with default parameters if no hyperparameter grid is given
    if hyperparameter_grid is None:
        model_collection = {
            "Dummy": DummyClassifier(**default_parameters["Dummy"]),
            "RandomForest": RandomForestClassifier(**default_parameters["RandomForest"]),
            "DecisionTree": DecisionTreeClassifier(**default_parameters["DecisionTree"]),
            "XGBoost": xgboost.XGBClassifier(**default_parameters["XGBoost"])
        }
    else:
        model_collection = {
            "XGBoost": xgboost.XGBClassifier(**hyperparameter_grid)
        }

    # Get model from model collection
    try:
        model = model_collection[model]
    except KeyError:
        model = model_collection["Dummy"]
        print(f"Model '{model}' not found in model collection! Using Default Dummy Classifier.")

    # Get train data in shape that .fit() expects
    if isinstance(model, xgboost.XGBClassifier):
        # XGBoost expects [0, 1, 2] instead of [1, 2, 3]
        encoder = LabelEncoder()
        train_labels = encoder.fit_transform(train_labels["damage_grade"].to_numpy())
    else:
        train_labels = train_labels["damage_grade"].to_numpy()

    # Set up cross validation
    cv_results = cross_validate(model, train_data, train_labels, cv=5, scoring=scoring, n_jobs=-1,
                                return_train_score=True)

    # Fit model
    if verbose:
        print(f"Fitting Model ...")
    model.fit(train_data, train_labels)

    # Print results
    if verbose:
        print("")
        print(f"CV Training ACC: {round(cv_results['train_accuracy'].mean(), 4)} "
              f"+/- {round(cv_results['train_accuracy'].std(), 4)} ")
        print(f"CV Test ACC: {round(cv_results['test_accuracy'].mean(), 4)} "
              f"+/- {round(cv_results['test_accuracy'].std(), 4)}")
        print("")
        print(f"CV Training MCC: {round(cv_results['train_matthews_corrcoef'].mean(), 4)} "
              f"+/- {round(cv_results['train_matthews_corrcoef'].std(), 4)} ")
        print(f"CV Test MCC: {round(cv_results['test_matthews_corrcoef'].mean(), 4)} "
              f"+/- {round(cv_results['test_matthews_corrcoef'].std(), 4)}")
        print("")
    
    return model, cv_results


def make_prediction(model=None, test_data=None, result_path=None, verbose=True):
    """
    Makes a prediction on the test data and saves it to the given path.

    :param model: Model to make prediction with (must have .predict() method)
    :type model: sklearn model
    :param test_data: Test data to make prediction on
    :type test_data: pandas DataFrame
    :param result_path: Path to save the prediction to
    :type result_path: str
    :param verbose: Verbosity level
    :type verbose: bool

    :return: None
    """

    # Make prediction
    predictions = model.predict(test_data)
    test_data["damage_grade"] = predictions

    # Since XGBoost predicts [0, 1, 2] we have to rename back to original names, i.e. [1, 2, 3]
    if isinstance(model, xgboost.XGBClassifier):
        if verbose:
            print("XGBoost model was used. Renaming predictions ...")
        test_data["damage_grade"] = test_data["damage_grade"].replace({0: 1,
                                                                       1: 2,
                                                                       2: 3})
    test_data.to_csv(result_path, columns=["damage_grade"], index_label="building_id")

    if verbose:
        print(f"Saved final prediction in '{result_path}'")
