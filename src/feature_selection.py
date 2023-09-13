import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold


def get_top_k_features_using_rfe(x_train: pd.DataFrame, y_train: pd.DataFrame, k = 0.50, step: int = 2, verbose: int = 0):
    """
    Applies Recursive Feature Elimination to get the best k features. Note that k can either be a integer, or a float.
    If it is a float, e.g. 0.5, then 50% of the total features, that are estimated to be relevant, will be selected.
    Uses a RandomForest as a classifier and therefore the _feature_importances as a scoring.

    :param x_train: DataFrame of the training data
    :param y_train: Dataframe of the labels
    :param k: Int or float: Number of features to be selected. The smaller it is, the more computation power is needed.
    :param step: Step size of the feature elimination. The smaller it is, the more computation power is needed.
    :param verbose: Verbosity level. Set to 0 to be quiet.

    :return: List of k top features, Fitted RFE object
    """

    # Define classifier
    clf = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator=clf, n_features_to_select=k, step=step, verbose=verbose)
    rfe.fit(x_train, y_train["damage_grade"].values.flatten())

    # Select feature name that are estimated to be the best features
    ranking = pd.DataFrame({"feature": x_train.columns, "rfe_support": rfe.support_})
    best_k_features = ranking[ranking["rfe_support"] == True]["feature"].to_list()

    return best_k_features, rfe


def get_top_k_features_using_rfe_cv(x_train: pd.DataFrame,
                                    y_train: pd.DataFrame,
                                    min_features_to_select=20,
                                    k_folds=5, scoring="matthews_corrcoef",
                                    step=2, verbose=0):
    """
    Applies Recursive Feature Elimination RFE with cross validation. As the estimator and thus the scoring, we use
    RandomForests respectively feauture_importance. CV is used via StratifiedKFold since we got an imbalanced target.

    :param x_train: DataFrame of the training data
    :param y_train: Dataframe of the labels
    :param min_features_to_select: Minimum number of features to keep
    :param k_folds: Number of folds in each CV
    :param scoring: Scoring metric, e.g. 'accuracy' (default is MCC)
    :param step: Step size of the RFE
    :param verbose: Verbosity level

    :return: List of k top features and fitted RFECV Object
    """

    # Define classifier
    clf = RandomForestClassifier(random_state=42)
    rfecv = RFECV(estimator=clf,
                  min_features_to_select=min_features_to_select,
                  cv=StratifiedKFold(k_folds, random_state=42, shuffle=True),
                  scoring=scoring,
                  step=step,
                  n_jobs=-1,
                  verbose=verbose)
    rfecv.fit(x_train, y_train["damage_grade"].values.flatten())

    # Select feature name that are estimated to be the best features
    ranking = pd.DataFrame({"feature": x_train.columns, "rfecv_support": rfecv.support_})
    best_k_features = ranking[ranking["rfecv_support"] == True]["feature"].to_list()

    return best_k_features, rfecv


def plot_rfe_ranking(rfe: RFE):
    """
    Plots results from RFE, i.e. feature and its rank in barplot.

    :param rfe: Fitted RFE object; Gets returned from 'get_top_k_features_using_rfe' function

    :return: None
    """

    # Create dataframe for plotting
    res = pd.DataFrame({"feature": rfe.feature_names_in_, "rfe_rank": rfe.ranking_})
    res = res.sort_values(by="rfe_rank")

    # Plot settings
    plt.figure(figsize=(20, 8), dpi=128)
    plt.title(f"Rank of Features by RFE (n_features = {rfe.n_features_} from {rfe.n_features_in_} in Total)", size=24)
    g = sns.barplot(data=res, x="feature", y="rfe_rank", edgecolor="black", palette="Spectral")
    plt.xticks(rotation=90)
    plt.bar_label(g.containers[0], padding=1.5)
    plt.show()


def get_top_k_features_using_mi(x_train: pd.DataFrame, y_train: pd.DataFrame, k: int = 30):
    """
    Computes the Mutual Information to the label and outputs the top k features as a list

    :param x_train: DataFrame of the train values
    :param y_train: DataFrame of the labels
    :param k: Number of top k features to output as list

    :return: List of top k features based on Mutual Information score
    """

    # Compute mutual information scores
    mi_scores = mutual_info_classif(x_train, y_train["damage_grade"].values.flatten())
    mi_scores = pd.Series(mi_scores)
    mi_scores.index = x_train.columns

    # Sorts descending based on mi score and gets k first features
    top_k_features =  mi_scores.sort_values(ascending=False)[: k].keys().tolist()

    return top_k_features, mi_scores


def plot_mi_ranking(mi_scores):
    """
    Plots results of the Mutual Information Score.

    :param mi_scores: Pandas Series -- Output of the get_top_k_features_using_mi function

    :return: None
    """

    mi_scores.sort_values(ascending=False, inplace=True)
    plt.figure(figsize=(20,8), dpi=128)
    plt.title("Mutual Information Scores Ordered Descending", size=24)
    g = sns.barplot(x=mi_scores.keys(), y=mi_scores.values, palette="Spectral", edgecolor="black")
    plt.xticks(rotation=90)
    plt.show()


def get_step_sizes(rfecv):
    """
    Helper function to get the number of features in each step from the RFECV.

    :param rfecv: Fitted RFECV object

    :return: List of number of features analysed
    """
    minfeat = rfecv.min_features_to_select
    startfeat = rfecv.n_features_in_
    step = rfecv.step

    res = []
    res.append(startfeat)

    while startfeat > minfeat:
        startfeat = startfeat - step
        res.append(startfeat)

    res.pop()
    res.append(minfeat)

    return list(set(res))


def plot_rfecv_scoring(rfecv):
    """
    Plots the Mean Test Scoring Value (MCC) of the steps of the RFECV.

    :param rfecv: Fitted RFECV object

    :return: None
    """
    plt.figure(figsize=(12,5))
    stepsize = get_step_sizes(rfecv)
    sns.lineplot(x=stepsize, y=rfecv.cv_results_["mean_test_score"])
    max_y = np.max(rfecv.cv_results_["mean_test_score"])
    min_y = np.min(rfecv.cv_results_["mean_test_score"])
    # Plot errorbars
    plt.errorbar(x=stepsize,
                 y=rfecv.cv_results_["mean_test_score"],
                 yerr=rfecv.cv_results_["std_test_score"],
                 fmt='o', color='red', alpha=0.5, label="SD")
    plt.axhline(y=max_y, color="green", linestyle="--", alpha=0.5, label=f"max = {max_y:.4f}")
    plt.axhline(y=min_y, color="grey", linestyle="--", alpha=0.5, label=f"min = {min_y:.4f}")
    plt.xticks(stepsize)
    plt.title(f"Recursive Feature Elimination with CV\n"+
              f"(best n_features = {rfecv.n_features_} of {rfecv.n_features_in_} in total)", size=16)
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Mean Test MCC")
    plt.legend()
    plt.show()