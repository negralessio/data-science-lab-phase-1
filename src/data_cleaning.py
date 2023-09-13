""" This python file contains all functionality for the data cleaning process."""
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn import preprocessing
from scipy.stats import chi2_contingency
from pandas.api.types import is_numeric_dtype


def format_dtypes(df):
    """
    Formats the data types of columns in a pandas DataFrame.

    :param df: The input dataframe to be formatted.
    :type df: pandas.DataFrame

    :return: The formatted dataframe.
    :rtype: pandas.DataFrame
    """
    # categorical values
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df[cat_cols] = df[cat_cols].astype('category')

    return df


def handle_outliers_IQR(df, ignore_cols, iqr_factor=1.5, method="replace", verbose=True):
    """
    Handle outliers in a mixed categorical and numerical dataframe using the IQR method. Note that this function
     only drops outliers based on IQR and does not consider any categorical features. It is assumed that any categorical
     features in the dataframe will not contribute to the detection of outliers.

    :param df: The input dataframe.
    :type df: pd.DataFrame
    :param ignore_cols: The list of columns to ignore.
    :type ignore_cols: list(str)
    :param iqr_factor: The factor to be multiplied with the IQR to determine the outlier bounds.
    :type iqr_factor: float,optional(default=1.5)
    :param method: The method to handle outliers. The options are:
        -"hard_drop": drops the rows with outliers.
        -"soft_drop": drops the rows where more than 30% of the row values are outliers.
        -"replace": replaces the outliers with the upper or lower limit.
    :type method: str,optional(default="drop")

    :return: The processed dataframe with outliers handled.
    :rtype: pd.DataFrame
    """
    # Identify the numerical columns in the dataframe
    tmp = df[[col for col in df.columns.tolist() if (is_numeric_dtype(df[col]) and col not in ignore_cols)]]
    non_tmp_df = df[df.columns.difference(tmp.columns)]

    # Calculate the IQR for each numerical column
    Q1 = tmp.quantile(0.25)
    Q3 = tmp.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - (iqr_factor * IQR)
    upper_bound = Q3 + (iqr_factor * IQR)

    # Identify the rows where any numerical value is outside the bounds aka identify the outliers
    outlier_mask = (tmp < lower_bound) | (tmp > upper_bound)
    outlier_rows = outlier_mask.any(axis=1)

    if verbose:
        print(f"Found {outlier_mask.sum().sum()} outliers,using method '{method}' to handle them:")
        print(f"Count per column: {outlier_mask.sum().to_dict()}")
        print(f"Lower bound: {lower_bound.to_dict()}")
        print(f"Upper bound: {upper_bound.to_dict()}")

    if method == "hard_drop":
        # Drop rows with outliers
        df_no_outliers = df[~outlier_rows]

        return df_no_outliers

    elif method == "soft_drop":
        # Compute the number of outliers per row and drop rows based on the threshold (30%)
        count_outliers = outlier_mask.notna().sum(axis=1)
        drop_indices = count_outliers[count_outliers >= 30 * len(tmp.columns)].index
        df_filtered_outliers = df.drop(drop_indices)

        return df_filtered_outliers

    elif method == "replace":
        upper_limit = tmp.mean() + 3 * tmp.std()
        lower_limit = tmp.mean() - 3 * tmp.std()
        # Replace outliers with upper or lower limit
        tmp = tmp.clip(lower=lower_limit, upper=upper_limit, axis=1)
        # Concat with other data
        df_replaced = pd.concat([non_tmp_df, tmp], axis=1)

        return df_replaced

    else:
        raise ValueError(f"Invalid outlier method: {method}, must be either 'hard_drop' or 'replace' or 'soft_drop'.")


def prepare_data(df, config, ignore_cols, outlier_method, verbose=True):
    """
    Cleans the input pandas DataFrame by dropping unnecessary columns, formatting column data types,
    and handling outliers.

    :param df: The input dataframe to be processed.
    :type df: pandas.DataFrame
    :param config: The configuration file containing the pipeline settings.
    :type config: configparser.ConfigParser
    :param ignore_cols: The list of columns to ignore when handling outliers.
    :type ignore_cols: list(str)
    :param outlier_method: The method to handle outliers. The options are:
        -"hard_drop": Drops the rows with outliers.
        -"soft_drop": Drops the rows where more than 30% of the row values are outliers.
        -"replace": Replaces the outliers with the upper or lower limit.
    :type outlier_method: str

    :return: The processed dataframe. The columns are dropped, the data types are formatted, and the outliers are
    handled.
    :rtype: pandas.DataFrame
    """
    # Drop unnecessary columns
    drop_cols = config.get("data_cleaning", "NO DATA CLEANING DEFINED").get("columns_to_remove")
    df = df.drop(drop_cols, axis=1)
    # Format column types
    df = format_dtypes(df)
    # Handle outliers
    df = handle_outliers_IQR(df, ignore_cols=ignore_cols, method=outlier_method, verbose=verbose)

    return df


def group_categorical_features(df: pd.DataFrame, default_val: str = "others", verbose: bool = False) -> pd.DataFrame:
    """
    Function that groups categorical features with many unique low populated realizations into "others".
    Significant for feature that have rarely occurring categorical values, e.g. "plan_configuration".
    Predefined mappings are within the function and contain the column to apply the mapping and the map itself.
    Not explicitly listed mappings are replaced with param "default_val", which is "others" (default)

    :param df: The input dataframe.
    :type df: pd.DataFrame
    :param default_val: The value to replace the rarely occurring categorical values with.
    :type default_val: str,optional(default="others")
    :param verbose: Whether to print the applied mappings or not.
    :type verbose: bool,optional(default=False)

    :return: The dataframe with grouped categorical features.
    :rtype: pd.DataFrame
    """
    # Define mapping for each feature with rarely occurring categorical values
    mapping_plan_configuration = {"col": "plan_configuration",
                                  "mapping": {"d": "d"}}
    mapping_legal_ownership_status = {"col": "legal_ownership_status",
                                      "mapping": {"v": "v"}}
    mapping_ground_floor_type = {"col": "ground_floor_type",
                                 "mapping": {"f": "f", "x": "x", "v": "v"}}
    # Put in list for iteration
    mappings = [mapping_plan_configuration, mapping_legal_ownership_status, mapping_ground_floor_type]

    # Apply mapping for each mapping defined in mappings list
    for mapping in mappings:
        # Overwrite column to be mapped with mapped values
        df[mapping.get("col")] = df[mapping.get("col")].map(mapping.get("mapping")).fillna(default_val)
        if verbose:
            print(f"Applied mapping / grouping for feature '{mapping.get('col')}'")

    return df


def get_pearson_correlated_features(data=None, threshold=0.7):
    """
    Calculates the pearson correlation of all features in the dataframe and returns a set of features with a
    correlation greater than the threshold.

    :param data: The input dataframe.
    :type data: pd.DataFrame
    :param threshold: The threshold for the correlation coefficient in the range of [0.0, 1.0].
    :type threshold: float,optional(default=0.7)

    :return: The set of features with a correlation greater than the threshold.
    :rtype: set
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Get the set of correlated features
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)

    return correlated_features


def get_cramers_v_correlated_features(data=None, threshold=0.7):
    """
    Calculates the cramers V correlation of all features and returns a set of features with a correlation greater
    than the threshold. Cramers V is based on Chi square, for reference
    see: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Note that this function is desined to work for categorical features only!
    Code was copied and modified from this source:
    https://www.kaggle.com/code/chrisbss1/cramer-s-v-correlation-matrix/notebook

    :param data: The input dataframe.
    :type data: pd.DataFrame
    :param threshold: The threshold for the correlation coefficient in the range of [0.0, 1.0].
    :type threshold: float,optional(default=0.7)

    :return: The set of features with a correlation greater than the threshold.
    :rtype: set
    """
    # Encode features
    label = preprocessing.LabelEncoder()
    data_encoded = pd.DataFrame()

    for i in data.columns:
        data_encoded[i] = label.fit_transform(data[i])

    # Internal function to calculate cramers V for two features
    def _cramers_V(var1, var2):
        crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))  # Cross table building
        stat = chi2_contingency(crosstab)[0]  # Keeping of the test statistic of the Chi2 test
        obs = np.sum(crosstab)  # Number of observations
        mini = min(crosstab.shape) - 1  # Take the minimum value between the columns and the rows of the cross table
        return (stat / (obs * mini))
        # return stat

    # Calculate values for each pair of features
    rows = []
    for var1 in data_encoded:
        col = []
        for var2 in data_encoded:
            cramers = _cramers_V(data_encoded[var1], data_encoded[var2])  # Cramer's V test
            col.append(round(cramers, 4))  # Keeping of the rounded value of the Cramer's V
        rows.append(col)

    # Create a pandas df from the results
    cramers_results = np.array(rows)
    corr_matrix = pd.DataFrame(cramers_results, columns=data_encoded.columns, index=data_encoded.columns)

    # Get the set of correlated features
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)

    return correlated_features


def get_mcc_correlated_features(data=None, threshold=0.7):
    """
    Calculates the MCC correlation of all features and returns a set of features with a correlation greater than
    the threshold. Cramers V is based on Chi square, for reference see: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Note that this function is designed to work for categorical features only!
    Code was copied and modified from this source:
    https://www.kaggle.com/code/chrisbss1/cramer-s-v-correlation-matrix/notebook

    :param data: The input dataframe.
    :type data: pd.DataFrame

    :return: The set of features with a correlation greater than the threshold.
    :rtype: set
    """
    # Encode features
    label = preprocessing.LabelEncoder()
    data_encoded = pd.DataFrame()

    label = preprocessing.LabelEncoder()
    data_encoded = pd.DataFrame()

    for c in data.columns:
        if c in data.columns:
            data_encoded[c] = label.fit_transform(data[c])
        else:
            data_encoded[c] = data[c]

    # Calculate values for each pair of features
    rows = []
    for var1 in data_encoded:
        col = []
        for var2 in data_encoded:
            phi = matthews_corrcoef(data_encoded[var1], data_encoded[var2])
            col.append(phi)  # phi
        rows.append(col)

    # Create a pandas df from the results
    phi_results = np.array(rows)
    corr_matrix = pd.DataFrame(phi_results, columns=data_encoded.columns, index=data_encoded.columns)

    # Get the set of correlated features
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)

    return correlated_features


def drop_correlated_features(data=None, config=None):
    """
    Gets the correlated features according to the configuration and drops them from the provided dataframe.
    Then the dataframe without the correlated features is returned.

    Example for the config:
    [
        {
            'feature_names': <List of feature names>,
            'threshold'    : A number between 0 and 1
            'method'       : <One out of the set {'MCC', 'CramesV', 'Pearson'}>
        },
        {
            'feature_names': <List of feature names>,
            'threshold'    : A number between 0 and 1
            'method'       : <One out of the set {'MCC', 'CramesV', 'Pearson'}>
        }, ...
    ]

    :param data: The input dataframe to drop the correlated features from.
    :type data: pd.DataFrame
    :param config: A list of dicts. Every dict has to contain the keys 'feature_names', 'method' and 'threshold'.
                   The 'feature_names' determine of which features the correlation is calculated.
                   Method has to be one out of the set {'MCC', 'CramesV', 'Pearson'}.
                   The value of method determines the function which is used to calculate the correlations.
                   Only features with a higher correlation than 'threshold' will be dropped.
    :type config: list

    :returns: The dataframe without the correlated features.
    :rtype: pd.DataFrame
    """
    # Corr methods: MCC, CramersV, Pearson
    corr_methods_collection = {'MCC': get_mcc_correlated_features,
                               'CramersV': get_cramers_v_correlated_features,
                               'Pearson': get_pearson_correlated_features}

    # Traverse all correlation dicts in the config
    # Note: This could be parallelized
    for d in config:
        feature_names = [x for x in data.columns if x in d['feature_names']]
        # Get correlated features
        try:
            corr_features = corr_methods_collection[d['method']](data=data[feature_names], threshold=d['threshold'])
        except KeyError:
            print(f"Correlation method '{d['method']}' is not implemented.")

        # Drop features
        if len(corr_features) > 0:
            data = data.drop(corr_features, axis=1)

    return data
