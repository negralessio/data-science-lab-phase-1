import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def encode_train_data(x_train: pd.DataFrame):
    """
    Encodes the train data using sklearn One Hot Encoded and returns fitted OHE object.
    We need the fitted OHE object to transform our test data thus we also return the OHE object.

    :param x_train:
    :return: Encoded DataFrame and fitted OHE object
    """
    x_train_cats = x_train.select_dtypes(['object', 'category'])

    # Fit One Hot Encoding Object
    ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.int64)
    x_train_cats_encoded = ohe.fit_transform(x_train_cats).toarray()

    # Transform encoded data to pandas dataframe
    x_train_cats_encoded = pd.DataFrame(x_train_cats_encoded, columns=ohe.get_feature_names_out(), index=x_train.index)

    # Drop old features
    feats_to_drop = list(ohe.feature_names_in_)
    x_train = x_train.drop(columns=feats_to_drop, axis=1)

    # Concat old dataframe with new encoded features
    x_train_encoded = pd.concat([x_train, x_train_cats_encoded], axis=1)

    return x_train_encoded, ohe


def encode_test_data(x_test: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """
    Applies the already fitted OHE object on the test dataframe x_test.
    First extracts categorical columns from the x_test and transforms them using the ohe object.
    Then the encoded data gets concatenated with the remaining not encoded features.

    :param x_test: Test DataFrame to transform using the fitted OHE object
    :param ohe: Fitted OneHotEncoder Object yielded from 'encode_train_data()' function
    :return: Encoded DataFrame
    """
    # Get categorical columns and transform them using already fitted OHE object
    x_test_cats = x_test.select_dtypes(['object', 'category'])
    x_test_cats_encoded = ohe.transform(x_test_cats).toarray()

    # Transform to pandas DataFrame
    x_test_cats_encoded = pd.DataFrame(x_test_cats_encoded, columns=ohe.get_feature_names_out(), index=x_test.index)

    # Drop old features
    feats_to_drop = list(ohe.feature_names_in_)
    x_test = x_test.drop(columns=feats_to_drop, axis=1)

    # Concat old dataframe with new encoded features
    x_test_encoded = pd.concat([x_test, x_test_cats_encoded], axis=1)

    return x_test_encoded


def normalize_train_data(x_train: pd.DataFrame, method: str = "standard"):
    """
    Function to normalize the train data.
    Fits StandardScaler on given train DataFrame and also outputs the scaler.

    :param x_train: train DataFrame
    :param method: Method to scale, either 'standard' or 'minmax'
    :return: Scaled Train DataFrame
    """
    assert method in ["standard", "minmax"], print("method must either be 'standard' or 'minmax'")

    scaler = None
    if method=="standard":
        scaler = StandardScaler()
    if method=="minmax":
        scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    # Transform back to pandas DataFrame
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
    return x_train_scaled, scaler


def normalize_test_data(x_test: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Function to normalize the test data. Uses already fitted StandardScaler or MinMax scaler object
    to transform given DataFrame.

    :param x_test: test DataFrame
    :param scaler: Fitted StandardScaler or MinMax Scaler object, yielded from 'normalize_train_data' function
    :return: Scaled Test DataFrame
    """

    x_test_scaled = scaler.transform(x_test)
    # Transform back to pandas DataFrame
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)
    return x_test_scaled


def get_risk_status_based_on_geo_level(data=None, df_to_add_info=None, labels=None, geo_level=1):
    """
    This function calculates the probability for the damage grade given the geo_level_id. 
    At first the total number of buildings in each geo_level_id is calculated (we call it t). 
    Then the number of buildings in with damage grade 1, 2 and 3 are seperately calculated for each geo_level_id (we call it d). 
    In order to calculate the risk for a specific damage grade, the ratio of buildings with the specific damage grade to the total buildings in the geo_level_id is calculated (d=/t). 
    This risk statistic is also given as a weighted number. 
    The weight is given by the ratio of buildings in the geo_level_id to all buildings in the dataset. 
    
    :param data: The raw data on which the statistics are calculated
    :param df_to_add_info: The dataframe where to add the new features
    :param labels: The series with the damage grade of each building
    :param geo_level: Specifies the geo level on which to calculate the statistics. Must be a number of {1, 2, 3}
    
    :returns a pandas Dataframe with the additional information
    """
    df = data.join(labels)
    
    # Get unique geo_level_ids and damage grades
    unique_geo_level_ids = pd.DataFrame(data={f"geo_level_{geo_level}_id": data[f"geo_level_{geo_level}_id"].unique()})
    unique_damage_grades = pd.DataFrame(data={"damage_grade": labels["damage_grade"].unique()})
    
    # Cross join
    unique_geo_level_ids["key"] = 0
    unique_damage_grades["key"] = 0
    stats_df = unique_geo_level_ids.merge(unique_damage_grades, on='key', how='outer')
    del stats_df["key"]

    # Lists to temporarily store the results
    totals = []
    damaged = []
    risks = []
    sample_size = []

    total_samples = df.shape[0]

    # Calculate stats
    for _, row in stats_df.iterrows():
        t = df[f"geo_level_{geo_level}_id"].where(df[f"geo_level_{geo_level}_id"] == row[f"geo_level_{geo_level}_id"]).count()
        totals.append(t)

        d = df[f"geo_level_{geo_level}_id"].where((df[f"geo_level_{geo_level}_id"] == row[f"geo_level_{geo_level}_id"]) & (df["damage_grade"] == row["damage_grade"])).count()
        damaged.append(d)

        r = d / t
        risks.append(r)

        sample_size.append(t / total_samples)

    # Add stats to dataframe
    stats_df["size_of_risk_sample"] = sample_size
    stats_df["risk"] = risks

    # Weighted rist status
    stats_df["weighted_risk_status"] = stats_df["size_of_risk_sample"] * stats_df["risk"]
    
    # Transform dataframe to the right format to join it to the data
    risk_df = pd.DataFrame(data={f"geo_level_{geo_level}_id": df[f"geo_level_{geo_level}_id"].unique()})
    
    # Lists for temporary results
    risk_sample_size = []
    dg1 = []
    dg2 = []
    dg3 = []
    dg1w = []
    dg2w = []
    dg3w = []
    
    # Look up temporary results and add them to the risk dataframe
    for _, row in risk_df.iterrows():
        t = stats_df["size_of_risk_sample"].where(stats_df[f"geo_level_{geo_level}_id"] == row[f"geo_level_{geo_level}_id"]).max()
        risk_sample_size.append(t)

        d = stats_df["risk"].where((stats_df[f"geo_level_{geo_level}_id"] == row[f"geo_level_{geo_level}_id"]) & (stats_df["damage_grade"] == 1)).max()
        dg1.append(d)
        dg1w.append(d * t)

        d = stats_df["risk"].where((stats_df[f"geo_level_{geo_level}_id"] == row[f"geo_level_{geo_level}_id"]) & (stats_df["damage_grade"] == 2)).max()
        dg2.append(d)
        dg2w.append(d * t)

        d = stats_df["risk"].where((stats_df[f"geo_level_{geo_level}_id"] == row[f"geo_level_{geo_level}_id"]) & (stats_df["damage_grade"] == 3)).max()
        dg3.append(d)
        dg3w.append(d * t)

    # Add info to dataframe
    risk_df["risk_sample_size"] = risk_sample_size
    risk_df["damage_grade_1_risk"] = dg1
    risk_df["damage_grade_2_risk"] = dg2
    risk_df["damage_grade_3_risk"] = dg3

    risk_df["damage_grade_1_risk_weighted"] = dg1w
    risk_df["damage_grade_2_risk_weighted"] = dg2w
    risk_df["damage_grade_3_risk_weighted"] = dg3w
    
    # Join features to data
    df_to_add_info = df_to_add_info.reset_index()
    risk_df = risk_df.reset_index()
    result = df_to_add_info.set_index(f"geo_level_{geo_level}_id").join(risk_df.set_index(f"geo_level_{geo_level}_id"))
    result = result.reset_index()
    result = result.set_index("building_id")
    
    return result


def get_quality_of_superstructure(raw_data=None, df_to_add_info=None):
    """
    The used superstructure has an influence on the resistance of a building against an earthquake. 
    After some research the result was as follws:
    Good superstructures: Steel, bamboo, timber, reinforced concrete
    Bad superstructures: Bricks, stone, mud
    Based on the features in the raw data the following ordinal feature is created: 
    Good superstructures get the value 1, Bad superstructures get the value -1 and everything else 0 (including combinations). 
    
    :param raw_data The raw dataframe including the has_superstructure_X columns
    :param df_to_add_info The dataframe where to add the information to
    
    :returns A dataframe with the addtitonal feature 'superstructure_quality'
    """
    # encode superstructure as good = 1, no idea = 0; bad = -1
    # Also set combinations of good+bad, good+other, bad+other to 0
    
    # Default to -1 --> all bad are right
    raw_data["superstructure_quality"] = -1

    # Update all good superstructures
    raw_data.loc[(raw_data["has_superstructure_bamboo"] == 1) | 
                (raw_data["has_superstructure_rc_engineered"] == 1) | 
                (raw_data["has_superstructure_rc_non_engineered"] == 1) | 
                (raw_data["has_superstructure_timber"] == 1), "superstructure_quality"] = 1

    # Update all other superstructures
    raw_data.loc[(raw_data["has_superstructure_other"] == 1), "superstructure_quality"] = 0

    # Update combinations of superstructures
    # Combination of good + other
    raw_data.loc[((raw_data["has_superstructure_bamboo"] == 1) | 
                (raw_data["has_superstructure_rc_engineered"] == 1) | 
                (raw_data["has_superstructure_rc_non_engineered"] == 1) | 
                (raw_data["has_superstructure_timber"] == 1)) & 
                (raw_data["has_superstructure_other"] == 1), "superstructure_quality"] = 0

    # Combination of good + bad
    raw_data.loc[((raw_data["has_superstructure_bamboo"] == 1) | 
                (raw_data["has_superstructure_rc_engineered"] == 1) | 
                (raw_data["has_superstructure_rc_non_engineered"] == 1) | 
                (raw_data["has_superstructure_timber"] == 1)) & 
                ((raw_data["has_superstructure_adobe_mud"] == 1) |
                (raw_data["has_superstructure_mud_mortar_stone"] == 1) |
                (raw_data["has_superstructure_cement_mortar_stone"] == 1) |
                (raw_data["has_superstructure_mud_mortar_brick"] == 1) |
                (raw_data["has_superstructure_cement_mortar_brick"] == 1) |
                (raw_data["has_superstructure_stone_flag"] == 1)), "superstructure_quality"] = 0

    # Combination of bad + other
    raw_data.loc[(raw_data["has_superstructure_other"] == 1) & 
                ((raw_data["has_superstructure_adobe_mud"] == 1) |
                (raw_data["has_superstructure_mud_mortar_stone"] == 1) |
                (raw_data["has_superstructure_cement_mortar_stone"] == 1) |
                (raw_data["has_superstructure_mud_mortar_brick"] == 1) |
                (raw_data["has_superstructure_cement_mortar_brick"] == 1) |
                (raw_data["has_superstructure_stone_flag"] == 1)), "superstructure_quality"] = 0
    
    # Join new info to df
    result = df_to_add_info.join(raw_data[["superstructure_quality"]], how="left")
    
    return result


from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE


def dimensionality_reduction(train_data=None, test_data=None, method=None, n_neighbors=None, n_components=2, random_seed=42):
    """
    Applies a dimensionality reduction method on the given data. 
    Implemented methods are LLE, Isomap, PCA and TSNE. 
    
    The features in the given dataframe have to be numerical and normalized.
    
    
    """
    assert method in ["LLE", "Isomap", "PCA", "TSNE"]
    
    if method == "LLE":
        embedding = LocallyLinearEmbedding(n_components=n_components)
    elif method == "Isomap":
        embedding = Isomap(n_components=n_components)
    elif method == "PCA":
        embedding = PCA(n_components=n_components, random_state=random_seed)
    elif method == "TSNE":
        embedding = TSNE(n_components=n_components)
    else: 
        print(f"[ERROR] Dimensionality reduction method '{method}' is not implemented")
        
    train_data_transformed = embedding.fit_transform(train_data)
    train_data_transformed = pd.DataFrame(train_data_transformed, index=train_data.index)
    test_data_transformed = embedding.transform(test_data)
    test_data_transformed = pd.DataFrame(test_data_transformed, index=test_data.index)
    
    return train_data_transformed, test_data_transformed
