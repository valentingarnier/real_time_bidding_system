import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler

def inspect_columns(df):
    """A helper function that does a better job than df.info() and df.describe()"""

    total_rows = len(df)
    result = pd.DataFrame({
        'total_rows': [total_rows] * df.shape[1],
        'rows_with_missing_values': df.isnull().sum(),
        'unique': df.nunique() == total_rows,
        'cardinality': df.nunique(),
        'with_null': df.isna().any(),
        'null_pct': round((df.isnull().sum() / total_rows) * 100, 2),
        '1st_row': df.iloc[0],
        'random_row': df.iloc[np.random.randint(low=0, high=total_rows)],
        'last_row': df.iloc[-1],
        'dtype': df.dtypes,
    })
    return result

def categorize_columns(train, target_variable):
    numerical_data_types = ['int64', 'float64', 'datetime64[ns]']  # adjusted data types
    categorical_data_types = ['object']

    numerical_columns = [column for column in train.columns if train[column].dtype in numerical_data_types]
    print(f"Numerical variables ({len(numerical_columns)}): {numerical_columns}")

    categorical_columns = [column for column in train.columns if train[column].dtype in categorical_data_types]

    if target_variable in categorical_columns:
        categorical_columns.remove(target_variable)
    print(f"Categorical variables ({len(categorical_columns)}): {categorical_columns}")

    return numerical_columns, categorical_columns

def handle_missing_data(df, categorical_columns, numerical_columns, train=True):
    no_missing = df.copy()
    for column in categorical_columns:
        mode_val = no_missing[column].mode()[0]
        no_missing[column].fillna(mode_val, inplace=True)

    # Impute numerical columns with median
    for column in numerical_columns:
        median_val = no_missing[column].median()
        no_missing[column].fillna(median_val, inplace=True)
    return no_missing

def one_hot_encode_labels(data, label_columns):
    """
    One-hot encode a list of label columns in a DataFrame.

    Parameters:
    - data: DataFrame containing the label columns to be one-hot encoded.
    - label_columns: List of label column names to be one-hot encoded.

    Returns:
    - DataFrame with original data and one-hot encoded label columns.
    - Encoder object for potential inverse transformation.
    """
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_labels = encoder.fit_transform(data[label_columns])

    # Create DataFrame for encoded labels
    data.reset_index(drop=True, inplace=True)

    encoded_df = pd.DataFrame(encoded_labels, columns=encoder.get_feature_names_out(label_columns))
    encoded_df.reset_index(drop=True, inplace=True)
    # Drop the original label columns and concatenate encoded DataFrame
    data = data.drop(label_columns, axis=1)
    data_encoded = pd.concat([data, encoded_df], axis=1)

    return data_encoded, encoder

def minmax_scale_data(data, numerical_columns, feature_range=(0, 1)):
    """
    Apply MinMax scaling to numerical columns in a DataFrame.

    Parameters:
    - data: DataFrame containing the numerical columns to be scaled.
    - numerical_columns: List of numerical column names to be scaled.
    - feature_range: Tuple (min, max) desired for scaling. Default is (0, 1).

    Returns:
    - DataFrame with scaled numerical columns.
    - Scaler object for potential inverse transformation.
    """
    scaler = MinMaxScaler(feature_range=feature_range)

    # Scaling numerical columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data, scaler

def SelectBest(X, y, alpha=0.01, K='all'):
    # fitting the kbest function
    bestFeatures = SelectKBest(score_func=f_regression, k=K)
    fit = bestFeatures.fit(X,y)
    new_X = bestFeatures.transform(X)
    # create dataframe for the results
    dfscores = pd.DataFrame(fit.scores_)
    dfpvalue = pd.DataFrame(fit.pvalues_)
    dfsignif = pd.DataFrame(fit.pvalues_ < alpha)
    dfcolumns = pd.DataFrame(X.columns)
    # concat dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores,dfpvalue,dfsignif],axis=1)
    # naming the dataframe columns and sorting
    featureScores.columns = ['Feature','ANOVA F-stats','p_value','p_value < alpha']
    featureScores.sort_values('ANOVA F-stats', inplace=True, ascending=False)
    featureScores = featureScores.loc[featureScores['p_value < alpha'] == True]
    return featureScores, new_X

def compute_mutual_information(X, y, thre