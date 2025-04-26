import pandas as pd
import numpy as np
import warnings
from scipy import stats
from imblearn.over_sampling import SMOTE
from plot_draw import draw_highrisk_count_plot

# Global variables
warnings.filterwarnings("ignore")
CSV_PATH = "HeartRisk_Preprocessed.csv"
THRESHOLD = 3
TARGET = "HighRisk"

def read_csv(path):
    """
    Read the CSV file.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        df (pd.DataFrame): Dataframe.
    """

    # Reading raw data
    df = pd.read_csv(path)

    return df


def get_x_y(df):
    """
    Get the X and y from DataFrame.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        X (pd.DataFrame): Dataframe with selected columns.
        y (pd.DataFrame): Dataframe with target column.
    """

    # Get X DataFrame with selected columns (except HighRisk column).
    X = df.copy()
    del X[TARGET]
    selected_features = X[['Sex', 'PhysicalActivities', 'HadHeartAttack',
                           'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
                           'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease',
                           'HadArthritis', 'DeafOrHardOfHearing',
                           'BlindOrVisionDifficulty', 'DifficultyConcentrating',
                           'DifficultyWalking', 'DifficultyDressingBathing',
                           'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers',
                           'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver'
                           ]]

    # Get y DataFrame with the target column.
    y = df[TARGET]

    return X, y, selected_features


def scale_data(X, scaler):
    """
    Scale data using passing Scaler.

    Parameters:
        X (pd.DataFrame): Dataframe needed to scale.
        scaler (Scaler): Scaler that been passed from parameter.

    Returns:
        X_scaled (NumPy ndarray): NumPy array containing the transformed data.
    """

    # Fit the scaler to X
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def drop_outliers(X, y):
    """
    Drop outliers in X and y.

    Parameters:
        X (pd.DataFrame): Dataframe with selected columns.
        y (pd.DataFrame): Dataframe with target column.

    Returns:
        X (pd.DataFrame): X without outliers.
        y (pd.DataFrame): y without outliers.
    """

    # Calculate Z score for each value in X
    z = np.abs(stats.zscore(X))

    # Find any value have z-score greater than THRESHOLD
    outliers = np.where(z > THRESHOLD)

    # Remove outliers from X and y
    X.drop(outliers[0], inplace=True)
    y.drop(outliers[0], inplace=True)

    # Reset index for both X and y after dropping
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y


def smote_balancing(X, y):
    """
    Applied SMOTE Balancing for X and y.

    Parameters:
        X (pd.DataFrame): Dataframe with selected columns.
        y (pd.DataFrame): Dataframe with target column.

    Returns:
        X (pd.DataFrame): X that have SMOTE Balancing.
        y (pd.DataFrame): y that have SMOTE Balancing.
    """

    # Create a SMOTE object.
    sm = SMOTE(random_state=42)

    # Apply SMOTE to X and y.
    X_sm, y_sm = sm.fit_resample(X, y)

    return X_sm, y_sm


df = read_csv(CSV_PATH)
X, y, selected_features = get_x_y(df)
smote_X, smote_y = smote_balancing(X, y)
draw_highrisk_count_plot(smote_y)
final_X, final_y = drop_outliers(smote_X, smote_y)
