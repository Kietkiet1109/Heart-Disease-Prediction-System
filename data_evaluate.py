import pandas as pd
import warnings
from plot_draw import plot_all_features_heatmap, draw_highrisk_count_plot

# Global variables
warnings.filterwarnings("ignore")
CSV_PATH = "HeartRisk.csv"
PREPROCESSED_CSV_PATH = "HeartRisk_Preprocessed.csv"
N_COLS = 4
N_ROWS = 5

def read_csv(path):
    """
    Read the CSV file.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        df (pd.DataFrame): corresponding Dataframe.
    """

    # Reading CSV data
    df = pd.read_csv(path)

    return df


def evaluate_data(df, preprocessed_df):
    """
    Evaluate DataFrame.

    Parameters:
        df (pd.DataFrame): Raw Dataframe.
        preprocessed_df (pd.DataFrame): Preprocessed Dataframe.
    """

    # Config data display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Print the DataFrame
    print(df)

    # Print the DataFrame summary
    print(df.describe())
    print("\n")
    print(df.info())

    # Get the features and their names.
    features = preprocessed_df[['Sex', 'PhysicalActivities', 'HadHeartAttack',
                   'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
                   'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease',
                   'HadArthritis', 'DeafOrHardOfHearing',
                   'BlindOrVisionDifficulty', 'DifficultyConcentrating',
                   'DifficultyWalking', 'DifficultyDressingBathing',
                   'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers',
                   'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRisk'
                  ]]

    # Create a count plot for Pokemon primary type
    draw_highrisk_count_plot(df['HighRisk'])

    # Create a heatmap between each feature and target (Is_Legendary)
    plot_all_features_heatmap(features)


df = read_csv(CSV_PATH)
preprocessed_df = read_csv(PREPROCESSED_CSV_PATH)
evaluate_data(df, preprocessed_df)
