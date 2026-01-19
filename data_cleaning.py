import pandas as pd
import warnings

# Global variables
warnings.filterwarnings("ignore")
RAW_CSV_PATH = "HeartRisk.csv"
PREPROCESSED_CSV_PATH = "HeartRisk_Preprocessed.csv"

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


def get_imputed_value(col_name, df, measure_type):
    """
    Calculate an imputation value for a specific column in a DataFrame.

    Parameters:
        col_name (str): Column name.
        df (pd.DataFrame): Dataframe.
        measure_type (str): Measure type (median, mode, mean).

    Returns:
        float: Imputed value.
    """

    # Check the measure type and return appropriate value
    if measure_type == "median":
        return df[col_name].median()

    elif measure_type == "mode":
        return float(df[col_name].mode())

    else:
        return df[col_name].mean()


def imputed_missing_value(col_name, df, measure_type):
    """
    Imputed missing value in a DataFrame.

    Parameters:
        col_name (str): Column name.
        df (pd.DataFrame): Dataframe.
        measure_type (str): Measure type (median, mode, mean).

    Returns:
        df (pd.DataFrame): Dataframe with no missing value.
    """

    # Create two new column names based on original column name.
    indicator_col_name = 'm_'   + col_name # Tracks whether imputed.
    imputed_col_name   = 'imp_' + col_name # Stores original & imputed data.

    # Get imputed value depending on preference.
    imputed_value = get_imputed_value(col_name, df, measure_type)

    # Populate new columns with data.
    imputed_column = []
    indicator_column = []

    # Fill in missing values and mark imputed rows.
    for i in range(len(df)): # Loop through each row
        if pd.isna(df.loc[i, col_name]): # Check missing value
            imputed_column.append(imputed_value) # Add imputed value
            indicator_column.append(1)  # Mark as imputed
        else:
            imputed_column.append(df.loc[i, col_name])
            indicator_column.append(0)

    # Add new columns to the dataframe
    df[indicator_col_name] = indicator_column
    df[imputed_col_name] = imputed_column

    # Drop the original column
    df = df.drop(columns=[col_name])
    return df


def implement_missing(df):
    """
    Identify columns with missing value in a DataFrame and fill them.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        df (pd.DataFrame): Dataframe with no missing value.
    """


    # Identify columns with missing values (count != len(df))
    missing_columns = df.describe(include='all').T.query(f"count != {len(df)}").index.tolist()

    # Fill missing values
    for col in missing_columns:
        # Check if column is numerical
        if pd.api.types.is_numeric_dtype(df[col]):
            # Fill numeric columns with mean
            df = imputed_missing_value(col, df, "mean")

        else:
            # Fill categorical columns with mode
            df = imputed_missing_value(col, df, "mode")

    return df


def create_dummy_variables(df):
    """
    Create dummy variables in a DataFrame.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        df (pd.DataFrame): Dataframe with dummy variables.
    """

    # Create dummy variables for 7 columns
    dummy_general_health = pd.get_dummies(df['GeneralHealth'], prefix='GeneralHealth', drop_first=True).astype(int)
    dummy_last_checkup_time = pd.get_dummies(df['LastCheckupTime'], prefix='LastCheckupTime', drop_first=True).astype(int)
    dummy_removed_teeth = pd.get_dummies(df['RemovedTeeth'], prefix='RemovedTeeth', drop_first=True).astype(int)
    dummy_had_diabetes = pd.get_dummies(df['HadDiabetes'], prefix='HadDiabetes', drop_first=True).astype(int)
    dummy_smoker_status = pd.get_dummies(df['SmokerStatus'], prefix='SmokerStatus', drop_first=True).astype(int)
    dummy_cigarette_usage = pd.get_dummies(df['ECigaretteUsage'], prefix='ECigaretteUsage', drop_first=True).astype(int)
    dummy_tetanus_last_10 = pd.get_dummies(df['TetanusLast10Tdap'], prefix='TetanusLast10Tdap', drop_first=True).astype(int)
    dummy_covid_pos = pd.get_dummies(df['CovidPos'], prefix='CovidPos', drop_first=True).astype(int)
    dummy_age_category = pd.get_dummies(df['AgeCategory'], prefix='AgeCategory', drop_first=True).astype(int)

    # Merged new dummy variables columns to DataFrame
    df = pd.concat([df, dummy_general_health, dummy_last_checkup_time,
                    dummy_removed_teeth, dummy_had_diabetes, dummy_smoker_status,
                    dummy_cigarette_usage, dummy_tetanus_last_10, dummy_covid_pos,
                    dummy_age_category], axis=1)
    return df


def convert_categorical(df):
    """
    Convert categorical variables to numerical variables.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        df (pd.DataFrame): Dataframe with numerical variables.
    """

    # Remap all categorical columns to numerical
    df['Sex'] = [0 if x == 'Male' else 1 for x in df['Sex']]
    df['PhysicalActivities'] = [0 if x == 'No' else 1 for x in df['PhysicalActivities']]
    df['HadHeartAttack'] = [0 if x == 'No' else 1 for x in df['HadHeartAttack']]
    df['HadAngina'] = [0 if x == 'No' else 1 for x in df['HadAngina']]
    df['HadStroke'] = [0 if x == 'No' else 1 for x in df['HadStroke']]
    df['HadAsthma'] = [0 if x == 'No' else 1 for x in df['HadAsthma']]
    df['HadSkinCancer'] = [0 if x == 'No' else 1 for x in df['HadSkinCancer']]
    df['HadCOPD'] = [0 if x == 'No' else 1 for x in df['HadCOPD']]
    df['HadDepressiveDisorder'] = [0 if x == 'No' else 1 for x in df['HadDepressiveDisorder']]
    df['HadKidneyDisease'] = [0 if x == 'No' else 1 for x in df['HadKidneyDisease']]
    df['HadArthritis'] = [0 if x == 'No' else 1 for x in df['HadArthritis']]
    df['DeafOrHardOfHearing'] = [0 if x == 'No' else 1 for x in df['DeafOrHardOfHearing']]
    df['BlindOrVisionDifficulty'] = [0 if x == 'No' else 1 for x in df['BlindOrVisionDifficulty']]
    df['DifficultyConcentrating'] = [0 if x == 'No' else 1 for x in df['DifficultyConcentrating']]
    df['DifficultyWalking'] = [0 if x == 'No' else 1 for x in df['DifficultyWalking']]
    df['DifficultyDressingBathing'] = [0 if x == 'No' else 1 for x in df['DifficultyDressingBathing']]
    df['DifficultyErrands'] = [0 if x == 'No' else 1 for x in df['DifficultyErrands']]
    df['ChestScan'] = [0 if x == 'No' else 1 for x in df['ChestScan']]
    df['AlcoholDrinkers'] = [0 if x == 'No' else 1 for x in df['AlcoholDrinkers']]
    df['HIVTesting'] = [0 if x == 'No' else 1 for x in df['HIVTesting']]
    df['FluVaxLast12'] = [0 if x == 'No' else 1 for x in df['FluVaxLast12']]
    df['PneumoVaxEver'] = [0 if x == 'No' else 1 for x in df['PneumoVaxEver']]
    df['HighRisk'] = [0 if x == 'No' else 1 for x in df['HighRisk']]

    return df


def drop_unused_columns(df):
    """
    Drop unused columns (features) in a DataFrame.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        df (pd.DataFrame): Dataframe without unused columns.
    """

    # Initialize unused columns
    drop_columns = ['State', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth',
                    'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage', 'CovidPos',
                    'TetanusLast10Tdap', 'AgeCategory']

    # Drop the unused columns
    df.drop(drop_columns, axis=1, inplace=True)

    return df


def export_csv(df, path):
    """
    Export DataFrame to csv file.

    Parameters:
        df (pd.DataFrame): Dataframe.
        path (str): Path for new csv file.
    """

    df.to_csv(path, index=False)


df = read_csv(RAW_CSV_PATH)
converted_df = convert_categorical(df)
dummy_df = create_dummy_variables(converted_df)
drop_df = drop_unused_columns(dummy_df)
final_df = implement_missing(drop_df)
export_csv(final_df, PREPROCESSED_CSV_PATH)
