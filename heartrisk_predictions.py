import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import accuracy_score

# Global variables
warnings.filterwarnings("ignore")
TEST_PATH = "HeartRisk_Test.csv"
ANN_MODEL_PATH = "ann_model.pkl"
STACKED_MODEL_PATH = "stacked_model.pkl"
SCALER_PATH = "minmax_scaler.pkl"
PRED_PATH = "HeartRisk_Predictions.csv"
TARGET = "HighRisk"
NUM_BASE_MODELS = 10

def read_csv(path):
    """
    Read the CSV file.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        df (pd.DataFrame): Raw Dataframe.
    """

    # Reading raw data
    df = pd.read_csv(path)

    return df


# Read the CSV file.
df = read_csv(TEST_PATH)

# Get X DataFrame with selected columns.
X = df.copy()
del X[TARGET]

# Load the MinMax Scaler
scaler = joblib.load(SCALER_PATH)

# Scale the data
X_scaled = scaler.fit_transform(X)

# Load the ANN model
ann_model = joblib.load(ANN_MODEL_PATH)

# Load the Stacked model
stacked_model = joblib.load(STACKED_MODEL_PATH)

# Create stacked features list
stacked_features_list = list()

# Make 10 predictions with the ANN Model
# 10 is the number of base models in the Stacked Model
# Can be get from get_base_models() in stacked_model.py
for i in range(NUM_BASE_MODELS):
    ann_prediction = ann_model.predict(X_scaled)
    stacked_features_list.append(ann_prediction)

# Combine stacked features list predictions to stacked features 2D Array
stacked_features_array = np.column_stack(stacked_features_list)

# Convert 2D Array to DataFrame
df_stacked = pd.DataFrame(stacked_features_array)

# Pass the prediction to the stacked model
stacked_predictions = stacked_model.predict(df_stacked)

# Compare accuracy of predictions with real data
accuracy = accuracy_score(df[TARGET], stacked_predictions)
print("Accuracy of predictions:", str(accuracy))

# Store the final predictions
df[TARGET] = stacked_predictions

# Export to csv file
df[[TARGET]].to_csv(PRED_PATH, index=False)
