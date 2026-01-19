import joblib
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from data_transformation import *

# Global variables
TEST_PATH = "HeartRisk_Test.csv"
SCALER_PATH = "minmax_scaler.pkl"
SCALER = MinMaxScaler()

def export_csv(df, path):
    """
    Export DataFrame to csv file.

    Parameters:
        df (pd.DataFrame): Dataframe.
        path (str): Path for new csv file.
    """

    df.to_csv(path, index=False)


def get_train_test_val(X, y):
    """
    Get train, test and validation data

    Parameters:
        X (pd.DataFrame): Dataframe with selected columns.
        y (pd.DataFrame): Dataframe with target column.

    Returns:
        X_train_scaled (NumPy ndarray): transform data of X train.
        X_test_scaled (NumPy ndarray): transform data of X test.
        X_val_scaled (NumPy ndarray): transform data of X val.
        y_train (NumPy ndarray): data of y train.
        y_test (NumPy ndarray): data of y test.
        y_val (NumPy ndarray): data of y val.
    """

    # Split data into train, test and validation sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # Export test data to csv file
    export_csv(pd.concat([X_test, y_test], axis=1), TEST_PATH)

    # Scale X train, test and validation data
    X_train_scaled = scale_data(X_train, SCALER)
    X_test_scaled = scale_data(X_test, SCALER)
    X_val_scaled = scale_data(X_val, SCALER)

    return X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val


def get_predictions(model, X_test):
    """
    Get the predictions from model

    Parameters:
        model: Passing model.
        X_test (NumPy ndarray): transformed data of X test.

    Returns:
        predictions (NumPy ndarray): Predicted target data..
    """

    # Make the probabilities by the model.
    predictions = model.predict(X_test)

    return predictions


def evaluate_model(y_test, predictions):
    """
    Evaluate appropriate model.

    Parameters:
        y_test (pd.DataFrame): Dataframe with target column.
        predictions (NumPy ndarray): NumPy array with predicted values.
    """

    # Convert the imp_Claim column to NumPy array
    y_test_array = np.array(y_test)

    # Create a confusion matrix
    cm = pd.crosstab(y_test_array, predictions, rownames=['Actual'], colnames=['Predicted'])

    # Evaluate confusion matrix
    print("\nConfusion Matrix")
    print(cm)
    print(metrics.classification_report(y_test, predictions))

    # Evaluate accuracy
    accuracy = metrics.accuracy_score(y_test, predictions)
    print('\nAccuracy:', accuracy)

    # Evaluate precision
    precision = metrics.precision_score(y_test, predictions)
    print('Precision:', precision)

    # Evaluate recall
    recall = metrics.recall_score(y_test, predictions)
    print('Recall:', recall)

    # Evaluate f1
    f1 = metrics.f1_score(y_test, predictions)
    print('F1:', f1)


def export_model(model, path):
    """
    Export model and scaler as .pkl file.

    Parameters:
        model: passing model.
        scaler: passing scaler - in this case is Min Max Scaler
    """

    joblib.dump(model, path)
    joblib.dump(SCALER, SCALER_PATH)
