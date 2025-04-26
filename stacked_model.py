from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from model_pipeline import *

# Global variables
MODEL_PATH = "stacked_model.pkl"

def get_base_models():
    """
    Get the list of base model

    Returns:
        models (list): List of base models
    """

    # Initialize model list
    models = list()

    # Append Logistic Regression Model
    models.append(LogisticRegression())

    # Append Decision Tree Classifier
    models.append(DecisionTreeClassifier())

    # Append Random Forest Classifier
    models.append(RandomForestClassifier(n_estimators=10))

    # Append SGD Classifier
    models.append(SGDClassifier())

    # Append SVC
    models.append(SVC(kernel='rbf'))

    # Append KNN Classifier
    models.append(KNeighborsClassifier())

    # Append Ridge Classifier
    models.append(RidgeClassifier())

    # Append Ada Boost Classifier
    models.append(AdaBoostClassifier())

    # Append Gradient Boost Classifier
    models.append(GradientBoostingClassifier())

    # Append XG Boost Classfier
    models.append(XGBClassifier())

    return models


def predict_validation(X_train, X_val, y_train):
    """
    Fit and get predict from the list of base model

    Parameters:
        X_train (NumPy ndarray): transform data of X train.
        X_val (NumPy ndarray): transform data of X val.
        y_train (NumPy ndarray): data of y train.

    Returns:
        df_predictions (pd.DataFrame): DataFrame contains each model predictions
                                       from validation data
        models (list): List of base models
    """

    # Create new DataFrame for contains base models predictions from validation data
    df_predictions = pd.DataFrame()

    # Get the list of base models
    models = get_base_models()

    # Loop through each model in the list
    for model in models:

        # Get name of the model
        model_name = model.__class__.__name__

        # Fit each models
        model.fit(X_train, y_train)

        # Get predictions of each model from validation data
        predictions = get_predictions(model, X_val)

        # Save the predictions in data frame
        df_predictions[model_name] = predictions

    return df_predictions, models


def predict_test(models, X_test, y_test):
    """
    Get predict for test data from the list of base model

    Parameters:
        models (list): List of base models
        X_test (NumPy ndarray): transform data of X test.
        y_test (NumPy ndarray): data of y test.

    Returns:
        df_test_predictions (pd.DataFrame): DataFrame contains each model
                                                  predictions from test data
    """

    # Create new DataFrame for contains base models predictions from test
    df_test_predictions = pd.DataFrame()

    # Loop through each model in the list
    for model in models:

        # Get name of the model
        model_name = model.__class__.__name__

        # Get predictions of each model from validation data
        test_predictions = get_predictions(model, X_test)

        # Evaluate each model based on validation data
        print(f"\nEvaluate {model_name} Model:")
        evaluate_model(y_test, test_predictions)

        # Save the predictions in data frame
        df_test_predictions[model_name] = test_predictions

    return df_test_predictions


def build_stacked_model(X, y):
    """
    Build and fit Stacked Model with X and y.

    Parameters:
        X (pd.DataFrame): Dataframe with selected columns.
        y (pd.DataFrame): Dataframe with target column.
    """

    # Get train, test and validation data
    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val(X, y)

    # Get the predictions DataFrame and list of base model
    df_predictions, models \
        = predict_validation(X_train, X_val, y_train)

    # Build and fit the Stacked Model
    stacked_model = LogisticRegression().fit(df_predictions, y_val)

    # Get the predictions DataFrame from test data
    df_test_predictions = predict_test(models, X_test, y_test)

    # Get the predictions with Stacked Model
    stacked_predictions = get_predictions(stacked_model, df_test_predictions)

    # Evaluate stacked model
    print("\nEvaluate Stacked Model: Logistic Regression")
    evaluate_model(y_test, stacked_predictions)

    # Export stacked model
    export_model(stacked_model, MODEL_PATH)


build_stacked_model(final_X, final_y)
