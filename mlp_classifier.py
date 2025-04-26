from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from model_pipeline import *

# Global variables
MODEL_PATH = "mlp_model.pkl"
PARAMS = {
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'hidden_layer_sizes': [(200,200), (300,200), (150,150)],
    'activation': ["logistic", "relu", "tanh"]
}

def tuning_model(model, X_val, y_val):
    """
    Tuning MLP Model to find the best parameters.

    Parameters:
        model: MLP model.
        X_val (NumPy ndarray): transform data of X val.
        y_val (NumPy ndarray): data of y val.

    Returns:
        best_params (Dictionary): a dictionary with best parameter values.
    """

    # Grid search of parameters, using 4 fold cross validation,
    # search across param grid, and use all available cores
    model = GridSearchCV(estimator=model, param_grid=PARAMS,
                         scoring='accuracy', cv=4,
                         n_jobs=-1, verbose=1,
                         return_train_score=False)

    # Fit the grid search model with validation data
    model.fit(X_val, y_val)

    # Print out the best parameter for grid search model
    best_params = model.best_params_
    print("\nBest parameters: ", best_params)

    return best_params


def build_mlp_model(X, y):
    """
    Build MLP Classifier Model with X and y.

    Parameters:
        X (pd.DataFrame): Dataframe with selected columns.
        y (pd.DataFrame): Dataframe with target column.
    """

    # Get train, test and validation data
    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val(X, y)

    # Build the MLP Classifier for tuning
    model = MLPClassifier()

    # Tuning model using validation data to get best parameters
    best_params = tuning_model(model, X_val, y_val)

    # Build and fit the MLP Classifier with best parameters
    model = MLPClassifier(activation=best_params['activation'],
                          hidden_layer_sizes=best_params['hidden_layer_sizes'],
                          learning_rate=best_params['learning_rate'],
                          solver=best_params['solver']
                         ).fit(X_train, y_train)

    # Get predictions for MLP Classifier Model
    predictions = get_predictions(model, X_test)

    # Evaluate MLP Classifier Model
    print(f"\nEvaluate MLP Classifier Model")
    evaluate_model(y_test, predictions)


build_mlp_model(final_X, final_y)
