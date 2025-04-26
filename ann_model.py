import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from model_pipeline import *

# Global variables
MODEL_PATH = "ann_model.pkl"
BATCH_SIZES = [64, 128, 256]
EPOCHS = [25, 50, 100]
DIMENSION = 1

def get_ann_predictions(model, X_test):
    """
    Get the predictions from model

    Parameters:
        model: Passing model.
        X_test (NumPy ndarray): transformed data of X test.

    Returns:
        predictions (NumPy ndarray): Predicted target data..
    """

    # Make the probabilities by the model.
    probabilities = model.predict(X_test)

    # Create prediction list.
    predictions = []

    # Loop through each probabilities.
    for i in range(len(probabilities)):

        # Check each probability (> 0.5 means positive, else negative).
        if probabilities[i][0] > 0.5:
            predictions.append(1)

        else:
            predictions.append(0)

    return predictions

def tuning_model(model, X_train, y_train, X_test, y_test, X_val, y_val):
    """
    Tuning ANN Model to find the best parameters.

    Parameters:
        model: ANN model.
        X_train (NumPy ndarray): transform data of X train.
        X_test (NumPy ndarray): transform data of X test.
        X_val (NumPy ndarray): transform data of X val.
        y_train (NumPy ndarray): data of y train.
        y_test (NumPy ndarray): data of y test.
        y_val (NumPy ndarray): data of y val.

    Returns:
        best_batch_size (int): best parameter for batch size.
        best_epoch (int): best parameter for epoch.
    """

    # Initialize variables
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_batch_size = 0
    best_epoch = 0

    # Loop for each batch size and epoch
    for batch_size in BATCH_SIZES:
        for epoch in EPOCHS:
            # Fit the model
            model.fit(X_train, y_train, epochs=epoch,
                      batch_size=batch_size, verbose=1,
                      validation_data=(X_val, y_val)
                      )

            # Get predictions for model
            predictions = get_ann_predictions(model, X_test)

            # Calculate accuracy, precision, recall and f1-score for each model
            accuracy = metrics.accuracy_score(y_test, predictions)
            precision = metrics.precision_score(y_test, predictions)
            recall = metrics.recall_score(y_test, predictions)
            f1 = metrics.f1_score(y_test, predictions)

            # Compare accuracy, precision, recall and f1-score
            # with current best accuracy, precision, recall and f1-score
            if (accuracy > best_accuracy and precision > best_precision
                    and recall > best_recall and f1 > best_f1):

                # Assign new values
                best_accuracy = accuracy
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                best_batch_size = batch_size
                best_epoch = epoch

    # Print out the best parameter for ANN model
    print(f"\nBest parameters: {best_batch_size} Batches and {best_epoch} Epochs.\n")

    return best_batch_size, best_epoch


def create_ann_model(n_features):
    """
    Create ANN Sequential Model.

    Parameters:
        n_features (int): the number of features

    Returns:
        model: ANN Sequential Model
    """

    # Create Sequential Model
    model = Sequential()

    # Add first layer to Model
    model.add(Dense(64, input_dim=n_features, activation='relu'))

    # Add second layer to Model
    model.add(Dense(32, activation='relu'))

    # Add third layer to Model
    model.add(Dense(16, activation='relu'))

    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))

    # Create an Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                         epsilon=1e-07, amsgrad=False, name='Adam')

    # Config ANN Model
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def build_ann_model(X, y):
    """
    Build and fit ANN Model with X and y.

    Parameters:
        X (pd.DataFrame): Dataframe with selected columns.
        y (pd.DataFrame): Dataframe with target column.
    """

    # Get train, test and validation data.
    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val(X, y)

    # Initialize number of features.
    n_features = X_train.shape[DIMENSION]

    # Create ANN Sequential Model.
    model = create_ann_model(n_features)

    # Get the best batch size and epoch
    batch_size, epoch= tuning_model(model, X_train, y_train,
                                    X_test, y_test, X_val, y_val)

    # Fit the ANN model
    model.fit(X_train, y_train, epochs=epoch,
              batch_size=batch_size, verbose=1,
              validation_data=(X_val, y_val)
              )

    # Get predictions for ANN model
    predictions = get_ann_predictions(model, X_test)

    # Evaluate ANN model
    print("\nEvaluate ANN Model:")
    evaluate_model(y_test, predictions)

    # Export ANN model
    export_model(model, MODEL_PATH)


build_ann_model(final_X, final_y)
