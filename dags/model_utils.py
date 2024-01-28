import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential, load_model


def load_training_data(data_path: str) -> tuple:
    """
    Load the training data

    Args:
        data_path: the path to the preprocessed data

    Returns:
        training and validation data
    """
    data_dict = {}
    # Load the training, validation, and testing data in a dictionary
    for name in ["X_train", "X_val", "y_train", "y_val"]:
        # Load the pickle file
        with open(f"{data_path}/{name}.p", "rb") as file:
            data_dict[name] = pickle.load(file)

    return data_dict.values()


def load_test_data(data_path: str) -> tuple:
    """
    Load the test data

    Args:
        data_path: the path to the preprocessed data

    Returns:
        test data
    """
    data_dict = {}
    for name in ["X_test", "y_test"]:
        with open(f"{data_path}/{name}.p", "rb") as file:
            data_dict[name] = pickle.load(file)

    return data_dict.values()


def create_model()-> Sequential:
    """ 
    Creates a sequential keras model for traffic signs classification

    The model architecture is based on the VGGNet architecture

    Returns:
        A keras model
    """
    # Number of classes, i.e. number of sign types
    n_classes = 43

    # Model Archtecture
    model = Sequential()

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(5, 5),
            padding="valid",
            activation="relu",
            input_shape=(32, 32, 1),
        )
    )
    model.add(
        Conv2D(filters=64, kernel_size=(5, 5), padding="valid", activation="relu")
    )
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation="relu")
    )
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation="relu")
    )
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))

    print(model.summary())

    return model


def train_model(data_path: str, output_path: str, num_epochs) -> None:
    """
    Train a traffic signs classification model

    Steps:
        1. Load the training and validation data
        2. Shuffle the training data
        3. Create the model
        4. Compile the model
        5. Train the model
        6. Plot the training and validation accuracy and loss at each epoch and save it to /data/output
        7. Save the model to /data/output

    Args:
        data_path: the path to the preprocessed data
        output_path: the path to save the model and the training history
    """
    # Load the training and validation data
    X_train, X_val, y_train, y_val = load_training_data(data_path)

    # shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    print("data shape:", X_train.shape)

    # Create the model
    model = create_model()

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=num_epochs,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=2,
    )

    # plot the training and validation accuracy and loss 
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.savefig(f"{output_path}/training_accuracy.png")
    plt.close()

    # save the model to /data/output
    model.save(f"{output_path}/model.h5")


def model_eval(data_path: str, output_path: str) -> None:
    """
    Evaluate the model performance on the test data

    Steps:
        1. Load the test data
        2. Load the model
        3. Evaluate the model
        4. Save the confusion matrix to /data/output
        5. Plot the confusion matrix and save it to /data/output

    Args:
        data_path: the path to the preprocessed data
        output_path: the path to save the evaluation results
    """
    # Load the test data
    X_test, y_test = load_test_data(data_path)

    # Load the model
    model = load_model(f"{output_path}/model.h5")

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    y_pred = model.predict(X_test, verbose=2)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # calculate the accuracy for each class
    accuracy = cm.diagonal() / cm.sum(axis=1)
    print(accuracy)

    # plot the confusion matrix and save it to /data/output. 
    plt.figure(figsize=(30, 30))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig(f"{output_path}/confusion_matrix.png")
    plt.close()
