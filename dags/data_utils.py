import os
import pickle
import urllib.request
import zipfile

import numpy as np
import skimage.morphology as morp
from skimage import color
from skimage.filters import rank


def download_and_unzip(url, zip_file_path, extract_to_path):
    urllib.request.urlretrieve(url, zip_file_path)
    print("Downloaded zip file from {} to {}".format(url, zip_file_path))
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)
    print("Unzipped file from {} to {}".format(zip_file_path, extract_to_path))
    # delete the downloaded zip file
    os.remove(zip_file_path)


def load_data(file_path):
    """
    Loads the datasets having a pickle data format.

    Args:
        file_path: name of the file (or the filepath)

    Returns:
        X: the feature matrix
        y: class labels
    """

    with open(file_path, mode="rb") as f:
        data = pickle.load(f)

    X, y = data["features"] / 255.0, data["labels"]

    return X, y


def local_histo_equalize(image):
    """
    Apply local histogram equalization to grayscale images.
        Parameters:
            image: A grayscale image.
    """
    kernel = morp.disk(30)
    # equalize with the kernel
    img_local = rank.equalize(image, footprint=kernel)
    return img_local


def image_preprocessing(image):
    """
    Preprocess the image.

    Convert the images to grayscale, apply local histogram equalization and
    reshape the images to the desired shape.

    Args:
        image: a numpy array representing the image

    Returns:
        image: normalized image
    """

    # RGB to grayscale
    image = color.rgb2gray(image)
    image_size = image[0].shape

    # Local histogram equalization
    images = list(map(local_histo_equalize, image))
    print("Shape of images: ", images[0].shape)
    # Reshape images
    images = np.array(images).reshape((-1, image_size[0], image_size[1], 1))
    print("Shape of images after reshaping: ", images[0].shape)
    return images


def data_preparation(input_data_path: str, output_data_path: str) -> None:
    """
    Prepare the data for training and testing.

    1. Load the raw data
    2. Preprocess the data
    3. Save the preprocessed data

    Args:
        input_data_path: the path to the raw data
        output_data_path: the path to save the preprocessed data
    """
    training = f"{input_data_path}/train.p"
    validation = f"{input_data_path}/valid.p"
    testing = f"{input_data_path}/test.p"

    X_train, y_train = load_data(training)
    X_test, y_test = load_data(testing)
    X_val, y_val = load_data(validation)

    # Image size
    image_size = X_train[0].shape

    # Number of training, validation, and test examples
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    # Number of unique traffic symbols
    n_classes = len(np.unique(y_train))

    print("Image size =", image_size)
    print("Number of training examples: ", n_train)
    print("Number of validation examples: ", n_val)
    print("Number of testing examples: ", n_test)
    print("Number of classes =", n_classes)

    # Preprocess the data
    X_train = image_preprocessing(X_train)
    X_val = image_preprocessing(X_val)
    X_test = image_preprocessing(X_test)

    print("Preprocessing done!")
    print("Saving the preprocessed data to the output path")
    # create the output path if it doesn't exist
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    # save the preprocessed data to the output path
    data_dict = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    for name, data in data_dict.items():
        print(f"Saving {name}...")
        with open(f"{output_data_path}/{name}.p", "wb") as f:
            pickle.dump(data, f)
