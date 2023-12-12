"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

Authored by Li Hao Yow @ 12/12/2023 (Final Edit)
Default template provided by Dr Matthew Ellis and Dr Po Yang.

version: v1.0
"""
from typing import List
from collections import Counter

import numpy as np
import scipy as sp

N_DIMENSIONS = 10
N_NEIGHBOURS = 7

def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, second_nearest: bool) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    # Initialize empty list to store string of labels for each square
    label_list = []

    for test_fvector in test:
        # Calculate Euclidean distance between training and each test feature vector using SciPy
        dist = sp.spatial.distance.cdist(train, [test_fvector], 'euclidean').flatten()

        # By logic of k-Nearest Neighbours algorithm, find the nearest 5 training feature vectors
        nearest_neighbours = np.argsort(dist)[:N_NEIGHBOURS]

        # Create list containing the labels of nearest 5 neighbouring vectors
        nearest_labels = train_labels[nearest_neighbours]

        # Determine the label via aggregate majority in nearest_labels
        if second_nearest:
            # Return second most common label
            knn = Counter(nearest_labels).most_common(2)[-1][0]
        else:
             # Return most common label
            knn = Counter(nearest_labels).most_common(1)[0][0]
        
        # Append second most/most common label to initial list (test feature vector is therefore labeled)
        label_list.append(knn)

    # Return list of labeled test data
    return label_list


# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    # The chosen dimensionality reduction method is PCA (Principal Components Analysis)\
    print("Running PCA dimensionality reduction algorithm...")

    if 'mean' and 'components' not in model:
        # First, calculate the empirical mean of the dataset using NumPy
        empirical_mean = np.mean(data, axis=0)

        # Transpose the data array to obtain the covariance matrix
        cov_matrix = np.cov(data.T)

        # Derive the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort the columns of the eigenvector matrices in order of decreasing eigenvalue (high to low variance)
        sorted_eigenvectors = np.argsort(eigenvalues)[::-1]

        # Derive the principal components up to N_DIMENSIONS
        principal_components = eigenvectors[:, sorted_eigenvectors[:N_DIMENSIONS]]
        
        # Update the model dictionary with the calculated mean and feature components
        model['mean'] = empirical_mean
        model['components'] = principal_components
    else:
        # Set model dictionary in evaluate() to previously computed values from process_training_data()
        empirical_mean = model['mean']
        principal_components = model['components']
    
    # Perform dimensionality reduction using dot product of centred input and derived principal components
    reduced_data = np.dot(data - empirical_mean, principal_components)
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.

    # Initialize model dictionary with the given parameters
    model = {
        'fvectors_train': fvectors_train,
        'labels_train': labels_train
    }

    # Pass feature vectors to reduce_dimensions()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)

    # Update model dictionary with dimensionally-reduced feature vectors
    model['fvectors_train'] = fvectors_train_reduced

    # Convert model dictionary to list to circumvent "not JSON serializable" error
    for key, value in model.items():
        if isinstance(value, np.ndarray):
            model[key] = value.tolist()

    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        # Apply Gaussian smoothing and median filtering (preprocessing steps using SciPy)
        filtered_image = sp.ndimage.gaussian_filter(image, sigma=1.0)
        median_filtered_image = sp.ndimage.median_filter(filtered_image, size=3)

        # Add preprocessed image to fvector list
        fvectors[i, :] = median_filtered_image.ravel()

    return fvectors

def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    # print("Classifying test feature vectors (in square mode)...")
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test, False)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    # Obtain list of labeled feature vectors from classify_squares
    label_list = classify_squares(fvectors_test, model)

    # Divide list of labeled feature vectors into each board for easier classification
    # Since label_list contains 1600 feature vectors, 1600 / 64 = 25 boards which corresponds to the amount of images used for testing
    board_list = np.array_split(label_list, len(label_list) // 64)

    # Derive the second most common label via the same classify function
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])
    second_nearest_list = classify(fvectors_train, labels_train, fvectors_test, True)

    # Iterate through each board
    for indexBoard, board in enumerate(board_list):
        # Iterate through pieces of each board
        for indexPiece, piece in enumerate(board):
            counts = Counter(board)
            # If a white pawn is on the first row of the board or if a black pawn is on the last row of the board, 
            # it is likely a misclassification, therefore apply the second most common label to this feature vector.
            if (indexPiece < 8 and piece == 'p') or (indexPiece > 55 and piece == 'P'):
                label_list[64 * indexBoard + indexPiece] = second_nearest_list[64 * indexBoard + indexPiece]

            # If each team has more than 1 King, 2 Rooks, Knights, Bishops or 8 Pawns, it is highly likely to be a misclassification.
            if (counts[piece] > 1 and piece in 'Kk') or (counts[piece] > 2 and piece in 'RNBrnb') or (counts[piece] > 8 and piece in 'Pp'):
                label_list[64 * indexBoard + indexPiece] = second_nearest_list[64 * indexBoard + indexPiece]

            # If the white team has more than 16 pieces on the board, there is likely to be a misclassification.
            if (counts[piece] > 16 and piece in "KQRNBP"):
                label_list[64 * indexBoard + indexPiece] = second_nearest_list[64 * indexBoard + indexPiece]                   
            
            # If the black team has more than 16 pieces on the board, there is likely to be a misclassification.
            if (counts[piece] > 16 and piece in 'kqrnbp'):
                label_list[64 * indexBoard + indexPiece] = second_nearest_list[64 * indexBoard + indexPiece]     
    
    return label_list
