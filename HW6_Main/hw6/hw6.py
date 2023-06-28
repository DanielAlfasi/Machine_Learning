from skimage import io
import numpy as np


def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = np.array((k, 3))
    random_indexs_for_centroids = np.random.randint(0, X.shape[0], size=k)
    centroids = X[random_indexs_for_centroids]
    return np.asarray(centroids).astype(float)


def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    k = len(centroids)
    distances = np.zeros((k, X.shape[0]))

    for i, centroid in enumerate(centroids):
        distances[i, :] = (np.sum(np.absolute(
            X-centroid)**p, axis=1, keepdims=True)**(1/p)).T

    return distances


def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    centroids = get_random_centroids(X, k)

    return kmeans_with_given_centroids(X, k, p, max_iter, centroids)


def kmeans_with_given_centroids(X, k, p, max_iter, centroids):
    recomputed_centroids = np.zeros_like(centroids)

    for iteration_index in range(max_iter):
        distances_from_centroids = lp_distance(X, centroids, p)

        # Find the minimum index for each row based on the lambda expression
        classes = np.argmin(distances_from_centroids, axis=0)

        for i in range(k):
            instances_for_centroid = X[classes == i]
            recomputed_centroids[i, :] = np.mean(
                instances_for_centroid, axis=0)

        if np.array_equal(centroids, recomputed_centroids):
            print(iteration_index)
            break
        else:
            centroids = recomputed_centroids
            recomputed_centroids = np.zeros_like(centroids)

    return centroids, classes


def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None

    # Initialize the centroids list and choose one centroid randomly
    # Choose first centroid
    centroid_idx = np.random.choice(X.shape[0])
    centroids = [X[centroid_idx]]

    # Create a new array without the chosen centroid
    mask = np.arange(X.shape[0]) != centroid_idx
    X_without_centroids = X[mask]

    # Repeat until we get k centroids
    for _ in range(k - 1):

        dist_sq = np.sum(
            (X_without_centroids[:, np.newaxis] - centroids) ** 2, axis=2)
        dist_sq = dist_sq.reshape(X_without_centroids.shape[0], len(centroids))
        min_dist_sq = np.min(dist_sq, axis=1)

        # compute the probabilities
        weights = min_dist_sq / np.sum(min_dist_sq)

        # add a new centroid
        centroid_idx = np.random.choice(
            range(X_without_centroids.shape[0]), p=weights)
        centroids.append(X_without_centroids[centroid_idx])

        mask = np.arange(X_without_centroids.shape[0]) != centroid_idx
        X_without_centroids = X_without_centroids[mask]

    # Convert list of centroids to array
    centroids = np.asarray(centroids).astype(float)

    centroids, classes = kmeans_with_given_centroids(
        X, k, p, max_iter, centroids)

    return centroids, classes


def calculate_total_p_distance(data, classes, centroids, p=2):
    total_distance = 0
    for i in range(data.shape[0]):
        # Find the corresponding centroid for this data point
        centroid = centroids[classes[i]]
        # Calculate the p-distance between the data point and its centroid
        distance = lp_distance(data[i].reshape(
            1, -1), centroid.reshape(1, -1), p)
        # Add this distance to the total
        total_distance += np.sum(distance)
    return total_distance
