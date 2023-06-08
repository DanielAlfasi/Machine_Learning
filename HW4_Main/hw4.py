import numpy as np


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        X = self.apply_bias_trick(X)
        self.theta = np.random.rand(X.shape[1])

        for iteration in range(self.n_iter):
            h_function = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot((h_function - y), X)

            self.theta = self.theta - (self.eta * gradient)
            self.Js.append(self.compute_cost(X, y))
            self.thetas.append(self.theta)

            if iteration > 0 and self.Js[iteration - 1] - self.Js[iteration] < self.eps:
                break

    def compute_cost(self, X, y):

        h_function = self.sigmoid(np.dot(X, self.theta))
        first_addend = np.dot(-y, np.log(h_function))
        second_addend = np.dot(
            (1-y), np.log(1-h_function))
        J_function = (first_addend - second_addend) / X.shape[0]

        return J_function

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def apply_bias_trick(self, X):
        """
        Applies the bias trick to the input data.

        Input:
        - X: Input data (m instances over n features).

        Returns:
        - X: Input data with an additional column of ones in the
            zeroth position (m instances over n+1 features).
        """

        ones = np.ones((len(X), 1))
        tempX = X.reshape(-1, 1) if X.ndim == 1 else X  # Handles 1D array case
        X = np.concatenate((ones, tempX), axis=1)

        return X

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """

        X = self.apply_bias_trick(X)
        h_function = self.sigmoid(np.dot(X, self.theta))
        preds = np.where(h_function < 0.5, 0, 1)

        return preds


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = 0
    # set random seed
    np.random.seed(random_state)

    data_with_labels = np.concatenate((X, y.reshape(-1, 1)), axis=1)

    np.random.shuffle(data_with_labels)
    folds_list = np.array_split(data_with_labels, folds)

    folds_X = [fold[:, : -1] for fold in folds_list]
    folds_Y = [fold[:, -1] for fold in folds_list]

    for i in range(folds):
        training_set_X = np.concatenate(
            [fold for j, fold in enumerate(folds_X) if i != j])
        training_set_Y = np.concatenate(
            [fold for j, fold in enumerate(folds_Y) if i != j])
        algo.fit(training_set_X, training_set_Y)

        validation_set_X = folds_X[i]
        validation_set_Y = folds_Y[i]
        predictions = algo.predict(validation_set_X)
        accuracy = np.sum(predictions == validation_set_Y) / \
            len(validation_set_Y)
        cv_accuracy += accuracy
    cv_accuracy = cv_accuracy / folds
    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.
    """
    p = (1 / (sigma * np.sqrt(2*np.pi))) * \
        (np.e**((-0.5)*((data-mu)/sigma)**2))

    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        num_samples, num_features = data.shape

        indices = np.random.choice(num_samples, self.k, replace=False)
        self.mus = data[indices]
        self.weights = np.full(self.k, 1 / self.k)
        self.sigmas = np.full((self.k, num_features), data.std(axis=0))

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """

        self.responsibilities = np.zeros((data.shape[0], self.k))

        for i in range(self.k):
            self.responsibilities[:, i] = self.weights[i] * \
                norm_pdf(data, self.mus[i, :], self.sigmas[i, :]).flatten()

        self.responsibilities = self.responsibilities / \
            self.responsibilities.sum(axis=1, keepdims=1)

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        number_of_instances = data.shape[0]

        for gaussian_index in range(self.k):
            self.weights[gaussian_index] = np.sum(
                self.responsibilities[:, gaussian_index])/number_of_instances
            self.mus[gaussian_index, :] = np.dot(
                self.responsibilities[:, gaussian_index], data) / (self.weights[gaussian_index] * number_of_instances)
            diff = data - self.mus[gaussian_index, :]
            self.sigmas[gaussian_index, :] = np.sqrt(
                np.dot(self.responsibilities[:, gaussian_index], diff**2) / (self.weights[gaussian_index] * number_of_instances))

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)

        prev_likelihood = float('-inf')

        for i in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)

            likelihood = np.sum(
                self.weights * norm_pdf(data, self.mus[:, np.newaxis], self.sigmas[:, np.newaxis]), axis=0)
            likelihood = np.sum((-1) * np.log(likelihood))

            if np.abs(prev_likelihood - likelihood) < self.eps:
                break

            prev_likelihood = likelihood

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.
    """
    pdf = 0

    for weight, mu, sigma in zip(weights, mus, sigmas):
        pdf += weight * norm_pdf(data, mu, sigma)

    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.classes = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """

        n_examples = X.shape[0]

        # Initialize dictionary
        self.classes = {}

        # Identify unique classes and their counts
        unique_classes, counts = np.unique(y, return_counts=True)

        for cls, count in zip(unique_classes, counts):
            # Compute prior probability
            prior_prob = count / n_examples
            # Extract corresponding rows from X
            X_cls = X[y == cls, :]
            X_cls = X_cls.reshape(-1, 1)
            # EM
            em = EM(self.k, random_state=self.random_state)
            # EM learning phase
            em.fit(X_cls)
            # EM paramters
            weights, mus, sigmas = em.get_dist_params()
            # Add to dictionary
            self.classes[cls] = [prior_prob, weights, mus, sigmas]

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        n_examples = X.shape[0]
        cls_posterior_dict = {}

        # Compute the posterior for each class
        for cls in self.classes:
            gmms_of_features_vector = gmm_pdf(X, self.classes[cls][1],
                                              self.classes[cls][2], self.classes[cls][3])

            # Calculate the product along each row to get the joint likelihood for each instance
            joint_likelihood = np.prod(gmms_of_features_vector, axis=1)

            # Multiply by the prior to get the posterior
            cls_posterior_dict[cls] = joint_likelihood * self.classes[cls][0]

        # Initialize an array to hold the predictions
        preds = np.zeros(n_examples)

        # For each instance, find the class with the maximum posterior
        for i in range(n_examples):
            max_cls = None
            max_posterior = 0

            for cls in self.classes:
                if cls_posterior_dict[cls][i] > max_posterior:
                    max_cls = cls
                    max_posterior = cls_posterior_dict[cls][i]

            # Assign the class with the maximum posterior to the i-th instance
            preds[i] = max_cls

        return preds


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    '''
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    lor_model = LogisticRegressionGD(best_eta, eps=best_eps)
    lor_model.fit(x_train, y_train)

    bayes_model = NaiveBayesGaussian(k)
    bayes_model.fit(x_train, y_train)

    lor_train_acc = accuracy_score(lor_model.predict(x_train), y_train)
    lor_test_acc = accuracy_score(lor_model.predict(x_test), y_test)
    bayes_train_acc = accuracy_score(bayes_model.predict(x_train), y_train)
    bayes_test_acc = accuracy_score(bayes_model.predict(x_test), y_test)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy of predictions.

    Parameters
    ----------
    y_true : array-like, shape = [n_examples]
        True class labels.
    y_pred : array-like, shape = [n_examples]
        Predicted class labels by the model.

    Returns
    -------
    accuracy : float
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'dataset_a_features': dataset_a_features,
            'dataset_a_labels': dataset_a_labels,
            'dataset_b_features': dataset_b_features,
            'dataset_b_labels': dataset_b_labels
            }
