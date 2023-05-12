import numpy as np


class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.08,
            (0, 1): 0.22,
            (1, 0): 0.22,
            (1, 1): 0.48
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.04,
            (0, 0, 1): 0.04,
            (0, 1, 0): 0.08,
            (0, 1, 1): 0.1,
            (1, 0, 0): 0.06,
            (1, 0, 1): 0.16,
            (1, 1, 0): 0.12,
            (1, 1, 1): 0.4
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for values, probablity in X_Y.items():
            if not np.isclose(probablity, X[values[0]] * Y[values[1]]):
                return True

        return False

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        for values_XYC, probablity_XYC in X_Y_C.items():
            if not np.isclose(probablity_XYC/C[values_XYC[2]], (X_C[(values_XYC[0], values_XYC[2])] * Y_C[(values_XYC[1], values_XYC[2])])
                              / C[values_XYC[2]]**2):
                return False

        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = np.log(((rate**k) * np.e**(-rate)) / factorial(k))

    return log_p


def factorial(scalar):
    return scalar * factorial(scalar - 1) if scalar > 0 else 1


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """

    # initialize an empty numpy array to store the log-likelihood values
    likelihoods = np.zeros(len(rates))

    # loop over each rate and compute the log-likelihood value for each sample
    for i, rate in enumerate(rates):
        likelihoods[i] = sum(poisson_log_pmf(sample, rate)
                             for sample in samples)
    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help
    rates_sum_of_likelihoods = {
        rate: likelihoods[i] for i, rate in enumerate(rates)}
    rate = max(rates_sum_of_likelihoods, key=rates_sum_of_likelihoods.get)
    return rate


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = sum(samples)/len(samples)

    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = (1 / (std * np.sqrt(2*np.pi))) * (np.e**((-0.5)*((x-mean)/std)**2))

    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        class_value_dataset = dataset[dataset[:, -1] == class_value]
        self.class_value = class_value
        self.dataset = dataset
        self.class_value_dataset = class_value_dataset
        self.mean_vector = np.mean(class_value_dataset[:, : -1], axis=0)
        self.covariance_matrix = np.cov(
            class_value_dataset[:, : -1], rowvar=False)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.class_value_dataset)/len(self.dataset)

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        variances = np.diagonal(self.covariance_matrix)
        likelihoods = normal_pdf(x, self.mean_vector, np.sqrt(variances))
        return np.prod(likelihoods)

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.ccd0.get_instance_posterior(
            x) > self.ccd1.get_instance_posterior(x) else 1
        return pred


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.

    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.

    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    correctly_classified = 0
    test_set_size = len(test_set)
    for instance in test_set:
        correctly_classified += 1 if map_classifier.predict(
            instance[:-1]) == instance[-1] else 0

    acc = correctly_classified/test_set_size
    return acc


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    d = len(mean)
    x_minus_mean = np.array(x) - np.array(mean)

    # Calculate the constant term
    constant_term = 1 / (((2 * np.pi) ** (d / 2)) *
                         (np.linalg.det(cov) ** 0.5))

    # Calculate the exponential term
    exponential_term = np.exp(-0.5 * x_minus_mean.T @
                              np.linalg.inv(cov) @ x_minus_mean)

    pdf = constant_term * exponential_term
    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        class_value_dataset = dataset[dataset[:, -1] == class_value]
        self.class_value = class_value
        self.dataset = dataset
        self.class_value_dataset = class_value_dataset
        self.mean_vector = np.mean(class_value_dataset[:, : -1], axis=0)
        self.covariance_matrix = np.cov(
            class_value_dataset[:, : -1], rowvar=False)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.class_value_dataset)/len(self.dataset)

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(
            x, self.mean_vector, self.covariance_matrix)

        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior


class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.ccd0.get_prior() > self.ccd1.get_prior() else 1
        return pred


class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        pred = 0 if self.ccd0.get_instance_likelihood(
            x) > self.ccd1.get_instance_likelihood(x) else 1
        return pred


# if a certain value only occurs in the test set, the probability for that value will be EPSILLON.
EPSILLON = 1e-6


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.

        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.class_value = class_value
        self.dataset = dataset
        self.class_value_dataset = dataset[dataset[:, -1] == class_value]
        self.feature_values_dict = {feature_index:
                                    np.unique(dataset[:, feature_index]) for feature_index in range(dataset[:, :-1].shape[1])}

    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = len(self.class_value_dataset)/len(self.dataset)
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """

        likelihood = 1
        n_i = len(self.class_value_dataset)
        for i, feature_i_value in enumerate(x):
            if feature_i_value in self.feature_values_dict[i]:
                V_j = len(np.unique(self.dataset[:, i]))
                n_i_j = np.count_nonzero(
                    self.class_value_dataset[:, i] == feature_i_value)
                likelihood *= (n_i_j + 1)/(V_j + n_i)
            else:
                likelihood *= EPSILLON

        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """

        pred = 0 if self.ccd0.get_instance_posterior(
            x) > self.ccd1.get_instance_posterior(x) else 1

        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        correctly_classified = 0
        test_set_size = len(test_set)
        for instance in test_set:
            correctly_classified += 1 if self.predict(
                instance[:-1]) == instance[-1] else 0

        acc = correctly_classified/test_set_size
        return acc
