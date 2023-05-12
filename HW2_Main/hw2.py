import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
             0.25: 1.32,
             0.1: 2.71,
             0.05: 3.84,
             0.0001: 100000},
             2: {0.5: 1.39,
             0.25: 2.77,
             0.1: 4.60,
             0.05: 5.99,
             0.0001: 100000},
             3: {0.5: 2.37,
             0.25: 4.11,
             0.1: 6.25,
             0.05: 7.82,
             0.0001: 100000},
             4: {0.5: 3.36,
             0.25: 5.38,
             0.1: 7.78,
             0.05: 9.49,
             0.0001: 100000},
             5: {0.5: 4.35,
             0.25: 6.63,
             0.1: 9.24,
             0.05: 11.07,
             0.0001: 100000},
             6: {0.5: 5.35,
             0.25: 7.84,
             0.1: 10.64,
             0.05: 12.59,
             0.0001: 100000},
             7: {0.5: 6.35,
             0.25: 9.04,
             0.1: 12.01,
             0.05: 14.07,
             0.0001: 100000},
             8: {0.5: 7.34,
             0.25: 10.22,
             0.1: 13.36,
             0.05: 15.51,
             0.0001: 100000},
             9: {0.5: 8.34,
             0.25: 11.39,
             0.1: 14.68,
             0.05: 16.92,
             0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0

    label_column = data.shape[1] - 1
    total_number_of_instances = data.shape[0]

    class_value, amount_of_type = np.unique(
        data[:, label_column], return_counts=True)

    sum_of_class_values_dict = dict(zip(class_value, amount_of_type))

    sum_of_all_class_values_squared = sum(
        [(sum_of_class_values_dict[class_value]/total_number_of_instances)**2 for class_value in sum_of_class_values_dict])

    gini = 1 - sum_of_all_class_values_squared
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    label_column = data.shape[1] - 1
    total_number_of_instances = data.shape[0]

    class_value, amount_of_type = np.unique(
        data[:, label_column], return_counts=True)
    sum_of_class_values_dict = dict(zip(class_value, amount_of_type))

    sum_of_all_class_values_squared = sum(
        [(sum_of_class_values_dict[class_value]/total_number_of_instances)*(np.log2(sum_of_class_values_dict[class_value]/total_number_of_instances)) for class_value in sum_of_class_values_dict])
    entropy = sum_of_all_class_values_squared * (-1)
    return entropy


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting
              according to the feature values.
    """

    goodness = 0
    groups = {}  # groups[feature_value] = data_subset

    impurity_before_splitting = impurity_func(data)

    total_number_of_instances = data.shape[0]

    feature_values, amount_of_instances = np.unique(
        data[:, feature], return_counts=True)

    values_of_feature_dict = dict(
        zip(feature_values, amount_of_instances))

    groups = {feature_value: data[data[:, feature] == feature_value]
              for feature_value in feature_values}

    if gain_ratio:
        information_gain = goodness_of_split(data, feature, calc_entropy)[0]

        split_information = (-1) * sum([(values_of_feature_dict[feature_value] / total_number_of_instances) * np.log2(
            values_of_feature_dict[feature_value] / total_number_of_instances) for feature_value in feature_values])
        if split_information == 0:
            goodness = 0
        else:
            goodness = information_gain/split_information
    else:
        sum_of_split = sum([(values_of_feature_dict[feature_value] /
                             total_number_of_instances)*impurity_func(groups[feature_value]) for feature_value in feature_values])
        goodness = impurity_before_splitting - sum_of_split

    return goodness, groups


class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        label_column = self.data.shape[1] - 1

        class_value, amount_of_type = np.unique(
            self.data[:, label_column], return_counts=True)

        class_values_dict = dict(zip(class_value, amount_of_type))

        pred = max(class_values_dict,  key=class_values_dict.get)

        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func):
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        if self.depth == self.max_depth:
            return

        goodness_of_feature_dict = dict()

        for feature_index in range(self.data.shape[1] - 1):
            goodness_of_feature_dict[feature_index] = goodness_of_split(
                self.data, feature_index, impurity_func, self.gain_ratio)

        best_feature_index = max(
            goodness_of_feature_dict,  key=lambda x: goodness_of_feature_dict[x][0])

        best_feature_data_group = goodness_of_feature_dict[best_feature_index][1]

        self.feature = best_feature_index

        # if len(best_feature_data_group) <= 1:
        #     self.terminal = True
        #     return

        for feature_value in best_feature_data_group:
            child = DecisionNode(
                best_feature_data_group[feature_value], -1, self.depth + 1, self.chi, self.max_depth, self.gain_ratio)
            self.add_child(child, feature_value)


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """

    root = DecisionNode(data, -1, 0, chi, max_depth, gain_ratio)

    queue = []
    queue.append(root)

    while queue:
        curr_node = queue.pop(0)
        # Does calc entropy with gain ratio = true gives me impurity = 0 if the node is pure
        if impurity(curr_node.data) == 0 or curr_node.depth >= curr_node.max_depth:
            curr_node.terminal = True
        else:
            curr_node.split(impurity)
            if chi_test(curr_node):
                for child in curr_node.children:
                    queue.append(child)
            else:
                curr_node.terminal = True
    return root


def chi_test(decision_node):
    """
    The function returns true if the given node has passed the chi square test (This split isnt random)

    Input: node of a decision tree

    Output: True - the node has passed the chi square test otherwise false

    """
    chi_value = calculate_chi(decision_node)
    degree_of_freedom = len(decision_node.children) - 1
    alpha_risk = decision_node.chi

    if alpha_risk == 1 or degree_of_freedom < 1:
        return True
    return chi_table[degree_of_freedom][alpha_risk] < chi_value


def calculate_chi(decision_node):
    """
    Calculate the chi square value for the given node

    Input: a desicion tree node

    Output: the chi square value for that node
    """
    chi_value = 0
    label_column = -1

    number_of_instances = decision_node.data.shape[0]
    label, amount_of_label = np.unique(
        decision_node.data[:, label_column], return_counts=True)
    labels_dict = dict(zip(label, amount_of_label))

    labels_probability_dict = {
        label: (labels_dict[label] / number_of_instances) for label in labels_dict}

    for child in decision_node.children:
        Df = child.data.shape[0]
        label_in_child, amount_of_label_in_child = np.unique(
            child.data[:, label_column], return_counts=True)
        child_labels_dict = dict(zip(label_in_child, amount_of_label_in_child))
        for label in labels_probability_dict:
            E = Df * labels_probability_dict[label]

            # If label does not exist in the child data subset, then append E to chi value (since E^2 / E = E)
            chi_value += (child_labels_dict[label] - E)**2 / \
                E if label in child_labels_dict else E

    return chi_value


def predict(root, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """

    pred = root.pred
    child = None

    if root.terminal == True:
        return pred

    feature_value_of_instance = instance[root.feature]

    for i in range(len(root.children)):
        if root.children_values[i] == feature_value_of_instance:
            child = root.children[i]
            break

    if child == None:
        return pred

    return predict(child, instance)


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    sum_of_success = 0
    data_set_size = dataset.shape[0]

    for instance in dataset:
        true_label = instance[-1]
        prediction = predict(node, instance)
        sum_of_success += 1 if true_label == prediction else 0

    accuracy = sum_of_success/data_set_size
    return accuracy


def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        root = build_tree(X_train, calc_entropy, True, 1, max_depth)
        training.append(calc_accuracy(root, X_train))
        testing.append(calc_accuracy(root, X_test))
    return training, testing


def chi_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []
    for alpha_risk in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        root = build_tree(X_train, calc_entropy, True, alpha_risk)
        chi_training_acc.append(calc_accuracy(root, X_train))
        chi_testing_acc.append(calc_accuracy(root, X_test))
        depth.append(find_tree_depth(root))
    return chi_training_acc, chi_testing_acc, depth


def find_tree_depth(node):
    """
    find the depth of a tree

    Input:
    - node: a root of a desicion tree

    Output: the depth of the tree
    """
    if node.terminal == True:
        return node.depth

    return max([find_tree_depth(children) for children in node.children])


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of nodes in the tree.
    """
    n_nodes = 1
    if node.terminal == True:
        return 1

    for child in node.children:
        n_nodes += count_nodes(child)

    return n_nodes
