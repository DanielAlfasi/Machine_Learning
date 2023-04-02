###### Your ID ######
# ID1: 208789172
# ID2: 318601622
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    # Performing mean normalization on input data
    meanX = np.mean(X, axis=0)
    rangeX = np.ptp(X, axis=0)
    X = (X-meanX) / rangeX
    
    # Performing mean normalization on true labels
    meanY = np.mean(y, axis=0)
    rangeY = np.ptp(y, axis=0)
    y = (y-meanY) / rangeY
    
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    ones = np.ones((len(X),1))
    tempX = X.reshape(-1,1) if X.ndim == 1 else X  # Handles 1D array case 
    X = np.concatenate((ones, tempX), axis=1)

    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.

    h_function = np.dot(X,theta) # Computes hypothesis function given input data (X) and weights (theta)
    squared_mistake_range = (h_function - y)**2 
    m = X.shape[0]  # Number of instances
    J = np.sum(squared_mistake_range) / (2*m)

    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration

    m = X.shape[0]
    for i in range(num_iters):
        # Computes hypothesis function with current theta 
        h_function = np.dot(X, theta)
        # Computes the mistake range, i.e the distance of hypothesis function (w.r.t current theta) from true labels
        mistake_range = h_function - y
        # Computes current gradient vector of the cost function
        gradient = np.dot(X.T, mistake_range) / m 
        # Updating theta using alpha and current gradient value 
        theta = theta - (alpha * gradient)
        # Computes and append the loss value of current iteration
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []

    pinv_X = np.dot(np.linalg.inv(np.dot(X.T,X)), X.T)
    pinv_theta = np.dot(pinv_X, y)

    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration

    m = X.shape[0]
    for i in range(num_iters):
        # Computes hypothesis function with current theta 
        h_function = np.dot(X, theta)
        # Computes the mistake range, i.e the distance of hypothesis function (w.r.t current theta) from true labels
        mistake_range = h_function - y
        # Computes current gradient vector of the cost function
        gradient = np.dot(X.T, mistake_range) / m 
        # Updating theta using alpha and current gradient value 
        theta = theta - (alpha * gradient)
        # Computes and append the loss value of current iteration
        J_history.append(compute_cost(X, y, theta))
        # Stop gradient descent if improvement of loss value is smaller than 1e-8
        if i > 0 and J_history[i-1] - J_history[i] < 1e-8: 
            break

    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}

    np.random.seed(42)
    rand_theta = np.random.random(size=X_train.shape[1])
    for alpha in alphas:
        current_theta = efficient_gradient_descent(X_train, y_train, rand_theta, alpha, iterations)[0]
        alpha_dict[alpha] = compute_cost(X_val, y_val, current_theta)

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    
    feature_error_dict = {}
    #np.random.seed(42)
    for i in range(5):
        np.random.seed(42)
        rand_theta = np.random.random(size = i+2) # Set theta size according to selected features
        for j in range(X_train.shape[1]):
            if j in selected_features:
                continue
            # Temporarily adding the next feature candidate
            selected_features.append(j) 
            # Applying bias trick to the relevant selected features columns 
            temp_X_train = apply_bias_trick(X_train[:,selected_features])
            temp_X_val = apply_bias_trick(X_val[:,selected_features])
            # Computes theta for current feature set and add the cost of using feature j to a dictionary
            current_theta = efficient_gradient_descent(temp_X_train, y_train, rand_theta, best_alpha, iterations)[0]
            feature_error_dict[j] = compute_cost(temp_X_val, y_val, current_theta) 
            selected_features.pop()
        
        # Getting the index of the best feature of the i'th iteration and add it to the selected features  
        min_feature_idx = min(feature_error_dict, key=feature_error_dict.get)    
        selected_features.append(min_feature_idx)
        # Clears feature error dictionary for next iteration
        feature_error_dict.clear();    
            
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()

    # Compute all possible pairs of features
    feature_pairs = [(df.columns[i], df.columns[j])
             for i in range(len(df.columns))
             for j in range(i, len(df.columns))]
    # Compute the product column of each pair and concatenate it to df_poly 
    for feature1, feature2 in feature_pairs:
        new_col = df[feature1] * df[feature2]
        new_col.name = feature1 + '*' + feature2 if feature1 != feature2 else feature1 + '^2'
        df_poly = pd.concat([df_poly, new_col], axis=1)

    return df_poly