import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0284578J(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # your code goes here
    iris_dataset = load_iris()

    #Get training set, test set
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size =0.2, random_state = N)

    #Construct target output for training and test set using one-hot encoding
    ytrain_onehot = []
    ytest_onehot = []
    
    for y in y_train:
        label = [0, 0, 0]
        label[y] = 1
        ytrain_onehot.append(label)
    for y in y_test:
        label = [0, 0, 0]
        label[y] = 1
        ytest_onehot.append(label)
        
    Ytr = np.array(ytrain_onehot)  # Convert list to NumPy array
    Yts = np.array(ytest_onehot)  # Convert list to NumPy array
    
    #List of training and test matrices
    Ptrain_list = []
    Ptest_list = []
    w_list = []
    error_train_array = []
    error_test_array = []

    # order 1 to 7
    lambda_value = 0.001
    for order in range(1, 8):
        
        poly = PolynomialFeatures(order)
        P_train = poly.fit_transform(X_train)
        P_test = poly.transform(X_test)
        Ptrain_list.append(P_train)
        Ptest_list.append(P_test)

        # Compute w (P.shape gives a tuple representing dimensions of P, P.shape[0] access first element, which corresponds to num of rows
        if (P_train.shape[0] > P_train.shape[1]): # n_rows > n_cols, (m > d) over-determined, left inverse, Primal Form
            w = np.linalg.inv(P_train.T@P_train + lambda_value * np.identity(P_train.shape[1]))@P_train.T@Ytr
            
            
        else: # n_rows < n_cols, (m < d) under-determined, right inverse, Dual Form (in case of square, inverse = left inverse = right inverse)
            w = P_train.T@np.linalg.inv(P_train@P_train.T + lambda_value * np.identity(P_train.shape[0]))@Ytr

        w_list.append(w)

        # Predict and measure number of correct classification
        ytrain_predict_onehot = P_train@w
        ytest_predict_onehot = P_test@w
        error_train = 0
        error_test = 0

        # Loop through training predictions
        for i in range(len(ytrain_predict_onehot)): # num of rows
            maxidx = np.argmax(ytrain_predict_onehot[i])
            # for error train array
            if (maxidx != y_train[i]):
                error_train += 1
                
        # Loop through test predictions 
        for i in range(len(ytest_predict_onehot)): # num of rows
            maxidx = np.argmax(ytest_predict_onehot[i])
            # for error train array
            if (maxidx != y_test[i]):
                error_test += 1
           
        error_train_array.append(error_train)
        error_test_array.append(error_test)

    
    error_train_array = np.array(error_train_array) # Convert list to NumPy array
    error_test_array = np.array(error_test_array) # Convert list to NumPy array


    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array



