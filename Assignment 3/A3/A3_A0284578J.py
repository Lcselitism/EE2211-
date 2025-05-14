import numpy as np


# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0284578J(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here
    # initalizations 
    a = 2.6
    b = 0.6
    c = 1.0
    d = 3.0

    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)
    
    for i in range(num_iters):
        #f1(𝑎) = 𝑎^4, f1'(a) = 4a^3
        a -= learning_rate * (4*a**3)
        a_out[i] = a
        f1_out[i] = a**4
        
        #f2(b) = sin^3(b), f2'(b) = 3sin(b)^2(cos(b))
        b -= learning_rate * (3 * np.sin(b)**2 * np.cos(b))
        b_out[i] = b
        f2_out[i] = np.sin(b)**3

        #f3(c,d) = c^5*d^3 + d^2sin(d), f3'(c) = 5c^4*d^3, f3'(d) = c**5 * 3*d**2 + d**2*np.cos(d) + np.sin(d)*2*d
        grad_c = 5 * c**4 * d**3
        grad_d = 3 * c**5 * d**2 + 2*d*np.sin(d) + d**2*np.cos(d)
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
        c_out[i] = c
        d_out[i] = d
        f3_out[i] = c**5 * d**3 + d**2*np.sin(d)


    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out


