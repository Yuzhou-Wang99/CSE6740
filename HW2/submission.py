from typing import Tuple, Union, Callable
import numpy as np


def initialize_parameters(
    X: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    """
    
    m,n = X.shape #m is the number of sampels and n is the dimension of each data
    
    #initialize MU
    index = np.random.choice(m, k, replace=False)
    MU = X[index]

    #initialize SIGMA
    #emp_cov = np.cov(X, rowvar=False)
    #SIGMA = np.array([emp_cov]*k)
    SIGMA = compute_sigma(X,MU)

    #initailize PI
    PI = np.full(k,float(1/k))
    return MU,SIGMA,PI
    
    raise NotImplementedError()
    
    


def compute_sigma(X: np.ndarray, MU: np.ndarray) -> np.ndarray:
    """
    Calculate covariance matrix, based in given X and MU values

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """

    k, n = MU.shape
    m = X.shape[0]
    
    SIGMA = np.zeros((k, n, n))

    for i in range(k):
        # Compute the differences from the mean
        Diff = X-MU[i] # subtract the ith mean from each row, i.e. from each data point. 
        SIGMA[i] = (Diff.T @ Diff)/m
    return SIGMA
    raise NotImplementedError()


def prob(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Union[float, np.ndarray]:
    """Calculate the probability of x (a single
    data point or an array of data points,
    you have to take care both cases) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] or numpy.ndarray[numpy.ndarray[float]]
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float or numpy.ndarray[float]
    """
    
    n = mu.shape[0]
    
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    
    norm = 1 / (np.sqrt((2 * np.pi) ** n * sigma_det))
    
    if x.ndim == 1:
        diff = x - mu
        exponent = -0.5 * np.dot(np.dot(diff.T, sigma_inv), diff)
        return norm * np.exp(exponent)
    else:
        probability = np.zeros(x.shape[0])
        for i, x_i in enumerate(x):
            diff = x_i - mu
            exponent = -0.5 * np.dot(np.dot(diff.T, sigma_inv), diff)
            probability[i] = norm * np.exp(exponent)
        return probability
    
    raise NotImplementedError()


def E_step(
    X: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray, PI: np.ndarray, k: int
) -> np.ndarray:
    """
    E-step - Expectation
    Calculate responsibility for each
    of the data points, for the given
    MU, SIGMA and PI.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    m,n = X.shape

    r = np.zeros((k,m))

    #compute the prob of each data point from the ith Guassian
    for i in range(k):
        r[i] = PI[i] * prob(X, MU[i], SIGMA[i])
    
    #normalize
    r/= r.sum(axis=0, keepdims=True)
    return r
    
    raise NotImplementedError()


def M_step(
    X: np.ndarray, r: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int

    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    """
    
    m,n = X.shape
    new_MU = np.zeros((k, n))
    new_SIGMA = np.zeros((k, n, n))
    new_PI = np.zeros(k)

    for i in range(k):
        r_row_sum = r[i].sum()
        new_MU[i] = np.dot(r[i], X) / r_row_sum
        diff = X - new_MU[i]
        new_SIGMA[i] = np.dot(r[i] * diff.T, diff) / r_row_sum
        new_PI[i] = r_row_sum / m

    return new_MU, new_SIGMA, new_PI
    
    raise NotImplementedError()


def likelihood(
    X: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray, PI: np.ndarray, k: int
) -> float:
    """Calculate a log likelihood of the
    trained model based on the following
    formula for posterior probability:

    log(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), log(sum((k=1 to K),
                                      mixing_k * N(x_n | mean_k,stdev_k))))

    Make sure you are using natural log, instead of log base 2 or base 10.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    log_likelihood = float
    """
    m = X.shape[0]
    log_likelihood = 0
    
    for i in range(m):
        likelihood_sum = 0
        for j in range(k):
            likelihood_sum += PI[j] * prob(X[i], MU[j], SIGMA[j])
        log_likelihood += np.log(likelihood_sum)
    
    return log_likelihood
    
    raise NotImplementedError()


def default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (
        abs(prev_likelihood) * 0.9 < abs(new_likelihood) < abs(prev_likelihood) * 1.1
    )

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


def train_model(
    X: np.ndarray,
    k: int,
    convergence_function: Callable = default_convergence,
    initial_values: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example
    above.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    
    if initial_values == None:
        MU, SIGMA, PI = initialize_parameters(X,k)
    else:
        MU, SIGMA, PI = initial_values
    
    converged = False
    conv_ctr = 0
    
    #prev_likelihood = likelihood(X,MU,SIGMA,PI,k)
    prev_likelihood =-np.inf


    while not converged:
        r = E_step(X,MU,SIGMA,PI,k)
        MU, SIGMA, PI = M_step(X,r,k)
        
        new_likelihood = likelihood(X,MU,SIGMA,PI,k)
        if not np.isfinite(new_likelihood):
            print("Likelihood became NaN or -inf. Stopping early.")
            break

        conv_ctr, converged = default_convergence(prev_likelihood,new_likelihood,conv_ctr)
        prev_likelihood = new_likelihood
    
    return MU, SIGMA, PI, r
    raise NotImplementedError()
