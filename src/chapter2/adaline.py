import numpy as np

class AdalineGD(object):
    """Adaptive Liner Neuron classifier

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset

    Attributes
    ------------
    w_ : ld-array
        weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch
    """

    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, Y):
        """Fit training data

        Parameters
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target value

        Returns
        -----------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        # for i in range(self.n_iter):
        #     output = self.net_input(X)


    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


