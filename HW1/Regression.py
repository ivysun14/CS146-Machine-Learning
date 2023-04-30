import numpy as np


class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg = reg_param
        self.dim = [m+1, 1]
        self.w = np.zeros(self.dim)

    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns:
         - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N, d = X.shape
        m = self.m
        X_out = np.zeros((N, m+1))
        if m == 1:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            # ================================================================ #
            for i in range(N):
                X_out[i][0] = 1
                X_out[i][1] = X[i][0]
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            for i in range(N):
                X_out[i][0] = 1
                for j in range(1, m+1):
                    X_out[i][j] = (X[i][0])**j
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return X_out

    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w)
        m = self.m
        N, d = X.shape

        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the linear regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        #
        # ================================================================ #

        X_out = self.gen_poly_features(X)

        for i in range(N):
            h_x = np.dot(np.transpose(self.w), X_out[i])
            # using mean squared error as loss
            loss += (1/N)*((h_x-y[i])**2)
            for j in range(d+1):
                grad[j] += 2*(1/N)*(h_x-y[i])*X_out[i][j]
                if i == N-1 & j != 0:  # after gradient be calculated for last sample
                    grad[j] += self.reg*self.w[j]

        loss += (self.reg/2) * \
            ((np.linalg.norm(self.w)-(self.w[0]*self.w[0]))**2)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grad

    def train_LR(self, X, y, eta=1e-3, batch_size=30, num_iters=1000):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations

        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N, d = X.shape
        for t in np.arange(num_iters):
            X_batch = None
            y_batch = None
            # ================================================================ #
            # YOUR CODE HERE:
            # Sample batch_size elements from the training data for use in gradient descent.
            # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
            # The indices should be randomly generated to reduce correlations in the dataset.
            # Use np.random.choice.  It is better to user WITHOUT replacement.
            # ================================================================ #

            X_batch = np.zeros((batch_size, d))
            y_batch = np.zeros((batch_size, ))
            to_draw = np.arange(0, N)

            for i in range(batch_size):
                index = np.random.choice(to_draw, replace=False)
                X_batch[i] = X[index]
                y_batch[i] = y[index]

            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss = 0.0
            grad = np.zeros_like(self.w)
            # ================================================================ #
            # YOUR CODE HERE:
            # evaluate loss and gradient for batch data
            # save loss as loss and gradient as grad
            # update the weights self.w
            # ================================================================ #

            loss, grad = self.loss_and_grad(X_batch, y_batch)
            self.w = np.subtract(self.w, eta*(1/batch_size)*grad)

            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss_history.append(loss)
        return loss_history, self.w

    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        m = self.m
        N, d = X.shape
        # ================================================================ #
        # YOUR CODE HERE:
        # obtain the optimal weights from the closed form solution
        # ================================================================ #

        X_out = self.gen_poly_features(X)

        one = np.linalg.inv(np.matmul(np.transpose(X_out), X_out))
        two = np.matmul(np.transpose(X_out), y)
        self.w = np.matmul(one, two)

        prediction = np.matmul(X_out, self.w)
        three = np.subtract(prediction, y)
        loss = (1/N)*np.dot(np.transpose(three), three)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, self.w

    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m

        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE TARGETS OF X
        # ================================================================ #

        X_out = self.gen_poly_features(X)

        for i in range(X.shape[0]):
            y_pred[i] = np.dot(np.transpose(self.w), X_out[i])

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return y_pred
