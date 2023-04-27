import numpy as np
import math


class Logistic(object):
    def __init__(self, d=784, reg_param=0):
        """"
        Inputs:
          - d: Number of features
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg = reg_param
        self.dim = [d+1, 1]
        self.w = np.zeros(self.dim)

    def gen_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N, d = X.shape
        X_out = np.zeros((N, d+1))
        # ================================================================ #
        # YOUR CODE HERE:
        # IMPLEMENT THE MATRIX X_out=[1, X]
        # ================================================================ #
        for i in range(N):
            for j in range(d+1):
                if j == 0:
                    X_out[i][j] = 1
                else:
                    X_out[i][j] = X[i][j-1]
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return X_out

    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w)
        N, d = X.shape

        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #

        X_out = self.gen_features(X)  # get proper dimension

        for i in range(N):
            x_i = X_out[i]
            h_w = np.dot(np.transpose(self.w), x_i)
            h_x = 1/(1+math.exp(-np.dot(np.transpose(self.w), x_i)))
            loss += (1/N)*(math.log(1+math.exp(h_w)))
            if y[i] == 1:
                loss -= h_w
            for j in range(d+1):
                grad[j] += (1/N)*(h_x-(y[i]+1)/2)*x_i[j]

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad

    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000):
        """
        Inputs:
         - X         -- numpy array of shape (N,d), features
         - y         -- numpy array of shape (N,), labels
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

    def predict(self, X):
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labelss for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros((X.shape[0], 1))
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X
        # ================================================================ #

        X_out = self.gen_features(X)  # N x d+1

        for i in range(X_out.shape[0]):
            x_i = X_out[i]
            h_x = 1/(1+math.exp(-np.dot(np.transpose(self.w), x_i)))
            if h_x >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = -1

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred
