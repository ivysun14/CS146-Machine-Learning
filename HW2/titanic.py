"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################


class Classifier(object):
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier):

    def __init__(self):
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y):
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None:
            raise Exception("Classifier not initialized. Perform a fit first.")

        n, d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier):

    def __init__(self):
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y):
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        survived = (y == 1).sum()
        survived_percent = survived/y.shape[0]
        death_percent = 1 - survived_percent
        self.probabilities_ = {
            "Survived": survived_percent, "Death": death_percent}
        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None:
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        n = X.shape[0]

        # Generate a non-uniform random sample from np.arange(2) of size n:
        y = np.random.choice(2, n, p=[self.probabilities_[
                             'Death'], self.probabilities_['Survived']])
        y.reshape((n, ))
        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname):
    n, d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20, 15))
    nrow = 3
    ncol = 3
    for i in range(d):
        fig.add_subplot(3, 3, i+1)
        data, bins, align, labels = plot_histogram(
            X[:, i], y, Xname=Xnames[i], yname=yname, show=False)
        n, bins, patches = plt.hist(
            data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend()  # plt.legend(loc='upper left')

    plt.savefig('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show=True):
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []
    labels = []
    for target in targets:
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))),
                            int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1]  # add last bin
        align = 'left'
    else:
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(
            data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend()  # plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error = 0
    test_error = 0

    for i in range(ntrials):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i+1)
        clf.fit(X_train, y_train)

        train_pred = clf.predict(X_train)
        train_error += 1 - \
            metrics.accuracy_score(y_train, train_pred, normalize=True)

        test_pred = clf.predict(X_test)
        test_error += 1 - \
            metrics.accuracy_score(y_test, test_pred, normalize=True)

    train_error = train_error/ntrials
    test_error = test_error/ntrials
    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None):
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname:
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("../data/titanic_train.csv", header=1, predict_col=0)
    X = titanic.X
    Xnames = titanic.Xnames
    y = titanic.y
    yname = titanic.yname
    n, d = X.shape  # n = number of examples, d =  number of features

    # ========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    plot_histograms(X, y, Xnames=Xnames, yname=yname)

    # ========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    # create MajorityVote classifier, which includes all model parameters
    clf = MajorityVoteClassifier()
    clf.fit(X, y)                  # fit training data using the classifier
    # take the classifier and run it on the training data
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - \
        metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    model = DecisionTreeClassifier(criterion='entropy')
    clf = model.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - \
        metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###

    # note: uncomment out the following lines to output the Decision Tree graph

    # save the classifier -- requires GraphViz and pydot
    from io import StringIO
    import pydotplus
    from sklearn import tree
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")

    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    k = [3, 5, 7]
    for i in k:
        model = KNeighborsClassifier(n_neighbors=i)
        clf = model.fit(X, y)
        y_pred = clf.predict(X)
        train_error = 1 - \
            metrics.accuracy_score(y, y_pred, normalize=True)
        print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')

    clf = MajorityVoteClassifier()
    train_error, test_error = error(clf, X, y)
    print('\t-- training error for majority vote: %.3f' % train_error)
    print('\t-- test error for majority vote: %.3f' % test_error)

    clf = RandomClassifier()
    train_error, test_error = error(clf, X, y)
    print('\t-- training error for random: %.3f' % train_error)
    print('\t-- test error for random: %.3f' % test_error)

    clf = DecisionTreeClassifier(criterion='entropy')
    train_error, test_error = error(clf, X, y)
    print('\t-- training error for decision tree: %.3f' % train_error)
    print('\t-- test error for decision tree: %.3f' % test_error)

    clf = KNeighborsClassifier(n_neighbors=5)
    train_error, test_error = error(clf, X, y)
    print('\t-- training error for K-neighbor: %.3f' % train_error)
    print('\t-- test error for K-neighbor: %.3f' % test_error)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    avg_validation = np.zeros(25)
    count = 0
    for i in range(1, 50, 2):
        model = KNeighborsClassifier(n_neighbors=i)
        validation_score = cross_val_score(model, X, y, cv=10)
        avg_validation[count] = np.average(validation_score)
        count += 1

    validation_error = 1-avg_validation
    x_ax = np.arange(1, 50, 2)
    fig = plt.figure()
    plt.plot(x_ax, validation_error, marker='o')
    plt.xlabel('k')
    plt.ylabel('validation error')
    fig.savefig('best_k.pdf')
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    train_error_arr = np.zeros(20)
    test_error_arr = np.zeros(20)

    for i in range(1, 21):
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        train_error, test_error = error(clf, X, y)
        train_error_arr[i-1] = train_error
        test_error_arr[i-1] = test_error

    x_ax = np.arange(1, 21)
    fig = plt.figure()
    plt.plot(x_ax, train_error_arr, label="train", marker='o')
    plt.plot(x_ax, test_error_arr, label="test", marker='o')
    plt.xlabel('tree depth')
    plt.ylabel('error')
    plt.legend()
    fig.savefig('tree_depth.pdf')
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    train_error_tree = np.zeros(10)
    test_error_tree = np.zeros(10)
    train_error_k = np.zeros(10)
    test_error_k = np.zeros(10)

    size = np.arange(0.10, 1.10, 0.10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1)  # initial split into 90/10

    for i in range(len(size)):
        if 1-size[i] == 0:  # use all 90% as training
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)
            train_error_tree[i] = 1 - \
                metrics.accuracy_score(y_train, train_pred, normalize=True)
            test_pred = clf.predict(X_test)
            test_error_tree[i] = 1 - \
                metrics.accuracy_score(y_test, test_pred, normalize=True)

            clf = KNeighborsClassifier(n_neighbors=7)
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)
            train_error_k[i] = 1 - \
                metrics.accuracy_score(y_train, train_pred, normalize=True)
            test_pred = clf.predict(X_test)
            test_error_k[i] = 1 - \
                metrics.accuracy_score(y_test, test_pred, normalize=True)
        else:
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=7)
            train_error, not_used = error(
                clf, X_train, y_train, test_size=1-size[i])
            train_error_tree[i] = train_error
            test_pred = clf.predict(X_test)
            test_error_tree[i] = 1 - \
                metrics.accuracy_score(y_test, test_pred, normalize=True)

            clf = KNeighborsClassifier(n_neighbors=7)
            train_error, not_used = error(
                clf, X_train, y_train, test_size=1-size[i])
            train_error_k[i] = train_error
            test_pred = clf.predict(X_test)
            test_error_k[i] = 1 - \
                metrics.accuracy_score(y_test, test_pred, normalize=True)

    fig = plt.figure()
    plt.plot(size, train_error_tree, label="train_tree", marker='o')
    plt.plot(size, test_error_tree, label="test_tree", marker='o')
    plt.plot(size, train_error_k, label="train_k", marker='o')
    plt.plot(size, test_error_k, label="test_k", marker='o')
    plt.xlabel('Train sample size')
    plt.ylabel('error')
    plt.legend()
    fig.savefig('learning_curve.pdf')

    ### ========== TODO : END ========== ###

    print('Done')


if __name__ == "__main__":
    main()
