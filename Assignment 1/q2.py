import math
import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
import random
import time
import sys
f = open('output.txt','w')
sys.stdout = f
test_size = 0.2
number_of_splits = 10


# data_train_ = None
# data_test_ = None
# labels_train_ = None
# labels_test_ = None

class Naive_Bayes_Classifier:
    """
        Class for a NaiveBayes Classifier
    """

    def __init__(self, X, c_vars=None, Y=None, alpha=0.0, laplace=False, flag=False):
        """Constructor

        Args:
            X (np.array): features or features+labels if Y is None
            c_vars (list, optional): [description]. Defaults to [].
            Y (np.array, optional): labels. Defaults to None.
        """
        if c_vars is None:
            c_vars = []
        self.alpha = alpha
        self.laplace = laplace
        self.d = X.shape[1] if Y is None else X.shape[1] + 1
        self.Y = X[:, -1] if Y is None else Y
        self.X = X[:, :self.d - 1] if Y is None else X
        self.c_vars = c_vars

        if laplace:
            self.ret_data()
        else:
            self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(self.X, self.Y,
                                                                                                    test_size=test_size,
                                                                                                    random_state=42 + random.randint(
                                                                                                        0, 100))
            self.set_data()
        self.labels_train = np.array(self.labels_train).reshape(-1, 1)
        self.labels_test = np.array(self.labels_test).reshape(-1, 1)
        self.values = []
        i = 0
        while i < self.d - 1:
            self.values.append(list(set(list(X[:, i]))))
            i = i + 1

    def set_alpha(self, alpha=0.0):
        self.alpha = alpha

    def set_laplace(self, val=False):
        self.laplace = val

    def set_C_vars(self, c_vars=None):
        if c_vars is None:
            c_vars = []

    def ret_data(self):
        self.data_train, self.data_test, self.labels_train, self.labels_test = data_train_, data_test_, labels_train_, labels_test_

    def set_data(self):
        global data_train_
        global data_test_
        global labels_test_
        global labels_train_
        data_train_, data_test_, labels_train_, labels_test_ = self.data_train, self.data_test, self.labels_train, self.labels_test

    def train_and_print_acc(self):
        """
            Function to train across k-folds and print average accuracy
        """
        i = 0
        K_Fold_Cross_Validation = KFold(n_splits=number_of_splits)
        accuracies = []
        if self.laplace is False:
            print('\
                   ////////////////////////////////\n\
                   /////////    SCORES    /////////\n\
                   ////////////////////////////////\n\
                   ')
        if self.laplace is False:
            for train_index, test_index in K_Fold_Cross_Validation.split(self.data_train):
                X_train, X_test = self.data_train[train_index], self.data_train[test_index]
                y_train, y_test = self.labels_train[train_index], self.labels_train[test_index]
                self.train_model(X_train, y_train)
                accuracy = self.test_model(X_test, y_test)
                i = i + 1
                print(f"\t   Accuracy at the {i} split  = ", accuracy)
                accuracies.append(accuracy)
            average_accuracy = sum(accuracies)
            average_accuracy /= len(accuracies)
            print("\nAverage train accuracy  = ", average_accuracy)

        self.train_model(self.data_train, self.labels_train)
        test_accuracy = self.test_model(self.data_test, self.labels_test)
        print("** Test accuracy = ", test_accuracy)
        return test_accuracy, test_accuracy, accuracies

    def learn_single_fold(self):
        """Function to train clasifier across a single fold

        Returns:
            float: validation accuracy
        """
        self.train_model(self.data_train, self.labels_train)
        return self.test_model(self.data_test, self.labels_test)

    def get_gaussian_prob(self, x, mean, std):
        """Function to get probability from a  gaussian distribution

        Args:
            x (float)
            mean (float)
            std (float)

        Returns:
            float: probability
        """
        return (float)(1.0 / (math.sqrt(2.0 * math.pi) * std)) * math.exp((-(x - mean) ** 2) / (2 * (std ** 2)))

    def get_class_prob(self, data, alpha=0.0):
        """Function to get probability of classes
        Args:
            data (np.array)

        Returns:
            list: list of class probabilities i.e P(C_i)
            :param alpha:
        """
        num = [0, 0, 0, 0]
        total = data.shape[0]
        i = 0
        while i < 4:
            num[i] = (len(np.where(data[:, 0] == i)[0])) / (total)
            i = i + 1
        return num

    def classify(self, instance):
        """Function to classify a datapoint

        Args:
            instance (np.array): datapoint

        Returns:
            int: predicted class
        """
        probs = []
        i = 0
        while i < 4:
            # for i in range(4):
            p = self.class_prob[i]
            j = 0
            while j < instance.shape[1]:
                if (j in self.c_vars):
                    p = p * self.get_gaussian_prob(instance[0][j], self.means[i][j], self.std[i][j])
                else:
                    p = p * self.P[i][j][self.values[j].index(instance[0][j])]
                j = j + 1
            probs.append(p)
            i = i + 1
        return np.argmax(np.array(probs))

    def get_prob_matrix(self, X_train, Y_train, alpha=0.0):
        """Function to get probability matrix of features given classes

        Args:
            X_train (np.array): features
            Y_train (np.array): labels

        Returns:
            np.array: probability matrix of features given classes i.e. P(X_i| C_j)
            :param X_train:
            :param Y_train:
            :param alpha:
        """
        P = [[0 for j in range(X_train.shape[1])] for i in range(4)]

        for i in range(4):
            X_ = X_train[np.where(Y_train[:, 0] == i)]
            for j in range(X_train.shape[1]):
                P[i][j] = []

                k = 0
                while k < len(self.values[j]):
                    p = (len(np.where(X_[:, j] == self.values[j][k])[0]) + alpha) / (X_.shape[0] + alpha * X_.shape[0])
                    P[i][j].append(p)
                    k = k + 1
                P[i][j] = np.array(P[i][j])

        P = np.array(P, dtype=object)
        return P

    def calculate(self, X_train, Y_train):
        """
            Function to get mean and std deviation for continuous features
        """
        means = [[np.mean(X_train[np.where(Y_train[:, 0] == i)][:, j]) for j in range(X_train.shape[1])] for i in
                 range(4)]
        std = [[np.std(X_train[np.where(Y_train[:, 0] == i)][:, j]) for j in range(X_train.shape[1])] for i in range(4)]
        return means, std

    def train_model(self, X_train, Y_train):
        """Function to train the classifier

        Args:
            X_train (np.array): training set features
            Y_train (np.array): training set labels
        """
        self.P = self.get_prob_matrix(X_train, Y_train, alpha=self.alpha)
        self.class_prob = self.get_class_prob(Y_train, alpha=self.alpha)
        self.means, self.std = self.calculate(X_train, Y_train)

    def test_model(self, X_test, Y_test):
        """Function to test the classifier

        Args:
            :param X_test (np.array): testing set features
            :param Y_test (np.array): testing set labels

        Returns:
            :param float: test accuracy
        """
        count = 0
        i = 0
        while i < X_test.shape[0]:
            if self.classify(np.array([X_test[i]])) == Y_test[i][0]:
                count += 1
            i = i + 1
        accuracy = count / X_test.shape[0]
        return accuracy


def remove_outliers(X, Y):
    """Function to remove outliers

    Args:
        X (np.array): features
        Y (np.array): labels

    Returns:
        np.array: filtered features
        np.array: filtered labels
    """
    X_ = (np.abs(X) > 3)
    var = np.sum(X_, axis=1)
    sums = np.where(var == np.max(var))

    X = np.delete(X, sums, axis=0)
    Y = np.delete(Y, sums, axis=0)

    return X, Y


def Remove_Outlier_Features(X, features_names, alpha=0.0, laplace=False):
    """Function to perform feature removal and retrain the classfier

    Args:
        X (np array) : input data
        features_names (list) : column names
    """
    Y = X[:, 9]
    X = X[:, :9]
    print("Removing outliers i.e. samples with max features beyond mean(mew) + 3*(standard deviation)(sigma).......")
    print("Samples before removal: {}".format(X.shape[0]))
    X, Y = remove_outliers(X, Y)
    print("Samples after removal: {}\n".format(X.shape[0]))
    print("\n ============= NORMALISING FEATURES ============ \n")
    print("Initial features: ", ", ".join([v for i, v in enumerate(features_names)]), "\n")
    print()
    features = list(range(X.shape[1]))
    c_vars = [2]
    nb = Naive_Bayes_Classifier(X, c_vars, Y)
    curr_acc = nb.learn_single_fold()
    while True:
        accs = []
        for i in range(X.shape[1]):
            temp_X = np.delete(X, i, axis=1)
            nb = Naive_Bayes_Classifier(temp_X, [(j if j < i else j - 1) for j in c_vars if j != i], Y)
            accs.append(nb.learn_single_fold())
        accs = np.array(accs)
        accs_improvement = accs - curr_acc
        remove_col = np.argmax(accs_improvement)

        if accs_improvement[remove_col] < 0 or X.shape[1] == 1:
            break
        curr_acc = accs[remove_col]
        features.pop(remove_col)
        X = np.delete(X, remove_col, axis=1)
        c_vars = [(j if j < remove_col else j - 1) for j in c_vars if j != remove_col]

    print("Remaining features: ", ", ".join([v for i, v in enumerate(features_names) if i in features]), "\n")
    if laplace is False:
        print('\n\n\
            /////////////////////////////////\n\
            //                             //\n\
            //         SOLVING 3           //\n\
            //                             //\n\
            /////////////////////////////////\n\n\
            ')
    else:
        print('\n\n\
            /////////////////////////////////\n\
            //                             //\n\
            //         SOLVING 4           //\n\
            //                             //\n\
            /////////////////////////////////\n\n\
            ')
    print("\n============== TRAINING STARTED ============\n")
    return Naive_Bayes_Classifier(X, c_vars, Y, alpha=alpha, laplace=laplace).train_and_print_acc()


if __name__ == "__main__":
    start = time.time()
    print('NAIVE BAYES IMPLEMENTATION WITHOUT LAPLACE CORRECTION')
    print("*" * 75 + '\n\n\n')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Dataset_A.csv")
    args = parser.parse_args()
    PATH = args.data_path
    print("\n ============= READING DATA ============ \n")
    dataset = pd.read_csv(PATH)
    print("Time elapsed  =  {} s".format(time.time() - start))
    print("\n ============= DATA READ ============ \n\n")
    print("\n ============= FEATURES ============ \n")
    l = dataset.columns
    for p in l:
        print(p)
    print("\n ========= CLEANING DATA ============ \n")
    for col_name in dataset.columns:
        dataset[col_name].fillna(dataset[col_name].mode()[0], inplace=True)
    print("\n ========= ENCODING ============ \n")
    Encoder = preprocessing.LabelEncoder()
    for col_name in ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1", "Segmentation"]:
        dataset[col_name] = Encoder.fit_transform(dataset[col_name])
    print("Time elapsed  =  {} s\n".format(time.time() - start))
    data = dataset
    print('\n\n\
            /////////////////////////////////\n\
            //                             //\n\
            //         SOLVING 1           //\n\
            //                             //\n\
            /////////////////////////////////\n\n\
            ')
    NB = Naive_Bayes_Classifier(np.array(data, dtype=int)[:, 1:], [2, 5, 7])
    print("============= TRAIN TEST SPLIT COMPLETE ============\n")
    print("Time elapsed  =  {} s\n".format(time.time() - start))

    print('\n\n\
            /////////////////////////////////\n\
            //                             //\n\
            //         SOLVING 2           //\n\
            //                             //\n\
            /////////////////////////////////\n\n\
            ')
    average_accuracy1, test_accuracy1, accuracies1 = Remove_Outlier_Features(np.array(data, dtype=int)[:, 1:],
                                                                             data.columns[1:], alpha=0)
    print("\n ============= TRAINING FINISHED ============ \n\n")
    print('Maximum Accuracy', max(accuracies1))
    print('Minimum Accuracy', min(accuracies1))
    li = list(range(1, 11))

    print('\n\nNAIVE BAYES IMPLEMENTATION WITH LAPLACE CORRECTION IN PLACE')
    print("*" * 75 + '\n\n\n')
    NB.set_alpha(50.0)
    NB.set_laplace(val=True)
    average_accuracy2, test_accuracy2, accuracies2 = Remove_Outlier_Features(np.array(data, dtype=int)[:, 1:],
                                                                             data.columns[1:], alpha=50, laplace=True)
