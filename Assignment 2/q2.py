import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings
import math
import copy as cp

from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

np.random.seed(42)
warnings.filterwarnings("ignore")


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("simulation2.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()
filename = 'wine_data.csv'

dataclasses = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
               'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
               'Proline']


def standard_scalar_normalize(dataset):
    """
    This function normalizes each column of the dataset (divides each column value with the corresponding mean value)
    Parameters:
    -----------
    dataset: dataset whose columns are to be normalized
    Returns:
    --------
    dataset: normalized dataset. Each column of the dataset adds to 1
    """

    for col in range(len(dataclasses)):
        column = dataclasses[col]
        row = len(dataset[column])
        mean = sum(dataset[column]) / row
        std = math.sqrt(dataset[column].var())
        for i in range(row):
            dataset[column][i] = (dataset[column][i] - mean) / std
    return dataset


def test_train_split_data(X, y, test_size=0.2):
    """
    Args:
    X (pd.DataFrame): input dataframe
    y (pd.Series): labels
    Returns:
    X_train: subset of input for training
    y_train: training set labels
    X_test: subset of input used for testing
    y_test: testing set labels
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(X.shape[0] * test_size)
    trainX = X.iloc[indices[split:]]
    trainY = y.iloc[indices[split:]]
    testX = X.iloc[indices[:split]]
    testY = y.iloc[indices[:split]]
    return trainX, testX, trainY, testY


def autolabel(ax, reacts):
    """
    Attach a text label above each bar, displaying its height.
    """
    for rect in reacts:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def sequential_forward_selection(learning_rate, hidden_layer, trainX, testX, trainY, testY):
    """
     :Arguments:

       trainX  :   subset of input for trainig
       testX :    subset of input for  testing
       trainY  :   training  set label
       testY   :  testing set label
       learning_rate : learning rate
       hidden_layer :    hidden layer
     :returns:
        best_features:  a list
        best_acc : a float value
    """
    best_features = []
    best_acc = 0
    for i in range(len(dataclasses)):
        best_column = ''
        for dataclass in dataclasses:
            if dataclass not in best_features:
                mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer, learning_rate_init=learning_rate,
                                        batch_size=32)
                mlp_clf.fit(trainX[best_features + [dataclass]], trainY)
                mlp_clf.predict(testX[best_features + [dataclass]])
                acc = mlp_clf.score(testX[best_features + [dataclass]], testY)
                if acc > best_acc:
                    best_acc = acc
                    best_column = dataclass
        if best_column != '':
            best_features.append(best_column)
    return best_features, best_acc


def cross_val_predict(model, kfold: KFold, X: np.array, y: np.array):
    ''' 
    Arguments:  
    model :  object of class Kfold
    kfold :  object of class Kfold
    X (pd.DataFrame): input dataframe
    y (pd.Series): labels
    returns:    
    actual_classes :   np.array
    predictions_classes :  np.array     
    predicted_probas :      
    '''
    model_ = cp.deepcopy(model)
    no_classes = len(np.unique(y))
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])

    for train_ndx, test_ndx in kfold.split(X):
        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
        actual_classes = np.append(actual_classes, test_y)
        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))
        predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)
    return actual_classes, predicted_classes, predicted_proba


def cross_val_predict_all_classifiers(classifiers, X, y, kfold):
    '''
     Arguments:
     classifiers  :  dictionary
     X (pd.DataFrame): input dataframe
    y (pd.Series): labels
        kfold :  object of class Kfold
      
      returns:
        actual:     a list of actual classes
        predictions  :          a list of predicted classes
        predicted_probas :  a list of predicted probabilities


    '''
    predictions = [None] * len(classifiers)
    predicted_probas = [None] * len(classifiers)
    actual = None
    for i, (name, classifier) in enumerate(classifiers.items()):
        actual, predictions[i], predicted_probas[i] = cross_val_predict(classifier, kfold, X, y)
    return actual, predictions, predicted_probas


def hard_voting(predictions):
    '''
        Arguments:
        predictions  :  a list of predicted classes
        returns:    
        hard_voting_predictions :  a list of predicted classes
    '''
    return [stats.mode(v)[0] for v in np.array(predictions).T]


def main():
    start = time.time()
    ######################  Q1  ########################
    print("\n ============= READING DATA ============ \n")
    dataset_ori = pd.read_csv(filename, header=None)
    labels_ori = dataset_ori[0]
    dataset = dataset_ori.iloc[:, 1:]
    dataset.columns = dataclasses
    print(dataset)
    print("Time elapsed  =  {} s".format(time.time() - start))
    print("\n ============= DATA READ ============ \n\n")
    print("\n ============= FEATURES ============ \n\n")
    j = 1
    for i in dataclasses:
        print(f'{j}.', end=' ')
        j += 1
        print(i)
    print("\n ============= STANDARD SCALAR NORMALISATION ============ \n\n")
    dataset_scaled = standard_scalar_normalize(dataset)
    print("\n ============= PROCESSED DATASET ============ \n\n")
    print(dataset_scaled)
    trainX, testX, trainY, testY = test_train_split_data(dataset_scaled, labels_ori, test_size=0.2)
    print("============= TRAIN TEST SPLIT COMPLETE ============\n")
    print("Train Data size: {} \nTest Data size = {}".format(len(trainX), len(testX)))
    print("Time elapsed  =  {} s\n\n".format(time.time() - start))

    ######################  Q2 ########################

    print("============= BINARY SUPPORT VECTOR MACHINE(SVM) ============\n")
    print("\n============== TRAINING STARTED ============\n")
    rbf = svm.SVC(kernel='rbf').fit(trainX, trainY)
    linear = svm.SVC(kernel='linear').fit(trainX, trainY)
    quadratic = svm.SVC(kernel='poly', degree=2).fit(trainX, trainY)
    print("Time elapsed  =  {} s".format(time.time() - start))
    print("\n ============= TRAINING FINISHED ============ \n\n")
    linear_pred = linear.predict(testX)
    rbf_pred = rbf.predict(testX)
    quadratic_pred = quadratic.predict(testX)
    linear_accuracy = accuracy_score(testY, linear_pred)
    linear_f1 = f1_score(testY, linear_pred, average='weighted')
    rbf_accuracy = accuracy_score(testY, rbf_pred)
    rbf_f1 = f1_score(testY, rbf_pred, average='weighted')
    quadratic_accuracy = accuracy_score(testY, quadratic_pred)
    quadratic_f1 = f1_score(testY, quadratic_pred, average='weighted')
    print('Accuracy (Linear Kernel): ', "%.2f" % (linear_accuracy * 100))
    print('F1 (Linear Kernel): ', "%.2f" % (linear_f1 * 100))
    print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy * 100))
    print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1 * 100))
    print('Accuracy (Quadratic Kernel): ', "%.2f" % (quadratic_accuracy * 100))
    print('F1 (Quadratic Kernel): ', "%.2f" % (quadratic_f1 * 100))
    values = {1: "Linear", 2: "Radial Basis Function", 3: "Quadratic"}
    acc_list = [linear_accuracy, rbf_accuracy, quadratic_accuracy]
    f1_list = [linear_f1, rbf_f1, quadratic_f1]
    value = values[acc_list.index(max(acc_list)) + 1]
    print(f"\nMaximum Accuracy is Achieved for {value} Kernel.")
    value = values[f1_list.index(max(f1_list)) + 1]
    print(f"Maximum F1-Score is Achieved for {value} Kernel.")
    print("\n ============= MULTI-LAYER PERCEPTRON(MLP) CLASSIFIER ============ \n\n")
    print("\n============== TRAINING STARTED ============\n")
    learning_rate = 0.001
    batch_size = 32
    acc_list = []
    print("\n * Stochastic Gradient Optimiser")
    print(" -> Batch Size = 32")
    print(" -> Learning Rate = 0.001")
    print(" Implemented with varying hidden layer:- ")
    print("   1. 1 hidden layer with 16 nodes \n   2. 2 hidden layers with 256 and 16 nodes respectively.")
    mlp_clf_1 = MLPClassifier(hidden_layer_sizes=(16,), learning_rate_init=learning_rate, batch_size=batch_size,
                              solver='sgd')
    mlp_clf_1.fit(trainX, trainY)
    mlp_clf_1.predict(testX)
    acc_list.append(mlp_clf_1.score(testX, testY))
    mlp_clf_2 = MLPClassifier(hidden_layer_sizes=(256, 16), learning_rate_init=learning_rate, batch_size=batch_size,
                              solver='sgd')
    mlp_clf_2.fit(trainX, trainY)
    mlp_clf_2.predict(testX)
    acc_list.append(mlp_clf_2.score(testX, testY))
    print(f"Accuracy is {acc_list[0]} for 1st Classifier.")
    print(f"Accuracy is {acc_list[1]} for 2nd Classifier.")
    print("Time elapsed  =  {} s\n".format(time.time() - start))
    print("\n ============= TRAINING FINISHED ============ \n\n")
    print("\n ========= VARYING LEARNING RATE TO FIND BEST ONE ========= \n")
    print("\n============== TRAINING STARTED ============\n")
    learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    best_hidden_layer = (256, 16) if acc_list.index(max(acc_list)) == 1 else (16,)
    acc_list = []
    for learn_rate in learning_rate:
        mlp_clf = MLPClassifier(hidden_layer_sizes=best_hidden_layer, learning_rate_init=learn_rate,
                                batch_size=batch_size)
        mlp_clf.fit(trainX, trainY)
        mlp_clf.predict(testX)
        acc_list.append(100 * mlp_clf.score(testX, testY))
        formatted_acc = "{:.4f}".format(acc_list[len(acc_list) - 1])
        formatted_lr = "{:.5f}".format(learn_rate)
        print(f" * Accuracy is {formatted_acc} for Learning Rate: {formatted_lr}")
    learning_rate_plot = ['0.00001', '0.0001', '0.001', '0.01', '0.1']
    plt.scatter(learning_rate_plot, acc_list, s=20, alpha=0.5)
    plt.plot(learning_rate_plot, acc_list)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy of MLP Classifiers')
    plt.title('Learning Rate VS. Accuracy of Multi-Layer Perceptron Plot')
    plt.savefig('learning_rate_vs_acc.png')
    print("Time elapsed  =  {} s\n".format(time.time() - start))
    print("\n ============= TRAINING FINISHED ============ \n\n")
    print("\n ============= FORWARD SELECTION METHOD ============ \n")
    best_learning_rate = learning_rate[acc_list.index(max(acc_list))]
    print("\n ****** THE BEST FEATURES ARE ******** \n")
    best_features, best_acc = sequential_forward_selection(best_learning_rate, best_hidden_layer, trainX, testX, trainY,
                                                           testY)
    c = 1
    for i in best_features:
        print(f" * {c}. {i}")
        c += 1
    print('Accuracy for these Best Features Selected from Forward Selection: ', best_acc)
    svm_rbf = svm.SVC(kernel='rbf')
    svm_poly = svm.SVC(kernel='poly', degree=2)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(256, 16), learning_rate_init=0.001,
                            batch_size=32)
    classifiers = dict()
    classifiers["SVM Radial Basis Function"] = svm_rbf
    classifiers["SVM Quadratic"] = svm_poly
    classifiers["Multi Level Perceptron Classifier"] = mlp_clf
    X = dataset_scaled.to_numpy()
    y = np.array(labels_ori)
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    actual, predictions, predicted_probas = cross_val_predict_all_classifiers(classifiers, X=X, y=y, kfold=kfold)
    hv_predictions = hard_voting(predictions)
    print("\n ============= ENSEMBLE LEARNING ============ \n")
    print(f"Accuracy of Hard Voting: {100 * accuracy_score(actual, hv_predictions):.5f} %.")
    plt.show()


if __name__ == "__main__":
    file = open('simulation2.txt', 'r+')
    file.truncate(0)
    main()
    file.close()
