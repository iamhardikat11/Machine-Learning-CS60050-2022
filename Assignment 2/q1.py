import time
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
from numpy.linalg import norm

np.random.seed(101)


class Logger(object):
    def __init__(self):
        '''Constructor'''
        self.terminal = sys.stdout
        self.log = open("simulation1.txt", "a")

    def write(self, message):
        '''Redirects the print output to the log file'''
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()
filename = 'wine_data.csv'
dataclasses = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
               'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
               'Proline']


def computeMI(x, y):
    '''
    Computes the mutual information between two variables'''
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([len(x[x == xval]) / float(len(x)) for xval in x_value_list])  # P(x)
    Py = np.array([len(y[y == yval]) / float(len(y)) for yval in y_value_list])  # P(y)
    for i in range(len(x_value_list)):
        if Px[i] == 0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy) == 0:
            continue
        pxy = np.array([len(sy[sy == yval]) / float(len(y)) for yval in y_value_list])  # p(x,y)
        t = pxy[Py > 0] / Py[Py > 0] / Px[i]  # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t > 0] * np.log2(t[t > 0]))  # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi


def normalized_mutual_info_score(x, y):
    """
        Computes the normalized mutual information between two variables
        Args:
            x (ndarray): first variable
            y (ndarray): second variable
    """
    mi = computeMI(x, y)  # Compute the mutual information
    h_x = computeMI(x, x)  # Compute the entropy of x
    h_y = computeMI(y, y)  # Compute the entropy of y
    return 2. * mi / (h_x + h_y)  # Return the normalized mutual information


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


def data_preprocessor(dataset=None):
    """
    Apply MinMax Scalar PreProcessing to the DataSet so that
    all numerical data get in the range [0.0,1.0]
    :param dataset: original dataset with label dropped
    :return: processed dataset
    """
    scalar = StandardScaler()
    scalar.fit(dataset)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled_dataset = scalar.transform(dataset)
    return scaled_dataset


def find_closest_cluster(distance):
    """ Find the closest cluster for each data point """
    return np.argmin(distance, axis=1)


class K_means:
    def __init__(self, n_clusters, max_iter=100, random_state=42):
        '''
        args:
        n_clusters: number of clusters
         max_iter: maximum number of iterations
         random_state: random state
           returns: None         
        '''
        self.error = None
        self.centroids = None
        self.labels = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initialise_centroids(self, X):
        '''
        args:
        X: data
        returns: None
            returns:
            centroids: initial centroids'''
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        '''
        args:
        X: data
        labels: labels
        returns:
        centroids: new centroids'''
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        '''
        args:
        X: data
        centroids: centroids
        returns:
        distance: distance between data and centroids'''
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def compute_sse(self, X, labels, centroids):
        '''
        args:
        X: data

        labels: labels
        centroids: centroids
        returns:
        sse: sum of squared error'''
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X):
        '''
        args:
        X: data
        returns: None'''

        self.centroids = self.initialise_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)

    def predict(self, X):
        '''
        args:
        X: data
        returns:
        labels: labels'''
        distance = self.compute_distance(X, self.centroids)
        return find_closest_cluster(distance)


def k_means_clustering(k, clustering_dataset, true_labels):
    """
    Apply K-Means Clustering Algorithm to the DataSet
    args:
    k: number of clusters
     clustering_dataset: dataset to be clustered
     true_labels: true labels of the dataset
    return: None
    """
    km = K_means(n_clusters=k, max_iter=2000)
    km.fit(clustering_dataset)
    km_labels = km.predict(clustering_dataset)
    true_labels = np.array(true_labels)
    nmi = normalized_mutual_info_score(true_labels, km_labels)
    return nmi


def main():
    start = time.time()
    ######################  Q1  ########################
    print("\n ============= READING DATA ============ \n")
    dataset_ori = pd.read_csv(filename, header=None)
    labels_ori = dataset_ori.iloc[:, 0].values.tolist()
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
    print("\n ============= APPLYING PCA ============ \n")
    scaled_dataset = data_preprocessor(dataset)
    pca = PCA(n_components=13)
    pca.fit(scaled_dataset)
    var = pca.explained_variance_ratio_[:]  # percentage of variance explained
    labels = ['PC' + str(i + 1) for i in range(len(var))]

    fig, ax = plt.subplots(figsize=(15, 7))
    plot1 = ax.bar(labels, var)

    ax.set_title('PCA Plot')
    ax.plot(labels, var)
    ax.set_title('Proportion of Variance Explained VS Principal Component')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Proportion of Variance Explained')
    autolabel(ax, plot1)
    plt.savefig('variance_ratio_pca.png')

    cumsum = [i for i in var]
    nc = -1
    for i in range(1, len(cumsum)):
        cumsum[i] += cumsum[i - 1]
        if cumsum[i] >= 0.95 and nc == -1:
            nc = i + 1
    fig, ax = plt.subplots(figsize=(15, 7))
    plot2 = ax.bar(labels, cumsum)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Variance Ratio s Cumulative sum')
    ax.set_xlabel('Number principal components')
    ax.set_title('Variance Ratio cumulative sum VS number principal components')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)

    ax.axvline('PC' + str(nc), c='red')
    ax.axhline(0.95, c='green')
    ax.text('PC5', 0.95, '0.95', fontsize=15, va='center', ha='center', backgroundcolor='w')
    autolabel(ax, plot2)

    print('Number of Components selected: {}'.format(nc))
    print('Variance captured: {} % ( > 95%)'.format(cumsum[nc - 1] * 100))
    plt.savefig('variance_ratio_cumulative sum.png')
    print("Time elapsed  =  {} s".format(time.time() - start))
    print("\n ============= COMPONENTS SELECTED FOR CLUSTERING ============ \n\n")

    ######################  Q2  ########################
    clustering_dataset = pd.DataFrame(pca.transform(scaled_dataset), columns=dataclasses).iloc[:, 0: nc]
    print(clustering_dataset)
    print("Time elapsed  =  {} s".format(time.time() - start))
    ######################  Q3  ########################
    print("\n ============= CLUSTERING STARTED ============ \n")
    normalised_mutual_info = []
    clustering_dataset = DataFrame.to_numpy(clustering_dataset)
    for i in range(len(labels_ori)): labels_ori[i] -= 1
    i = 1
    for k in range(2, 9):
        nmi = k_means_clustering(k, clustering_dataset, true_labels=labels_ori)
        normalised_mutual_info.append(nmi)
        print(f"--> Iteration {i}: For value of K: {k} Normalised Mutual Info:- {nmi}.")
        i += 1
    print("\nTime elapsed  =  {} s".format(time.time() - start))
    print("\n ============= CLUSTERING FINISHED ============ \n\n")
    labels = [item for item in range(2, 9)]
    fig, ax = plt.subplots(figsize=(15, 7))
    plot3 = ax.bar(labels, normalised_mutual_info)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.plot(labels, normalised_mutual_info)
    ax.set_title('NMI vs K Plot')
    ax.set_xlabel('K')
    ax.set_ylabel('Normalised Mutual Info.')
    autolabel(ax, plot3)
    plt.savefig('k_vs_nmi.png')
    maxi_k = max(normalised_mutual_info)
    print("\n ============= NORMALISED MUTUAL INFO. RESULT ============ \n\n")
    print(
        f"The Maximum value for NMI is :- {maxi_k} and is Obtained for K = {2 + normalised_mutual_info.index(maxi_k)}")
    plt.show()


if __name__ == '__main__':
    file = open('simulation1.txt', 'r+')
    file.truncate(0)
    main()
    file.close()
