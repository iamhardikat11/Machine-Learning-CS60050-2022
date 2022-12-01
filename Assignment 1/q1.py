import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
        
def cal_entropy(train_data, train_labels, key):
    entropy_ = 0
    values = train_labels.unique()
    for value in values:
        fraction = train_labels.value_counts()[value] / len(train_data)
        x = fraction
        entropy_ -= x * np.log2(x)
    target_variables = train_labels.unique()
    variables = train_data[key].unique()
    feature_entropy = 0
    epsilon = np.finfo(float).eps
    for variable in variables:
        entropy = 0
        feat = train_data[key]
        feat_var = feat[feat == variable]
        den = len(feat_var)
        for target_variable in target_variables:
            num = len(feat_var[train_labels == target_variable])
            fraction = num / (den + epsilon)
            entropy -= fraction * np.log(fraction + epsilon)
        fraction = den / len(train_data)
        feature_entropy -= fraction * entropy
    return entropy_ - abs(feature_entropy)


def get_best_feature(train_data, train_labels):
    IG = []
    for key in train_data.keys():
        IG.append(cal_entropy(train_data, train_labels, key))
    return train_data.keys()[np.argmax(IG)]


def get_best_feature(train_data, train_labels):
    IG = []
    for key in train_data.keys():
        IG.append(cal_entropy(train_data, train_labels))
    return train_data.keys()[np.argmax(IG)]


def data_cleanup(df):
    """
    Handling missing data using an imputing strategy and converting numerical columns to categorical.
    For now:
    Missing categorical data is filled with mode.
    Missing numerical data is filled with median.
    """
    # Fill numerical data with median
    df['Work_Experience'].fillna(df['Work_Experience'].median(), inplace=True)
    df['Family_Size'].fillna(df['Family_Size'].median(), inplace=True)

    # Fill categorical data with mode
    df['Ever_Married'].fillna(df['Ever_Married'].mode()[0], inplace=True)
    df['Graduated'].fillna(df['Graduated'].mode()[0], inplace=True)
    df['Profession'].fillna(df['Profession'].mode()[0], inplace=True)
    df['Var_1'].fillna(df['Var_1'].mode()[0], inplace=True)

    # Convert Work_Experience to categorical
    df['Work_Experience'] = df['Work_Experience'].mask(df['Work_Experience'] <= 2, 1)
    df['Work_Experience'] = df['Work_Experience'].mask((df['Work_Experience'] > 2) & (df['Work_Experience'] <= 5), 2)
    df['Work_Experience'] = df['Work_Experience'].mask((df['Work_Experience'] > 5) & (df['Work_Experience'] <= 8), 3)
    df['Work_Experience'] = df['Work_Experience'].mask(df['Work_Experience'] > 8, 4)

    # Convert age to categorical using label encoding based on certain range
    df['Age'] = df['Age'].mask(df['Age'] <= 12, 1)
    df['Age'] = df['Age'].mask((df['Age'] > 12) & (df['Age'] <= 19), 2)
    df['Age'] = df['Age'].mask((df['Age'] > 19) & (df['Age'] <= 26), 3)
    df['Age'] = df['Age'].mask((df['Age'] > 26) & (df['Age'] <= 45), 4)
    df['Age'] = df['Age'].mask((df['Age'] > 45) & (df['Age'] <= 60), 5)
    df['Age'] = df['Age'].mask(df['Age'] > 60, 6)



    # Convert Family_Size to categorical
    df['Family_Size'] = df['Family_Size'].mask(df['Family_Size'] <= 2, 1)
    df['Family_Size'] = df['Family_Size'].mask((df['Family_Size'] > 2) & (df['Family_Size'] <= 4), 2)
    df['Family_Size'] = df['Family_Size'].mask(df['Family_Size'] > 4, 3)
    return df


class Node:
    """
    The Node class for our decision tree.
    It contains the attributes required for each node and functions for various tasks.
    """

    def __init__(self, attr, prob_label):
        """
        Initializes a node with proper values.
        Args:
            attr (str): The decision attribute selected for the node
                    on the basis of which we split the tree further.
            prob_label (int): This is the most probable outcome if we were to convert this 
                    node to a leaf. It is calculated by determining which outcome 
                    occurs the most in the data points we have at this node.
        """
        self.attr = attr
        self.prob_label = prob_label
        self.children = {}  # dictionary of children nodes

    def is_leaf(self):
        """
        Checks if the given node is a leaf.
        Returns:
            bool: True, if the node is a leaf, otherwise False.
        """
        return not self.children  # if the node has no children, it is a leaf

    def node_count(self):
        """
        Finds the number of nodes in the subtree rooted at the given node.
        Returns:
            int: Number of nodes in the subtree.
        """
        if self.is_leaf():
            return 1
        count = 1
        for child in self.children.values():
            count += child.node_count()
        return count

    def prune(self, tree, accuracy, valid):
        """
        Prunes a node by performing reduced error pruning.
        Args:
            tree: The complete tree object
            accuracy: The accuracy on validation set that has been obtained thus far.
            valid: The validation dataframe used for calculating accuracy while pruning.
        Returns:
            float: If pruning this node returns a better accuracy, then return that else return original accuracy.
        """

    def prune(self, tree, accuracy, valid):
        """
        Prunes a node by performing reduced error pruning.
        Args:
            tree: The complete tree object
            accuracy: The accuracy on validation set that has been obtained thus far.
            valid: The validation dataframe used for calculating accuracy while pruning.
        Returns:
            float: If pruning this node returns a better accuracy, then return that else return original accuracy.
        """
        if self.is_leaf():
            return accuracy

        # If the node is not a leaf, we recursively prune all its children.
        # Then check if accuracy on validation increases after pruning the node.
        # If it does, we prune the node and return the new accuracy.
        # Else we return the original accuracy.
        for _, node in self.children.items():
            accuracy = node.prune(tree, accuracy, valid)

            # remove this node from the tree
            original_children = self.children
            self.children = {}

            # calculate accuracy on validation set
            new_accuracy = tree.test(valid)
            if new_accuracy > accuracy:
                return new_accuracy

            # if accuracy does not increase, we restore the node
            self.children = original_children
            return accuracy

    def print_tree_dfs(self, filename, line_gap=""):
        """
            Prints the tree to the given file.
            Args:
                file (str): The file to which the tree is to be printed.
                line_gap (str): The line gap to be used to just give intendation.
        """
        if self.is_leaf():
            print(f"{line_gap}OutCome = {self.prob_lable}", file=filename)
            return
        print(line_gap, end="", file=filename)
        print(self.attr, end=" ", file=filename)
        line_gap += "\t\t"
        for child in self.children:
            if child is None:
                continue
            child.print_tree_dfs(filename, line_gap)
        return


class DecisionTree:
    """
    The main Decision Tree class having metadata for the decision tree, and functions
    for various operations of the decision tree.
    """

    def __init__(self, max_depth=10, min_samples=1):
        """
        Initializes a decision tree with proper metadata.
        Args:
            max_depth (int, optional): Maxmimum depth of the decision tree. Defaults to 15.
            min_samples (int, optional): Minimum number of samples that must be present to
                    branch the tree further. Defaults to 1.
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree_depth = 0

    def train(self, train):
        """
        Trains the decision tree model.
        Args:
            train (pd.DataFrame): The training dataset.
        """
        train_data = train.drop(['ID', 'Segmentation'], axis=1)
        train_labels = train['Segmentation']
        self.root = self.build_tree(train_data, train_labels)

    def build_tree(self, train_data, train_labels, depth=0):
        """
        Builds the entire decision tree recursively, by selecting the feature to be split on,
        and then splitting the data into all unique values of that feature.
        Args:
            train_data (pd.DataFrame): The training dataset without the output labels.
            train_labels (pd.Series): The output labels for each row in the training dataset.
            depth (int, optional): Depth of the current node. Defaults to 0.
        Returns:
            Node: Root node of the tree
        """
        if (depth == self.max_depth) or (len(train_data) <= self.min_samples) or (len(train_labels.unique()) == 1):
            return self.create_leaf(train_labels)

        attr = get_best_feature(train_data, train_labels)
        node = Node(attr, train_labels.value_counts().sort_index().idxmax())

        for val in train_data[attr].unique():
            data_subset = train_data[train_data[attr] == val].copy()
            data_subset.drop(attr, axis=1, inplace=True)
            labels_subset = train_labels[data_subset.index]

            if data_subset.empty:
                return self.create_leaf(train_labels)
            node.children[val] = self.build_tree(data_subset, labels_subset, depth + 1)

        self.tree_depth = max(self.tree_depth, depth)
        return node

    def create_leaf(self, labels):
        """
        Creates and returns a leaf node for the decision tree.
        Args:
            labels (pd.Series): The output labels of the data points at this node.
        Returns:
            Node: The leaf node created.
        """
        prob_label = labels.value_counts().sort_index().idxmax()
        return Node('Segmentation', prob_label)

    def predict_one(self, test_instance, root):
        """
        Predicts the outcome on one row of data i.e. one test instance.
        Args:
            test_instance (dict): The test instance for which prediction is to be made.
            root (Node): The root node of the decision tree.
        Returns:
            string: Returns the predicted value from the set of labels.
        """
        if root.is_leaf():
            return root.prob_label
        edge = test_instance[root.attr]

        if edge not in root.children:
            return root.prob_label
        return self.predict_one(test_instance, root.children[edge])

    def predict(self, test_data):
        """
        Predicts the outcome on a set of test data.
        Args:
            test_data (pd.DataFrame): The test dataset for which predictions are to be made.
        Returns:
            pd.Series: Predicted outcomes (series of 0, 1 values) for the test dataset.
        """
        predictions = pd.Series([self.predict_one(row, self.root) for row in test_data.to_dict(orient='records')])
        return predictions

    def test(self, test):
        """
        Tests the decision tree model on the test dataset.
        Args:
            test (pd.DataFrame): The test dataset.
        Returns:
            float: Accuracy of the model on the test dataset.
        """
        test_data = test.drop(['ID', 'Segmentation'], axis=1)
        test_labels = test['Segmentation']
        predictions = self.predict(test_data)
        return 100.0 * accuracy_score(test_labels, predictions)


def main():
    data = pd.read_csv('Dataset_A.csv')
    data = data_cleanup(data)
    dt_best = None
    best_dpt = -1
    max_acc = -10 ** 18
    sum_acc = 0.0
    val_data_max = None
    test_data_max = None
    for i in range(1, 11):
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42 + i)
        train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42 + i)
        dt = DecisionTree(max_depth=10)
        dt.train(train_data)
        pred = dt.predict(test_data)
        acc = accuracy_score(test_data.iloc[:, -1], pred)
        print(f" Iteration {i} Test Accuracy = {(acc * 100.0)} at a current depth of {dt.tree_depth}")
        sum_acc += acc
        if acc > max_acc:
            max_acc = acc
            dt_best = dt
            best_dpt = dt.tree_depth
            val_data_max = val_data
            test_data_max = test_data
    avg_acc = sum_acc / 10
    print(f'Average Accuracy = {100.0 * avg_acc} %')
    print(f'Maximum Accuracy = {100.0 * max_acc} %')
    print('Depth of Tree with Maximum Accuracy = ', best_dpt)
    flag = True
    plt_x = []  # depth
    plt_y = []  # accuracy
    acc = max_acc
    dt = dt_best
    test_data = test_data_max
    iter = 0
    while True:
        iter += 1
        ini_acc = acc
        for _, node in dt.root.children.items():
            node.prune(dt, acc, val_data_max)
        acc = dt.test(test_data)
        plt_y.append(acc)
        plt_x.append(dt.tree_depth)
        if ini_acc >= acc:
            flag = False
            break
        if flag is False:
            break
    print('Number of Times Pruning Operation Happened:- ', iter)
    print(f"Final accuracy after pruning: {acc}")
    dt.root.print_tree_dfs('output2.txt', '\n')
    plt.plot(plt_x, plt_y)
    plt.show()
    plt.title('Plot of Depth vs. Accuracy')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.savefig('my_plot.png')


if __name__ == '__main__':
    main()
