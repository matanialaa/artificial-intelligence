import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list,  target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        for a in counts.values():
            i = (a / len(rows))
            impurity += i * np.log2(i)
        impurity *= -1
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        l_e = self.entropy(left, left_labels)
        r_e = self.entropy(right, right_labels)
        all_len = len(right) + len(left)
        info_gain_value = current_uncertainty - ((len(left) / all_len) * l_e) - \
                          ((len(right) / all_len) * r_e)
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        t_indexes, f_indexes = [], []
        for index, sampl in enumerate(rows):
            if question.match(
                    sampl):  # checking the sample if it matches the question and adding the indext to the right array - and using it later for labels and
                t_indexes.append(index)  # rows
            else:
                f_indexes.append(index)
        true_labels = labels[t_indexes]
        true_rows = rows[t_indexes]
        false_labels = labels[f_indexes]
        false_rows = rows[f_indexes]
        gain = self.info_gain(false_rows, false_labels, true_rows, true_labels, current_uncertainty)
        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        for f_indx, feature in enumerate(rows.T):
            f_vals = []
            for row in rows:
                f_vals.append(row[f_indx])
            f_vals.sort()
            for i in range(len(f_vals) - 1):
                question = Question(feature, f_indx, f_vals[i + 1])
                c_gain, c_t_rows, c_t_labels, c_f_rows, c_f_labels = self.partition(rows, labels, question,
                                                                                    current_uncertainty)
                if c_gain >= best_gain:
                    best_gain, best_true_rows, best_true_labels, best_false_rows, best_false_labels, best_question = \
                        c_gain, c_t_rows, c_t_labels, c_f_rows, c_f_labels, question
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer. 

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        if self.entropy(rows, labels) == 0:
            return Leaf(rows, labels)
        gain, best_question, true_rows, true_labels, false_rows, false_labels = self.find_best_split(rows, labels)
        true_branch = self.build_tree(true_rows, true_labels)
        false_branch = self.build_tree(false_rows, false_labels)
        # ========================

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        if type(node) is Leaf:
            predict = node.predictions
            al_values = list(predict.values())
            al_keys = list(predict.keys())
            if len(al_keys) <= 1:
                return al_keys[0]
            if al_values[1] > al_values[0]:
                return al_keys[1]
            return al_keys[0]

        if node.question.match(row):
            prediction = self.predict_sample(row, node.true_branch)
        else:
            prediction = self.predict_sample(row, node.false_branch)
        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = None

        # ====== YOUR CODE: ======
        all_predictions = []
        rootie = self.tree_root
        for r in rows:
            predict = self.predict_sample(r, rootie)
            all_predictions.append(predict)
        y_pred = np.array(all_predictions)
        # ========================

        return y_pred
