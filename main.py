import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("error", category=RuntimeWarning)


# implementation Dtree with C4.5


class Dtree:
    class Dataset:

        def __init__(self, data):
            self.data = data
            self.is_left = False

        def is_left_sex(self):  # to check that tree is allowed to use pregnant attrebute
            return self.is_left

    class Node:

        def __init__(self, data=None, label=None, condition=None):
            self.data = data  # to store dataframe
            self.label = label  # to store attrebute name
            self.condition = condition  # to store treshold
            self.right = None  # to store sibllings
            self.left = None  # to store sibllings

        def __str__(self):
            return f'{self.label},{self.condition}'

    def __init__(self, data_set):
        self.data_set = self.Dataset(data_set)
        self.root = self.Node(self.data_set)

    def Entropy(self, array):  # calculate entropy for a array
        unique, counts = np.unique(array, return_counts=True)
        probabilities = counts / len(array)
        return np.sum(probabilities * np.log2(probabilities)) * (-1)

    def SplitINFO(self, head, right, left):  # calculate spliteinfo for array and splitted array
        try:
            p_right = len(right) / len(head)
            p_left = len(left) / len(head)
            a = (p_right * np.log2(p_right) + p_left * np.log2(p_left)) * (-1)
        except RuntimeWarning:  # if array is empty
            return 'splited'
        return a

    def InformationGain_OptimizedForRegression(self, head_entropy, len_head, left,
                                               right):  # calculate spliteinfo for array and splitted array
        return head_entropy - ((len(left) / len_head) * self.Entropy(left)) - (
                (len(right) / len_head) * self.Entropy(right))

    def InformationGain(self, head, left, right):  # calculate Information Gain for array and splitted array
        return self.Entropy(head) - ((len(left) / len(head)) * self.Entropy(left)) - (
                (len(right) / len(head)) * self.Entropy(right))

    def GainRatio(self, head, right, left):  # calculate GainRAtio for array and splitted array
        if self.SplitINFO(head, right, left) == 'splited':
            return 'splited'
        return self.InformationGain(head, right, left) / self.SplitINFO(head, right, left)

    def findCriterionfortreshold(self, sorted_age_array, sorted_target_array):  # finding treshold
        entropy_head = self.Entropy(sorted_target_array)
        len_head = len(sorted_target_array)
        b = {}

        for i, j in enumerate(sorted_age_array):

            try:  # calculating informatin gain for each age
                if sorted_age_array[i] != sorted_age_array[i + 1]:
                    index = i + 1
                    splited_array = np.split(sorted_target_array, [index])
                    b[j + 0.5] = (index,
                                  self.InformationGain_OptimizedForRegression(entropy_head, len_head, splited_array[0],
                                                                              splited_array[1]))

            except IndexError:
                pass

        treshold = max(b, key=lambda k: b[k][1])
        splited_array = np.split(sorted_target_array, [b[treshold][0]])
        return treshold, self.InformationGain_OptimizedForRegression(entropy_head, len_head, splited_array[0],
                                                                     splited_array[1]) / self.SplitINFO(
            sorted_target_array, splited_array[0], splited_array[1])

    def finddivideCriterion(self, inp):  # finding the best Criterion for making tree
        data_set = inp.data
        criterions = {}

        for col in data_set.columns.tolist():

            if col == 'AGE' or col == 'MEDICAL_UNIT':  # if attrebute is regression
                temp = data_set[[col, 'CLASIFFICATION_FINAL']].to_numpy()
                sorted_temp = temp[temp[:, 0].argsort()]
                sorted_age_array, sorted_target_array = sorted_temp[:, 0], sorted_temp[:, 1]
                treshold = self.findCriterionfortreshold(sorted_age_array, sorted_target_array)
                criterions[col] = (treshold[0], treshold[1])

            elif col != 'CLASIFFICATION_FINAL':  # if attrebute is classification

                if col == 'PREGNANT' and inp.is_left:
                    pass

                else:
                    temp = data_set[[col, 'CLASIFFICATION_FINAL']]
                    head = data_set[['CLASIFFICATION_FINAL']].to_numpy()
                    side_1 = temp[temp[f'{col}'] <= 1.5].to_numpy()[:, 1]
                    side_2 = temp[temp[f'{col}'] > 1.5].to_numpy()[:, 1]
                    gain_ratio = self.GainRatio(head, side_1, side_2)

                    if gain_ratio == 'splited':
                        criterions[col] = (1.5, -1)

                    else:
                        criterions[col] = (1.5, self.GainRatio(head, side_1, side_2))

        max_val = max(criterions, key=lambda k: criterions[k][1])
        return (max_val, criterions[max_val])

    def inorder(self, node):  # to show tree
        if node.right == None and node.left == None:
            return
        else:
            self.inorder(node.left)
            print(node)
            self.inorder(node.right)

    def predict(self, node, sample):  # to predict sample with tree
        if node.left is None and node.right is None:
            return node.condition

        feature = node.label
        threshold = node.condition

        if sample[feature] <= threshold:
            return self.predict(node.left, sample)
        else:
            return self.predict(node.right, sample)

    def making_tree(self, node):

        data_set = node.data

        try:  # if dataset is emplty
            criterion = self.finddivideCriterion(data_set)
        except ValueError:
            criterion = None
        data_set = data_set.data

        if criterion == None:  # main condition
            return node

        if data_set.shape[1] == 2:  # main condition
            node.label = data_set.columns[-1]
            node.condition = data_set[data_set.columns[-1]].mode()[0]
            return node

        if (data_set[data_set.columns[-1]] == data_set[data_set.columns[-1]].iloc[0]).all():  # main condition
            node.label = data_set.columns[-1]
            node.condition = data_set[data_set.columns[-1]].mode()[0]
            return node

        if criterion[1][1] == -1:  # main condition
            node.label = data_set.columns[-1]
            node.condition = data_set[data_set.columns[-1]].mode()[0]
            return node

        node.label = criterion[0]
        node.condition = criterion[1][0]

        # splitting dataframe according to label and condition
        right_data = self.Dataset(data_set[data_set[criterion[0]] > criterion[1][0]].drop(columns=[criterion[0]]))
        left_data = self.Dataset(data_set[data_set[criterion[0]] <= criterion[1][0]].drop(columns=[criterion[0]]))

        # recursive part
        if not right_data.data.empty:
            if criterion[0] == 'SEX':  # if male
                try:
                    right_data = self.Dataset(right_data.data.drop('PREGNANT', axis=1))  # delete pregnant
                except KeyError:
                    pass
            node.right = self.making_tree(self.Node(data=right_data))
        else:
            node.right = None

        if not left_data.data.empty:
            if criterion[0] == 'SEX':  # if female
                left_data.is_left = True
            node.left = self.making_tree(self.Node(data=left_data))  # able to use pregnant
        else:
            node.left = None

        return node
