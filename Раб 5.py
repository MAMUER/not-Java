# 1.1.2.3
import numpy as np


class Trigonometry:

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def tg(self, x):
        return np.tan(x)

    def arcsin(self, x):
        if -1 <= x <= 1:
            return np.arcsin(x)
        else:
            return "Ошибка"

    def arccos(self, x):
        if -1 <= x <= 1:
            return np.arccos(x)
        else:
            return "Ошибка"

    def arctg(self, x):
        return np.arctan(x)

    def deg_to_rad(self, x):
        return x * np.pi / 180

    def rad_to_deg(self, x):
        return x * 180 / np.pi


a = Trigonometry()
print(a.sin(np.pi / 2))
print(a.cos(np.pi / 2))
print(a.arccos(np.pi / 2))
print(a.arcsin(np.pi / 2))
print(a.arctg(np.pi / 2))
print(a.tg(np.pi / 2))
print(a.deg_to_rad(180))
print(a.rad_to_deg(np.pi))

# 2.1.2.3(1)
class Tree:
    def __init__(self, left, right):
        self.left = left
        self.right = right


t = ['a', ['b', ['d', [], []], ['e', [], []]], ['c', ['f', [], []], []]]
print('root = ', t[0])
print('left subtree = ', t[1])
print('right subtree = ', t[2])

# 2.1.2.3(2)
class Tree:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
            print("\n")
        print(self.data),
        if self.right:
            self.right.PrintTree()

    def insert(self, data):
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Tree(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Tree(data)
                else:
                    self.right.insert(data)

        else:
            self.data = data


root = Tree(12)
root.insert(1)
root.insert(13)
root.insert(33)
root.insert(10)
root.PrintTree()

# 1.3.1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
target = [0, 0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X[:, :-1],
    X[:, -1],
    test_size=0.2
)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

tree.plot_tree(classifier)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 1.4.1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics

url = r'https://raw.githubusercontent.com/likarajo/petrol_consumption/master/data/petrol_consumption.csv'
dataset = pd.read_csv(url)
print(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = DecisionTreeClassifier()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


df.plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


plt.scatter(dataset['Petrol_tax'], dataset['Petrol_Consumption'], color='b', label="Заработная плата")
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel("Опыт(лет)")
plt.ylabel("Заработная плата")
plt.show()

tree.plot_tree(regressor)

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))