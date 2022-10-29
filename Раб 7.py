# 1.2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

url1 = r'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
url2 = r'https://raw.githubusercontent.com/AnnaShestova/salary-years-simple-linear-regression/master/Salary_Data.csv'

dataset1 = pd.read_csv(url1)
dataset2 = pd.read_csv(url2)

X_digits, Y_digits = dataset1.iloc[:, :-1].data, dataset1.target
print('Dataset Sizes: ', X_digits.shape, Y_digits.shape)

X_boston, Y_boston = dataset2.data, dataset2.target
print('Dataset Sizes: ', X_boston.shape, Y_boston.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_digits, Y_digits, train_size=0.8, test_size=0.2,
                                                    stratify=Y_digits, random_state=123)
print('Train/Test Sizes: ', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

mlp_classifier = MLPClassifier(random_state=123)
mlp_classifier.fit(X_train, Y_train)

Y_preds = mlp_classifier.predict(X_test)

print(Y_preds[:15])
print(Y_test[:15])
print('Test Accuracy: %.3f' % mlp_classifier.score(X_test, Y_test))
print('Training Accuracy: %.3f' % mlp_classifier.score(X_train, Y_train))


def plot_confusion_matrix(Y_test, Y_preds):
    conf_mat = confusion_matrix(Y_test, Y_preds)
    fig = plt.figure(figsize=(6, 6))
    plt.matshow(conf_mat, cmap=plt.cm.Blues, fignum=1)
    plt.yticks(range(10), range(10))
    plt.xticks(range(10), range(10))
    plt.colorbar()
    for i in range(10):
        for j in range(10):
            plt.text(i - 0.2, j + 0.1, str(conf_mat[j, i]), color='tab:red')


plot_confusion_matrix(Y_test, mlp_classifier.predict(X_test))

print("Loss: ", mlp_classifier.loss_)
print("Number of Coefs: ", len(mlp_classifier.coefs_))
print("Number of Intercepts: ", mlp_classifier.intercepts_)
print("Number of Iterations for Which Estamator Ran: ", mlp_classifier.n_iter_)
print("Name of Output Layer Activation Function: ", mlp_classifier.out_activation_)

mlp_regressor = MLPRegressor(random_state=123)
mlp_regressor.fit(X_train, Y_train)

Y_preds = mlp_regressor.predict(X_test)

print(Y_preds[:10])
print(Y_test[:10])
print('Test R^2 Score: %.3f' % mlp_regressor.score(X_test, Y_test))
print('Training R^2 Score: %.3f' % mlp_regressor.score(X_train, Y_train))

print("Loss: ", mlp_regressor.loss_)
print("Number of Coefs: ", len(mlp_regressor.coefs_))
print("Number of Intercepts: ", mlp_regressor.intercepts_)
print("Number of Iterations for Which Estamator Ran: ", mlp_regressor.n_iter_)
print("Name of Output Layer Activation Function: ", mlp_regressor.out_activation_)
