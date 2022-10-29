# 1.3.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a = np.array([0, 0, 0])
ax.scatter(a[0], a[1], a[2])

b = np.array([3, 3, 3])
ax.scatter(b[0], b[1], b[2])

c = np.array([1, 1, 1])
ax.scatter(c[0], c[1], c[2])

d = np.array([2, 2, 2])
ax.scatter(d[0], d[1], d[2])

print(np.linalg.norm(b - a))
print(np.linalg.norm(c - d) ** 2)
print(np.linalg.norm(d - a, ord=np.inf))
print(np.linalg.norm(b - c, ord=1))

plt.show()

# 1.3.2
import matplotlib.pyplot as plt
import numpy as np

Z = np.zeros((5, 5))
Z += np.arange(5)
print(Z)

# 2.3.1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

iris = sns.load_dataset('iris')

x_tr, x_t, y_tr, y_t = train_test_split(
    iris.iloc[:, :-1],
    iris.iloc[:, -1],
    test_size=0.15
)

k = 1  # 5 and 10
model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_tr, y_tr)

y_pr = model.predict(x_t)

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=iris,
    x='petal_width', y='petal_length',
    hue='species',
    s=70
)
plt.xlabel('Длина лепестка, см')
plt.ylabel('Ширина лепестка, см')
plt.legend(loc=2)
plt.grid()

for i in range(len(y_t)):
    if np.array(y_t)[i] != y_pr[i]:
        plt.scatter(x_t.iloc[i, 3], x_t.iloc[i, 2], color='red', s=150)

print(f'accuracy: {accuracy_score(y_t, y_pr):.3}')

plt.show()

# 3.3.2
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

da = pd.DataFrame({"цвет глаз": ["карий", "карий", "чёрный", "серый", "жёлтый"]})

scale_mapper = {"карий": 1, "чёрный": 2, "серый": 3, "жёлтый": 4}

da["цвет глаз"].replace(scale_mapper)
print(da, "\n")

dic = [{"карий": 2, "чёрный": 7}, {"жёлтый": 1, "серый": 9}]

d = DictVectorizer(sparse=False)

features = d.fit_transform(dic)

print(features)
