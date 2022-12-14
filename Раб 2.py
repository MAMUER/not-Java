# 1.3.1
import numpy as np

x = np.array([[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]])
print(x)

# 1.3.2
import numpy as np

x = np.arange(25).reshape(5, 5)

print(x.shape, x.size)

# 1.3.3
import numpy as np

x = np.random.random((3, 3, 3))
print(x)

# 1.3.4
import numpy as np

x = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
print(x)

# 1.3.5
import numpy as np

x = np.random.random(5)

print(np.sort(x)[::-1])

# 1.3.6
import numpy as np

x = np.arange(25).reshape(5, 5)

print(x.shape, x.size, x)

# 2.3.1
import pandas as pd
import math as m

a = pd.Series([2, 3, 7])
b = pd.Series([0, -5, 9])
print(m.sqrt((a[0] - b[0]) ^ 2 + (a[1] - b[1]) ^ 2 + (a[2] - b[2]) ^ 2))

# 2.3.2
import pandas as pd

url = 'https://raw.githubusercontent.com/akmand/datasets/main/FMLPDA_Table4_3.csv'

dataset = pd.read_csv(url)

# 2.3.3
import pandas as pd

b = 'https://raw.githubusercontent.com/akmand/datasets/main/FMLPDA_Table4_3.csv'

a = pd.read_csv(b)
print(a.head(2), '\n', a.tail(3), '\n', a.shape, '\n', a.describe(), '\n', a.iloc[1:4], '\n',
      a[a['vegetation'] == 'chapparal'].head(2))

# 3.3.2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

b = 'https://raw.githubusercontent.com/akmand/datasets/master/iris.csv'
a = pd.read_csv(b)

print(a.head(5))
