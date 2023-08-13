from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris() #loading the dataset
iris.keys()

irisDict = (['data','target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

iris = pd.DataFrame (
    data = np.c_[iris['data'], iris['target']],
    columns = iris['feature_names'] + ['target']
)

print(iris.head(10))

species = []
for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] ==1:
        species.append('versicolor')
    else:
        species.append('virginica')
iris['species'] = species

print(iris.groupby('species').size())
print(iris.describe())
