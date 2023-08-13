from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#Plotting
setosa = iris[iris.species == "setosa"]
versicolor = iris[iris.species == 'versicolor']
virginica = iris[iris.species == 'virginica']

fig, ax = plt.subplots()
fig.set_size_inches(13, 7) #adjust lenght and width of plot 

#labels and scatter points 
ax.scatter(setosa['petal length (cm)'], setosa['petal width (cm)'], label = "Setosa", facecolor="blue")
ax.scatter(versicolor['petal length (cm)'], setosa['petal width (cm)'], label = "Versicolor", facecolor="green")
ax.scatter(virginica['petal length (cm)'], setosa['petal width (cm)'], label = "Virginica", facecolor="red")

ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.grid()
ax.set_title("Iris Petals")
ax.legend()

plt.show()