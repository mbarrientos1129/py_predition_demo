from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
    elif iris['target'][i] == 1:
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

#Performing Classifications

#Dropping the target and species since only measurments ae needed
X = iris.drop(['target', 'species'], axis = 1)

#Covert into NumPy array and assign petal lenght and width
X = X.to_numpy() [:, (2,3)]
Y = iris['target']

#Split inot train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.5, random_state = 42)

log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

#Training Predictions
training_prediction = log_reg.predict(X_train)
test_prediction = log_reg.predict(X_test)

print(training_prediction)
print(test_prediction)
