
# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# loading the data
iris = load_iris()

# variables creation
X = scale(iris.data)
y = iris.target

# classes creation
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model creation
model = KMeans(n_clusters=3, random_state=0)


# training the model
model.fit(X_train)
# predictions
predictions = model.predict(X_test)

# label creation
labels = model.labels_
clustercount = np.bincount(labels)

for i in range(len(predictions)):
    print(classes[predictions[i]])



print('labels:', labels)
print('predictions:', predictions)
print('accuracy:', accuracy_score(y_test, predictions))
print('actual:', y_test)

# Visualising the clusters - On the first two columns
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y == 1, 0], X[y== 1, 1],
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y == 2, 0], X[y== 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1],
            s = 100, c = 'yellow', label = 'Centroids')

print(plt.show())