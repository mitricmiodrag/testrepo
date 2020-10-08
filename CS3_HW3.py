#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:43:59 2020

@author: miodragmitric
"""

### already given
import numpy as np, random, scipy.stats as ss

def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode

def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]


### clean up data and make it only numerical
import pandas as pd
data = pd.read_csv ("/Users/miodragmitric/Documents/Python_for_Research/Case studies/KNN_predictions/asset-v1-HarvardX+PH526x+2T2019+type@asset+block@wine.csv", index_col=0)
numeric_data = data
numeric_data["is_red"] = (data["color"] == "red").astype(int)
numeric_data = numeric_data.drop(labels = ["color", "quality","high_quality"], axis = 1)

###preprocess data / sort of z score / so that each parameter equally affects the prediction.
###sort of a normalization
import sklearn.preprocessing
scaled_data = sklearn.preprocessing.scale(numeric_data)
numeric_data = pd.DataFrame(scaled_data,columns = numeric_data.columns)

import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components = 2)
principal_components = pca.fit_transform(numeric_data)
principal_components

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]
y = principal_components[:,1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

x = np.random.randint(0, 2, 1000)
y = np.random.randint(0 ,2, 1000)

def accuracy(predictions, outcomes):
    return np.mean(predictions==outcomes)*100
    # write your code here!
print(accuracy(x,y))

print(accuracy(0,data["high_quality"]))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)
print(accuracy(library_predictions,data["high_quality"]))

n_rows = data.shape[0]
random.seed(123)
selection = random.sample(range(n_rows), 10)
print(selection)


predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])

my_predictions = np.array([knn_predict(p,predictors[training_indices,:], outcomes[training_indices],k=5) for p in predictors[selection]])
percentage = accuracy(my_predictions, outcomes[selection])






