import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets

#Visualization libraries:
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='darkgrid')

import warnings
warnings.filterwarnings('ignore')

# Reading the data
data = pd.read_csv('Iris.csv')
# print(data.head())
#pairplot_df = sns.pairplot(data)
#pairplot_df.savefig('pairplot_df.png')

# Picking the numeric data from .csv K-means works on numeric data
selected_cols = data.iloc[:, [1, 2, 3, 4]].values

# Finding the optimum number of cluster with elbow method:
# Plotting Scree Plot to find optimum number of clusters:
#cluster_range = list(range(2, 12, 2))
wcss = []

for c in range(1, 11):
    kMeans = KMeans(init='k-means++', n_clusters=c, max_iter=300, n_init=10, random_state=0)
    kMeans.fit(selected_cols)
    wcss.append(kMeans.inertia_)

# Ploting the elbow method results
# plt.plot(range(1, 11), wcss)
# plt.title('The elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS') # within cluster sum of squares
# plt.show()

# Applying kmeans to the dataset / creating classifier
kMeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kMeans.fit_predict(selected_cols)

# Visualization of clusters:
plt.scatter(selected_cols[y_kmeans == 0,0], selected_cols[y_kmeans == 0, 1], s=100,
            c='red', label='Iris-setosa')
plt.scatter(selected_cols[y_kmeans == 1,0], selected_cols[y_kmeans == 1, 1], s=100,
            c='blue', label='Iris-versicolour')
plt.scatter(selected_cols[y_kmeans == 2,0], selected_cols[y_kmeans == 2, 1], s=100,
            c='green', label='Iris-virginica')

# Plotting centroids of cluster:
plt.scatter(kMeans.cluster_centers_[:, 0], kMeans.cluster_centers_[:, 1], s=100,
            c='yellow', label='Centroids')
plt.legend()
plt.show()


