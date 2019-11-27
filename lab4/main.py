from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Step 1: Method to import iris dataset
def importIris():

    iris = datasets.load_iris()

    return iris

# Step 2: Method to perform kmeans clustering
def KMeansCluster(data, clusters):

    # print(clusters)

    # Do the kmeans cluster with the number of cluseters
    # specified by clusters
    # With random_state as 8687 so the data is always made
    # same way
    kmeans = KMeans(n_clusters=clusters, random_state=8697)

    # Fit the data with the iris data set
    kmeans = kmeans.fit(data.data)
    #print(kmeans.labels_)
    #print(kmeans.cluster_centers_)
    return kmeans

# Step 3: Method to plot number of data points within clusters number of clusters
def withinClusters(kmeansList, iris):

    within = []
    clusterIndex = []
    # For each of the KMean results in the kmeans list
    for kmeans in kmeansList:

        # Add the inertia_ to the inertia_ list
        # As this is the number of data in clusters
        within.append(kmeans.inertia_)

        # Add the number of clusters to the cluster index
        clusterIndex.append(kmeans.n_clusters)

    # Create a scatter plot of the number of clusters against
    # the mean number of values inside the clusters
    plt.scatter(clusterIndex, within)
    plt.title("Within Clusters")
    plt.show()

# Step 4: Method to plot plot the number of data points outside cluseters against the number of clusters
def betweenCluster(kmeansList, iris):

    between = []
    clusterIndex = []
    # For each of the KMean results in the kmeans list
    for kmeans in kmeansList:


        # Add the size of the data set - the inertia_ to the inertia_ list
        # As this is the number of data in clusters
        between.append(len(iris.data) - kmeans.inertia_)
        clusterIndex.append(kmeans.n_clusters)

    # Create a scatter plot of the number of clusters against
    # the mean number of values outside the clusters
    plt.scatter(clusterIndex, between)
    plt.title("Between Clusters")
    plt.show()

# Step 5: Method to plot the Calinski-Herbasz against the number of clusters
def calinskiHerbasz(clusterList, iris, algType):

    chIndex = []
    clusterIndex = []
    # For each of the cluster results in the clusters list
    for cluster in clusterList:

        lables = cluster.labels_

        # Add the ch score of the kmeans result to a list
        chIndex.append(metrics.calinski_harabasz_score(iris.data, lables))
        clusterIndex.append(cluster.n_clusters)

    # Create a scatter plot of the number of clusters against
    # the ch values
    plt.scatter(clusterIndex, chIndex)
    plt.title("Calinski-Herbasz " + algType)
    plt.show()

'''
Step 6
The natural cluster arrangment is 3. This is because it has the
highest Calinski-Herbasz index compared to the other results.

This makes sense as there are 3 different categories represented
in the iris data set.
'''

# Step 7: Method to perform Hierarchical Clustering
def hierarchicalCluster(iris, clusters):

    clustering = AgglomerativeClustering(distance_threshold=None, n_clusters=clusters)
    clustering = clustering.fit(iris.data)
    return clustering

'''
Step 8
Similar to KMeans we can see that the natural arrangment is also 3 as it has the greatest
CH index. However with Hierarchical Clustering the difference between the best value
and all other values is greater than it was using KMeans.

This also makes sense as there are 3 different categories represented in the iris data set
'''

def main():

    iris = importIris()

    kmeansList = []
    for i in range(2, 11):
        kmeansList.append(KMeansCluster(iris, i))

    withinClusters(kmeansList, iris)
    betweenCluster(kmeansList, iris)
    calinskiHerbasz(kmeansList, iris, "KMeans Cluster")

    hierarchicalClusterList = []
    for i in range(2, 11):
        hierarchicalClusterList.append(hierarchicalCluster(iris, i))
    calinskiHerbasz(hierarchicalClusterList, iris, "Hierarchical Cluster")


main()
