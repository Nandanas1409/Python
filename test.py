import numpy as np
def initialize_centroids(X,k):
    indices = np.random.choice(len(X),size=k,replace=True)
    return X[indices]
def assign_clusters(X,centroids):
    distance = np.linalg.norm(X[:,np.newaxis]-centroids,axis=2)
    return np.argmin(distance,axis=1)
def update_centroids(X,labels,k):
    new_centroids = np.array(X[labels==i].mean(axis=0) for i in range(k))
    return new_centroids
def has_converged(old_centroids,new_centroids) :
    np.allclose(old_centroids,new_centroids)
def kmeans(X,k) :
    centroids = initialize_centroids(X,k)
    labels = assign_clusters(X,centroids)
    new_centroids = update_centroids(X,labels,k)

    print(f"Epoch {i+1} : ")
    print("Centorids : ")
    for idx,c in enumerate(centroids) :
        print(f"Cluster {idx} : {c}")
    print("Points in each cluster : ")
    for cluster_id in range(k) :
        points_in_cluster = X[labels==cluster_id]
        print(f"Point of {cluster_id} : ")
        print(points_in_cluster if len(points_in_cluster) > 0 else "[]")

    if has_converged(centroids,new_centroids) :
        print("Clustreing completed in ",i+1,"th iteration")
