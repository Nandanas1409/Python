import numpy as np
def initialize_centroids(X,k) :
    indices = np.random.choice(len(X),size=k,replace=False)
    return X[indices]
def assign_clusters(X,centroids) :
    distances = np.linalg.norm(X[:,np.newaxis]-centroids,axis=2)
    return np.argmin(distances,axis=1)
def update_centroids(X,labels,k) :
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids
def has_converged(old_centroids,new_centroids) :
    return np.allclose(old_centroids,new_centroids)
def kmeans(X,k,max_iters = 100) :
    centroids = initialize_centroids(X,k)
    for i in range(max_iters) :
        labels = assign_clusters(X,centroids)
        new_centroids = update_centroids(X,labels,k)
        print(f"Epoch {i+1} : ")
        print("Centroids : ")
        for idx,c in enumerate(new_centroids):
            print(f"Cluster {idx} : {c}")
        print("Points in each cluster : ")
        for cluster_id in range(k):
            points_in_cluster = X[labels == cluster_id]
            print(f"Points for cluster_id {cluster_id}:")
            print(points_in_cluster if len(points_in_cluster) > 0 else "[]")
        if has_converged(centroids,new_centroids) :
            print(f"Converged after {i+1}iterations")
            break
        final_labels = assign_clusters(X, centroids)
        centroids = new_centroids
        return(centroids,labels)
def main() :
    k = int(input("Enter no. of clusters : "))
    n = int(input("Enter no. of cluster points : "))
    print("Enter cluster points(x,y) : ")
    points = []
    for i in range(n) :
        coord_str = input(f"Coord {i+1} : ")
        x,y = map(float,coord_str.strip().split(','))
        points.append([x,y])
    
    X = np.array(points)
    final_centroids, final_labels = kmeans(X,k)
    print("\n--- Final Result ---")
    print("Final Centroids:")
    for idx, c in enumerate(final_centroids):
        print(f"Cluster {idx}: {c}")

    print("\nFinal cluster assignments:")
    for cluster_id in range(k):
        points_in_cluster = X[final_labels == cluster_id]
        print(f"Cluster {cluster_id}:")
        print(points_in_cluster if len(points_in_cluster) > 0 else "[]")
if __name__ == "__main__" :
    main()
