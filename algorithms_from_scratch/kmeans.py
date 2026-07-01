import numpy as np

class MLKMeansScratch:
    def __init__(self, n_clusters=3, max_iterations=100, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        
    def compute_distances(self, X):
        distances = np.linalg.norm(X[:,np.newaxis] - self.centroids, axis=2)
        return distances
        
    def fit(self,X):
        num_samples, num_features = X.shape

        random_indexes = np.random.choice(num_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indexes]
        
        for i in range(self.max_iterations):
            distances = self.compute_distances(X)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                for j in range(self.n_clusters)
            ])
            
            centroid_shift = np.sum(np.linalg.norm(self.centroids - new_centroids, axis=1))
            self.centroids = new_centroids
            
            if centroid_shift < self.tolerance:
                break
        
        return labels
        
    
    def predict(self, X):
        distances = self.compute_distances(X)
        return np.argmin(distances, axis=1)