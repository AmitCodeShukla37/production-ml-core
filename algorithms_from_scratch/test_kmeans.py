import numpy as np
import pytest
from kmeans import MLKMeansScratch

def test_kmeans_convergence():
    np.random.seed(42)
    
    # Create 3 highly distinct vector clusters in 2D space
    cluster1 = np.random.randn(30, 2) + np.array([5, 5])
    cluster2 = np.random.randn(30, 2) + np.array([-5, -5])
    cluster3 = np.random.randn(30, 2) + np.array([5, -5])
    
    X = np.vstack((cluster1, cluster2, cluster3))
    
    # Initialize and fit
    model = MLKMeansScratch(n_clusters=3, max_iterations=50, tolerance=0.00001)
    labels = model.fit(X)
    
    # Verify centroids initialization and final grouping sizes
    assert model.centroids.shape == (3, 2)
    assert len(np.unique(labels)) == 3
