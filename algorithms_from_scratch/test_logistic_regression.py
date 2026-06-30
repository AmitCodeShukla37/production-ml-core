import numpy as np
import pytest
from logistic_regression import MLScratchLogisticRegression

def test_logistic_regression_convergence():
    """
    Automated check that feeds synthetic matrix rows to your 
    NumPy class and validates gradient descent convergence.
    """
    # 1. Establish deterministic random matrices
    np.random.seed(42)
    num_samples = 100
    num_features = 2
    
    # 2. Build two cleanly separate coordinate clusters
    class_0 = np.random.randn(num_samples // 2, num_features) - 2
    class_1 = np.random.randn(num_samples // 2, num_features) + 2
    
    X = np.vstack((class_0, class_1))
    y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))

    # 3. Instantiate and run your logic
    model = MLScratchLogisticRegression(learning_rate=0.1, num_iterations=500)
    model.fit(X, y)
    
    # 4. Extract predictions and calculate performance accuracy
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    
    # 5. Core validation boundaries
    assert model.weights is not None, "Weight vector initialization failure."
    assert accuracy > 0.95, f"Vectorized math failed to converge. Accuracy: {accuracy}"
