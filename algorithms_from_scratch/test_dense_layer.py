import numpy as np
import pytest
from dense_layer import MLScratchDenseLayer

def test_dense_layer_forward_and_activation():
    np.random.seed(42)
    
    # 5 samples, 4 input features
    X = np.random.randn(5, 4)
    
    # Layer with 4 inputs and 3 output units using ReLU
    layer = MLScratchDenseLayer(input_dim=4, output_dim=3, activation='relu')
    
    output = layer.forward(X)
    
    # Check shape transformation: (5, 4) -> (5, 3)
    assert output.shape == (5, 3)
    
    # Core ReLU Check: No value in the final activation output can be negative!
    assert np.all(output >= 0.0)
