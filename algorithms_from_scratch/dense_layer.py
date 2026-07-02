import numpy as np

class MLScratchDenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros(output_dim)
        
    
    def forward(self, X):
        
        z = np.dot(X, self.weights) + self.bias
        
        if self.activation == 'relu':
            return np.maximum(z,0)
        else:
            return z