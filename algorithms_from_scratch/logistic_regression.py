import numpy as np

class MLScratchLogisticRegression:                               # Binary Cross Entropy for Loss : -1/m Sum [y_pred * log(y) + (1 - y_pred) * log(1 - y)]
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    
    def compute_loss(self, y_true, y_predicted):
        m = y_true.shape[0]
        loss = -1/m * np.sum(y_true * np.log(y_predicted + 1e-15) + (1 - y_true) * np.log(1 - y_predicted + 1e-15))
        return loss
    
    def compute_gradients(self,X, y_true, y_predicted):
        dw = np.dot(X.T, (y_predicted - y_true)) / y_true.shape[0]              # dw = 1/m * X.T * (y_predicted - y_true)  -> shape of dw = (n_features, 1)
        db = np.sum(y_predicted - y_true) / y_true.shape[0]                     # db = 1/m * sum(y_predicted - y_true)  -> shape of db = (1,)
        return dw, db
    
    def fit(self, X, y):
        num_samples, num_features = X.shape            # Shape of data (number of samples, number of features)
        self.weights = np.zeros(num_features)          # (We have N features so N weights)  Z = W*X + b -> (W1*X1 + W2*X2 + ... + WN*XN + b)
        self.bias = 0.0
        
        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias  # Z = W*X + b
            y_predicted = self.sigmoid(linear_model)  


            if i % 100 == 0:
                # Binary Cross-Entropy Loss formula
                loss = self.compute_loss(y, y_predicted)
                print(f"Iteration {i}: Loss drops to -> {loss:.4f}")

            dw, db = self.compute_gradients(X, y, y_predicted)  # Compute gradients
            self.weights -= self.learning_rate * dw  # Update weights       
            self.bias -= self.learning_rate * db  # Update bias
        
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)