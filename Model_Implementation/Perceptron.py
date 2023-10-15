import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preparation
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Choose Iris Setosa as the positive class (you can choose any class)
positive_class = 0

# Combine the other classes (Iris Versicolor and Iris Virginica) into the negative class
X_binary = X[y != positive_class, :]
y_binary = y[y != positive_class]

# Label encoding: Set the positive class to 1 and the negative class to 0
y_binary = (y_binary == positive_class).astype(int)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# Create Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.accuracy_history = []
        self.epochs = []

    def train(self, X, y):
        print(X)
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_epochs):
            for i in range(X.shape[0]):
                # Compute the output (prediction)
                prediction = self.predict(X[i])
                
                # Update the weights and bias -> y[i] - prediction = the error
                update = self.learning_rate * (y[i] - prediction)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, x):
        # Compute the weighted sum and apply a step function
        weighted_sum = np.dot(self.weights, x) + self.bias
        return 1 if weighted_sum >= 0 else 0
        

# Initialize model and define loss function and optimizer
model = Perceptron()

# Train the Model
model.train(X_train, y_train)

# Make predictions on the test data
predictions = [model.predict(x) for x in X_test]

# Visualise the data 
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette="Dark2", s=70, marker='x')

# Plot the decision boundary (a straight line)
print(model.weights)
w1, w2, w3, w4 = model.weights  # Assuming you have the Perceptron's weights
b = model.bias  # Assuming you have the Perceptron's bias
x_decision = np.linspace(4, 7.5, 10)  # Adjust the range based on your data
y_decision = (-w2 * x_decision - b) / w3
plt.plot(x_decision, y_decision, color='black', linestyle='--', label='Decision Boundary')

plt.title("Scatter Plot of Iris Data (First Two Classes)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title="Class", loc="upper right", labels=["Setosa", "Versicolor"])
plt.show()

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print("Test Accuracy:", accuracy)