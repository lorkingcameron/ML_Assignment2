# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Keep the first two classes (Setosa and Versicolor)
X = X[:100]
y = y[:100]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
    
    def fit(self, X, y, learning_rate, num_epochs):
        # Training the logistic regression model with manual gradient computation
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self(X)
            logits = nn.functional.softmax(outputs, dim=1)

            # Calculate loss manually
            loss = -torch.log(logits[range(len(y)), y]).mean()

            # Manual gradient computation
            dL_dy = logits
            dL_dy[range(len(y)), y] -= 1
            dL_dy /= len(y)

            # Backward pass (compute gradients)
            self.zero_grad()

                # Manually compute gradients for the weight and bias
            grad_weight = torch.mm(X.t(), dL_dy)
            grad_bias = dL_dy.sum(dim=0)

                # Update model parameters manually
            with torch.no_grad():
                self.linear.weight -= learning_rate * grad_weight.t()
                self.linear.bias -= learning_rate * grad_bias

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def predict(self, data):
        with torch.no_grad():
            outputs = self(data)
            predicted = torch.max(outputs, 1)[1]
            return predicted

# Initialize the model and choose hyperparameters
input_size = 4  # Number of features in the Iris dataset
num_classes = 2  # Number of classes
learning_rate = 0.01
num_epochs = 1000

model = LogisticRegression(input_size, num_classes)

# Train the model
model.fit(X_train, y_train, learning_rate, num_epochs)

# Test the model
predicted = model.predict(X_test)
accuracy = (predicted == y_test).sum().item() / y_test.size(0)
print(f'Accuracy on the test set: {accuracy:.4f}')

# Save the model if needed
# torch.save(model.state_dict(), 'logistic_regression_model.ckpt')