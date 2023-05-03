import torch
import torch.nn as nn
import torch.nn.functional as F

# Define neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define first fully connected layer with 10 input neurons and 5 output neurons
        self.fc1 = nn.Linear(10, 5)
        # Define second fully connected layer with 5 input neurons and 2 output neurons
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        # Apply ReLU activation to the output of the first layer
        x = F.relu(self.fc1(x))
        # Pass the output of the first layer through the second layer
        x = self.fc2(x)
        # Return the final output
        return x

# Create an instance of the neural network
net = Net()

# Generate random input data for testing
input_data = torch.randn(1, 10)

# Pass the input data through the network to get a prediction
output = net(input_data)

# Print the output tensor
print(output)
