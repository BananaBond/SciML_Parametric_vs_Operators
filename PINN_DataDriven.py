import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle


# Import your NeuralNet class
from Common import NeuralNet, MultiVariatePoly




# DataDrivenModel class
class DataDrivenModel(nn.Module):
    def __init__(self, net):
        super(DataDrivenModel, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, inputs, outputs, verbose=True):
        predictions = self.forward(inputs)
        loss = torch.mean((predictions - outputs) ** 2)
        if verbose:
            print("Loss: ", round(loss.item(), 4))
        return loss


# Training function
def train(model, training_set, num_epochs, optimizer):
    for epoch in range(num_epochs):
        for inputs, outputs in training_set:
            inputs = torch.stack(inputs)
            outputs = torch.stack(outputs)
            optimizer.zero_grad()
            loss = model.compute_loss(inputs, outputs)
            loss.backward()
            optimizer.step()



# Initialize the NeuralNet and DataDrivenModel
net = NeuralNet(input_dimension=3, output_dimension=1, n_hidden_layers=4, neurons=20, regularization_param=0.,
                regularization_exp=2., retrain_seed=42)
model = DataDrivenModel(net)

# Load the training data

with open('inputs_parametric.pkl', 'rb') as f:
    inputs_parametric = pickle.load(f)
with open('outputs_parametric.pkl', 'rb') as f:
    outputs_parametric = pickle.load(f)


training_set = DataLoader(list(zip(inputs_parametric, outputs_parametric)), batch_size=32, shuffle=True)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(model, training_set, num_epochs=100, optimizer=optimizer)
