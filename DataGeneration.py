import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch


from mpl_toolkits.mplot3d import Axes3D

# Define the domain
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
N_dataSamples = 100

# Set the parameters
T = 1  # final time
d = 2 # number of terms in the sum

# Define the initial condition function
def u0(x1, x2, mu):
    u = 0
    for m in range(1, d+1):
        u -= mu[m-1] * np.sin(np.pi * m * x1) * np.sin(np.pi * m * x2) / np.sqrt(m)
    return u / d

# Define the solution function
def u(t, x1, x2, mu):
    u = 0
    for m in range(1, d+1):
        u -= np.exp(-(np.pi * m) ** 2 * t) * mu[m-1] * np.sin(np.pi * m * x1) * np.sin(np.pi * m * x2) / np.sqrt(m)
    return u / d

# Generate a set of mu values
mu_values = np.random.uniform(-1, 1, (N_dataSamples, d)) # 100 sets of d mu values

# Convert mu_values to a list of lists
mu_values_list = mu_values.tolist()

# Generate the data
data = []
for mu in mu_values:
    U0 = u0(X1, X2, mu)
    U = u(T, X1, X2, mu)
    data.append((U0, U, mu))

# Format the data for the parametric approach
inputs_parametric = []
outputs_parametric = []
for U0, U, mu in data:
    for i in range(100):
        for j in range(100):
            inputs_parametric.append([X1[i, j], X2[i, j], mu])
            outputs_parametric.append(U[i, j])


# Format the data for the operator approach
inputs_operator = [U0 for U0, U in data]
outputs_operator = [U for U0, U in data]


# Save the data for the parametric approach
with open('inputs_parametric.pkl', 'wb') as f:
    pickle.dump(inputs_parametric, f)
with open('outputs_parametric.pkl', 'wb') as f:
    pickle.dump(outputs_parametric, f)

# Save the data for the operator approach
with open('inputs_operator.pkl', 'wb') as f:
    pickle.dump(inputs_operator, f)
with open('outputs_operator.pkl', 'wb') as f:
    pickle.dump(outputs_operator, f)



# Plot the initial condition and the solution for the first set of mu values
U0, U = data[0]

fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(X1, X2, U0, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('U0')
ax.set_title('Initial condition')

ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(X1, X2, U, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('U')
ax.set_title('Solution at t={}'.format(T))

plt.show()


# Load the training data
with open('inputs_parametric.pkl', 'rb') as f:
    inputs_parametric = pickle.load(f)
with open('outputs_parametric.pkl', 'rb') as f:
    outputs_parametric = pickle.load(f)

# Convert to tensors and repeat mu for each data point
inputs_parametric = [torch.tensor([x1, x2] + mu.tolist() * 10000) for x1, x2, mu in inputs_parametric]
outputs_parametric = [torch.tensor(U).repeat(10000) for U in outputs_parametric]

# Create a TensorDataset and DataLoader
dataset = torch.utils.data.TensorDataset(inputs_parametric, outputs_parametric)
training_set = DataLoader(dataset, batch_size=32, shuffle=True)
