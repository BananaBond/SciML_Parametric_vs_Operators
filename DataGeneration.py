import numpy as np
import matplotlib.pyplot as plt

# Define the domain
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)

# soboleng points give a weired input/output solution
# num_points = 100
# Generate random Sobol sequence points
# soboleng = quasirandom.SobolEngine(dimension=2)
# sobol_points = soboleng.draw(num_points).numpy()

# Map Sobol sequence points to the desired range
# x1_samples = 2 * sobol_points[:, 0] - 1
# x2_samples = 2 * sobol_points[:, 1] - 1
# X1, X2 = np.meshgrid(x1_samples, x2_samples)

# Set the parameters
T = 1  # final time
d = 2  # number of terms in the sum
N_dataSamples = 10  # number of data samples


# Define the initial condition function
def u0(x1, x2, mu):
    u = 0
    for m in range(1, d + 1):
        u -= mu[m - 1] * np.sin(np.pi * m * x1) * np.sin(np.pi * m * x2) / np.sqrt(m)
    return u / d


# Define the solution function
def u(t, x1, x2, mu):
    u = 0
    for m in range(1, d + 1):
        u -= np.exp(-(np.pi * m) ** 2 * t) * mu[m - 1] * np.sin(np.pi * m * x1) * np.sin(np.pi * m * x2) / np.sqrt(m)
    return u / d


# Generate a set of mu values
mu_values = np.random.uniform(-1, 1, (N_dataSamples, d))

# Generate the data
data = []
for mu in mu_values:
    U0 = u0(X1, X2, mu)
    U = u(T, X1, X2, mu)
    data.append((U0, U, mu))

# Format the data for the parametric approach (spatial structure should inherently be captured by the PDE)
inputs_parametric = []
outputs_parametric = []
for U0, U, mu in data:
    for i in range(100):
        for j in range(100):
            inputs_parametric.append([X1[i, j], X2[i, j], mu.tolist()])
            outputs_parametric.append(U[i, j])

# Format the data for the operator approach
inputs_operator = []
outputs_operator = []
for U0, U, mu in data:
    inputs_operator.append(np.dstack([X1, X2, U0]))
    outputs_operator.append(np.expand_dims(U, axis=-1))

# Convert to numpy arrays
inputs_parametric_np = np.array(inputs_parametric)
outputs_parametric_np = np.array(outputs_parametric)

# Save the numpy arrays
np.save("inputs_parametric.npy", inputs_parametric_np)
np.save("outputs_parametric.npy", outputs_parametric_np)

# Assuming inputs_operator and outputs_operator are your data
np.save('inputs_operator.npy', inputs_operator)
np.save('outputs_operator.npy', outputs_operator)

# Sanity check: Plot the initial condition and the solution for the first set of mu values
U0, U, _ = data[0]

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
