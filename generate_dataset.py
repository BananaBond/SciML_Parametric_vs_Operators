import numpy as np
from torch import quasirandom

# Set the parameters
t = 1  # final time
d = 2  # number of terms in the sum
num_points = 100 # number of points in the domain

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

number_of_sets = 64

input_mu = []
input    = []
output   = []

for i in range(int(number_of_sets)):

    soboleng = quasirandom.SobolEngine(dimension=2)
    sobol_points = soboleng.draw(num_points).numpy()

    x1 = sobol_points[:,0] * 2 - 1  # Rescale the first dimension to match the range (-1, 1)
    x2 = sobol_points[:,1] * 2 - 1  # Rescale the second dimension to match the range (-1, 1)

    X1, X2 = np.meshgrid(x1, x2)

    mu = np.random.random(d)*2. - 1.  # set all parameter values to a random value
    input_mu.append(mu)
    input.append(np.array([X1, X2, u0(X1, X2, mu)]).transpose(1, 2, 0))
    output.append(u(t, X1, X2, mu))

np.save("input_mu", input_mu)
np.save("input", input)
np.save("output", output)