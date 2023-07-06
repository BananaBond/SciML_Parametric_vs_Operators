import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the domain
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)

# Set the parameters
t = 0.01  # final time
d = 2 # number of terms in the sum
mu = np.random.uniform(-1, 1, d)  # mu values are random numbers between -1 and 1, each time the solution looks a bit different



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

# Compute the initial condition
U0 = u0(X1, X2, mu)

# Compute the solution at the final time
U = u(t, X1, X2, mu)

# Compute the global minimum and maximum of the solution
U_min = min(U0.min(), U.min())
U_max = max(U0.max(), U.max())

# Plot the initial condition and the solution
fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(X1, X2, U0, cmap='viridis', vmin=U_min, vmax=U_max)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('T')
ax.set_title('Initial condition')
ax.set_zlim(U_min, U_max)

ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(X1, X2, U, cmap='viridis', vmin=U_min, vmax=U_max)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('T')
ax.set_title('Solution at t={}'.format(t))
ax.set_zlim(U_min, U_max)

plt.show()

# Compute and plot the solution at several time points
time_points = np.linspace(0, t, 5)  # 5 time points between 0 and T
fig, axs = plt.subplots(1, len(time_points), figsize=(15, 5), subplot_kw={'projection': '3d'})

for ax, t in zip(axs, time_points):
    U = u(t, X1, X2, mu)
    surf = ax.plot_surface(X1, X2, U, cmap='viridis', vmin=U_min, vmax=U_max)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('T')
    ax.set_title('Solution at t={:.3f}'.format(t))
    ax.set_zlim(U_min, U_max)

plt.tight_layout()
plt.show()