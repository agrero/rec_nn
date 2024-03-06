import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# random seed
np.random.seed(42)

n = 100  # dim x dim for the matrix M
M = np.random.randn(n, n)

# M Setup
mean = np.mean(M)
M -= mean
std = np.std(M)
M /= std

# Update parameters
dt = 0.01
leakage = 0.01  # Leakage factor
threshold = 1

# weights 
w = np.random.uniform(0, 1, size = (n,n))

# gen plot
fig, ax = plt.subplots()
heatmap = ax.imshow(M, cmap='hot', interpolation='nearest')
plt.colorbar(heatmap)

def update(frame):
    global M
    # check spiking
    si = M.copy()
    si[si <= threshold] = 0
    
    # update
    dxi = np.sum(M) * dt  # Change in value
    noise = np.random.randn(n, n)  # Random noise
    # si = spiking mask, w = random weights from -1 to 1
    M += si * (dxi + noise - leakage) * w

    M[M >= threshold] = 0

    # set data
    heatmap.set_data(M)
    return heatmap,

ani = FuncAnimation(fig, update, frames=range(100), interval=50, blit=True)
plt.show()