import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to calculate the energy of the lattice
def lattice_energy(lattice, J=1.0):
    """
    Calculates the energy of the given lattice.

    Parameters:
    lattice (ndarray): The input lattice.
    J (float): The interaction strength parameter (default is 1.0).

    Returns:
    float: The energy of the lattice.
    """
    energy = 0
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            energy += -J * lattice[i, j] * (lattice[(i+1)%len(lattice), j] + lattice[(i-1)%len(lattice), j] + lattice[i, (j+1)%len(lattice)] + lattice[i, (j-1)%len(lattice)])
    return energy / 2

# Function to perform a Metropolis update on the lattice
def metropolis(lattice, T=1.0, J=1.0):
    """
    Performs a Metropolis update on the given lattice.

    Parameters:
    lattice (ndarray): The input lattice.
    T (float): The temperature parameter (default is 1.0).
    J (float): The interaction strength parameter (default is 1.0).

    Returns:
    ndarray: The updated lattice after the Metropolis update.
    """
    new_lattice = lattice.copy()  # Create a copy of the lattice

    i, j = np.random.randint(0, new_lattice.shape[0], 2)  # Randomly select a position in the lattice
    new_lattice[i, j] *= -1  # Flip the spin at the selected position
    dE = lattice_energy(new_lattice, J) - lattice_energy(lattice, J)  # Calculate the change in energy

    if dE < 0 or np.random.rand() < np.exp(-dE / T):  # Accept the new lattice configuration based on the Metropolis criterion
        return new_lattice  # Return the updated lattice
    return lattice  # Return the original lattice if the update is not accepted

# Function to initialize the lattice
def initialize_lattice(N):
    """
    Initializes a lattice with random spins.

    Parameters:
    N (int): The size of the lattice.

    Returns:
    ndarray: The initialized lattice.
    """
    return np.random.choice([-1, 1], size=(N, N))

# Create a lattice
N = 50
lattice = initialize_lattice(N)

# Set up the figure and axis
fig, ax = plt.subplots()
im = ax.imshow(lattice, cmap='binary', interpolation='nearest')

# Function to update the lattice for each frame of the animation
def update(frame):
    global lattice
    lattice = metropolis(lattice, T=2.0)  # Adjust temperature as desired
    im.set_array(lattice)
    return im,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()