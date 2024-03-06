# Importing the Processing library
from processing import *

# Function to calculate the energy of the lattice
def lattice_energy(lattice, J=1.0):
    energy = 0
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            energy += -J * lattice[i][j] * (lattice[(i+1)%len(lattice)][j] + lattice[(i-1)%len(lattice)][j] + lattice[i][(j+1)%len(lattice)] + lattice[i][(j-1)%len(lattice)])
    return energy / 2

# Function to perform a Metropolis update on the lattice
def metropolis(lattice, T=1.0, J=1.0):
    new_lattice = [row[:] for row in lattice]  # Create a copy of the lattice

    i, j = int(random(len(new_lattice))), int(random(len(new_lattice)))  # Randomly select a position in the lattice
    new_lattice[i][j] *= -1  # Flip the spin at the selected position
    dE = lattice_energy(new_lattice, J) - lattice_energy(lattice, J)  # Calculate the change in energy

    if dE < 0 or random(1) < exp(-dE / T):  # Accept the new lattice configuration based on the Metropolis criterion
        return new_lattice  # Return the updated lattice
    return lattice  # Return the original lattice if the update is not accepted

# Function to initialize the lattice
def initialize_lattice(N):
    return [[choice([-1, 1]) for _ in range(N)] for _ in range(N)]

# Initialize lattice size
N = 50
lattice = initialize_lattice(N)

# Setup function
def setup():
    size(N, N)
    noLoop()

# Draw function
def draw():
    global lattice
    background(255)
    for i in range(N):
        for j in range(N):
            fill(0) if lattice[i][j] == -1 else fill(255)
            rect(i * N, j * N, N, N)

# Function to update the lattice for each frame of the animation
def update():
    global lattice
    lattice = metropolis(lattice, T=2.0)  # Adjust temperature as desired
    redraw()

# Run the sketch
run()
