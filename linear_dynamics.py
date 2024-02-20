import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

np.random.seed(42)

# Set the size of the matrix
N = 10

# Generate a random matrix with mean zero and standard deviation of 1
A = np.random.normal(0, 1, (N, N))

# Define the system dynamics as a linear function
def linear_dynamics(t, x):
    dxdt = np.dot(A, x)
    return dxdt

# Set the initial condition
initial_condition = np.random.rand(N)

# Set the time span for integration
t_span = (0, 10)

# Solve the system using solve_ivp
solution = solve_ivp(linear_dynamics, t_span, initial_condition, method='RK45')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y.T)
plt.title('Linear Dynamical System with Random Matrix')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.legend([f'State {i+1}' for i in range(N)])
plt.show()

eigenvalues = np.linalg.eigvals(A)

# Plot the eigenvalues on the complex plane
plt.figure(figsize=(8, 8))
plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='red', marker='o')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.title('Eigenvalues of the Matrix')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.show()

# Generate a random antisymmetric matrix
A = np.random.rand(N, N)
A = -A + A.T  # Ensure antisymmetry

# Compute the eigenvalues of the matrix
eigenvalues = np.linalg.eigvals(A)

# Plot the eigenvalues on the complex plane
plt.figure(figsize=(8, 8))
plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='blue', marker='o')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Eigenvalues of Antisymmetric Matrix')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.show()


# Set the initial condition
initial_condition = np.random.rand(N)

# Set the time span for integration
t_span = (0, 10)

# Solve the system using solve_ivp
solution = solve_ivp(linear_dynamics, t_span, initial_condition, method='RK45')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y.T)
plt.title('Dynamics of System with Antisymmetric Matrix')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.legend([f'State {i+1}' for i in range(N)])
plt.show()