import numpy as np
from numpy.random import rand
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import pandas as pd

import random

# # Define the differential equation
# def model(X, t, r, s, e, f, a):
#     dXdt = X * (r - s * X + np.sum(e * f * a * X) - np.sum(f.T * a.T * X))
#     return [dXdt[i] for i in range(len(X))]

# # Set parameters
# num_parameter_combinations = 5

# # Generate random parameter combinations for r and s
# parameter_combinations = np.random.rand(num_parameter_combinations, 2)

# # Set arrays for e, f, and a
# e = np.array([[1, 2], [3, 4]])
# f = np.array([[5, 6], [7, 8]])
# a = np.array([[9, 10], [11, 12]])

# # Set time points
# t = np.linspace(0, 100, 100)  # Adjust time range and steps as needed

# # Plot the results for each parameter combination
# for i in range(num_parameter_combinations):
#     r = parameter_combinations[i, 0]
#     s = parameter_combinations[i, 1]

#     # Generate a random initial value based on the dimensionality of the equation
#     dimensionality = len(e)
#     X0 = np.random.rand(dimensionality)

#     # Solve the differential equation
#     solution = odeint(model, X0, t, args=(r, s, e, f, a))

#     # Plot the solution
#     for j in range(dimensionality):
#         plt.plot(t, solution[:, j], label=f'X_{j+1}, r={r:.2f}, s={s:.2f}')

# plt.xlabel('Time')
# plt.ylabel('X_i')
# plt.legend()
# plt.show()


random.seed(42)

N = 10

A = [np.exp(random.random()) for i in range(N)]

init_conditions = {
    'X': [np.exp(random.random()) for i in range(N)],
    'A' : [i/(sum(A)) for i in A],
    'r': [random.random() for i in range(N)],
    's': [random.random() for i in range(N)],
    'f': [random.random() for i in range(N)],
}

random_matrix = pd.DataFrame(init_conditions)

Y0 = np.hstack([random_matrix['X'], ])

def dx(X, A, r, s,f ):

  self_reg = r - s * X
  pred = X @ (f * A)
  prey = X @ (f * A).T
  return X * (self_reg + pred - prey)


def dA(X, A, t):
    dA = np.zeros_like(A)
    # Vectorise this later
    for i in range(N):
        for j in range(N):
            direct = random_matrix['f'].iloc[i, j] * X[j]
            others = sum([random_matrix['A'].iloc[i, k] * random_matrix['f'].iloc[i, k] * X[k] for k in range(N)])
            dA[i, j] = random_matrix['r'].iloc[i] * A[i, j] * (direct - others)
    return dA

