# This script implements the Conjugate Gradient Method to solve a system of linear equations Ax = b.
# The method iteratively minimizes the quadratic function associated with the system, which corresponds 
# to finding the optimal solution by iterating over search directions.
# The solution trajectory is visualized on a contour plot of the quadratic form.
"""
    Conjugate Gradient Method for solving Ax = b.
    Arguments:
    - A: Coefficient matrix (n x n)
    - b: Right-hand side vector (n)
    - x0: Initial guess for the solution (n)
    - max_iter: Maximum number of iterations (default 100)
    - tol: Tolerance for the residual norm (default 1e-8)
    
    Returns:
    - x_history: History of the solution vectors at each iteration
    - residual_history: History of the residual norms at each iteration
    """
import numpy as np
import matplotlib.pyplot as plt

def conjugate_gradient(A, b, x0, max_iter=100, tol=1e-8):
   
    x = x0  # Initial guess
    r = b - np.dot(A, x)  # Initial residual
    p = r  # Initial search direction
    r_old = np.dot(r, r)  # Initial dot product for residual
    x_history = [x.copy()]  # Store the solution vector at each iteration
    residual_history = [np.linalg.norm(r)]  # Store residual norm at each iteration

    k = 0  # Iteration counter

    # Iterative process
    while k < max_iter:
        Ap = np.dot(A, p)
        alpha = r_old / np.dot(p, Ap)
        x = x + alpha * p
        x_history.append(x.copy())
        r = r - alpha * Ap
        residual_history.append(np.linalg.norm(r))
        r_new = np.dot(r, r)
        if np.sqrt(r_new) < tol:
            break
        beta = r_new / r_old
        p = r + beta * p
        r_old = r_new
        k += 1
    
    return np.array(x_history), residual_history

# Example usage: Solve a system using the Conjugate Gradient method
A = np.array([[4, 1], [1, 3]])  # Symmetric positive definite matrix
b = np.array([1, 2])  # Right-hand side vector
x0 = np.zeros_like(b)  # Initial guess (zero vector)

# Run the Conjugate Gradient method
x_history, residual_history = conjugate_gradient(A, b, x0, max_iter=100)

# Define the quadratic objective function for visualization
def quadratic_form(x, A, b):
    return 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)

# Generate a grid for contour plotting
x1_vals = np.linspace(-0.5, 1.5, 100)
x2_vals = np.linspace(-0.5, 1.5, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)

# Evaluate the quadratic form at each grid point
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i, j], X2[i, j]])
        Z[i, j] = quadratic_form(x, A, b)

# Contour plot with solution trajectory
plt.figure(figsize=(10, 8))
plt.contour(X1, X2, Z, levels=50, cmap="viridis")
x_hist = np.array(x_history)
plt.plot(x_hist[:, 0], x_hist[:, 1], marker="o", color="red", label="Solution trajectory")
plt.scatter(x_hist[:, 0], x_hist[:, 1], c="red")
plt.title("Conjugate Gradient: Contour Plot with Solution Trajectory")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.grid(True)
plt.show()

