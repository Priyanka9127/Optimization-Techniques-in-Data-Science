"""
This code applies gradient descent to minimize the Rosenbrock function and visualizes the optimization process
with 2D contour and 3D surface plots. It also creates an animation of the optimization path.

Key components:
1. Rosenbrock function and its gradient calculation.
2. Step size determination via backtracking line search.
3. Gradient descent algorithm with convergence check.
4. Visualization with 2D and 3D plots and animation of the optimization path.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm
from matplotlib.animation import FuncAnimation

# Rosenbrock function
def func(x):
    return 20 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Gradient of the Rosenbrock function using central difference
def grad(x):
    h = 1e-6
    grad_value = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += h
        x_backward[i] -= h
        grad_value[i] = (func(x_forward) - func(x_backward)) / (2 * h)
    return grad_value

# Step size calculation using Backtracking line search
def step_size(x, grad_x, alpha=1, beta=0.8):
    func_x = func(x)
    while func(x - alpha * grad_x) > func_x - 0.5 * alpha * norm(grad_x)**2:
        alpha *= beta
    return alpha

# Gradient Descent Algorithm
def gradient_descent(x_init, tol=1e-7):
    x = np.array(x_init, dtype=float)
    path_x, path_y, path_z = [x[0]], [x[1]], [func(x)]
    prev_func_val = func(x)
    iterations = 0
    
    while True:
        grad_x = grad(x)
        alpha = step_size(x, grad_x)
        x -= alpha * grad_x
        current_func_val = func(x)
        path_x.append(x[0])
        path_y.append(x[1])
        path_z.append(current_func_val)  # Store the function value for 3D plotting
        iterations += 1
        
        # Check for convergence
        if abs(current_func_val - prev_func_val) < tol:
            break
        prev_func_val = current_func_val

    return x, current_func_val, iterations, path_x, path_y, path_z

# Initial guess and running the gradient descent
x_init = [0, 0]
optimal_x, optimal_value, total_iterations, path_x, path_y, path_z = gradient_descent(x_init)

# Generate data for plotting
x_vals = np.linspace(-0.5, 1.2, 100)
y_vals = np.linspace(-0.5, 1.2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 20 * (Y - X**2)**2 + (1 - X)**2

# Set up the figure for side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 2D Contour Plot
contour = ax1.contour(X, Y, Z, levels=100, cmap='viridis')
ax1.set_title("2D Contour Plot of Rosenbrock Function")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
fig.colorbar(contour, ax=ax1, label="Function Value")
scat2d = ax1.plot([], [], 'o-', color="blue", label="Optimization Path")[0]

# 3D Surface Plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
ax2.set_title("3D Surface Plot of Rosenbrock Function")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")
ax2.set_zlabel("$f(x, y)$")
scat3d = ax2.scatter([], [], [], color='blue', marker='o', label="Optimization Path")
final_point3d = ax2.scatter([optimal_x[0]], [optimal_x[1]], [optimal_value], color='red', s=100, label="Final Point")

# Animation function for updating both plots
def update(frame):
    # Update 2D plot
    scat2d.set_data(path_x[:frame+1], path_y[:frame+1])
    # Update 3D plot
    scat3d._offsets3d = (path_x[:frame+1], path_y[:frame+1], path_z[:frame+1])
    return scat2d, scat3d

# Create animation
ani = FuncAnimation(fig, update, frames=len(path_x), blit=False, repeat=False)

# Show the plots
plt.legend()
plt.show()

