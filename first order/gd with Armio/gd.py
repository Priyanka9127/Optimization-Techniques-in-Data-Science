# This script uses Gradient Descent with the Armijo Rule for step size selection to minimize the Rosenbrock function.
# The Rosenbrock function, often used in optimization tests, is given by:
# f(x, y) = 100 * (y - x^2)^2 + (1 - x)^2
# The Armijo Rule is used to select an optimal step size during each iteration of the gradient descent algorithm.
# The script performs the following:
# 1. Defines the Rosenbrock function and its gradient.
# 2. Implements the Armijo Rule to adaptively adjust the step size.
# 3. Applies Gradient Descent with Armijo Rule to minimize the function starting from an initial guess.
# 4. Plots the optimization process using both 2D contour and 3D surface plots.
# 5. Animates the descent path over iterations, showing how the algorithm converges to the minimum.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def gradient(xy):
    x, y = xy
    grad_x = -400 * x * (y - x**2) - 2 * (1 - x)
    grad_y = 200 * (y - x**2)
    return np.array([grad_x, grad_y])

def armijo_rule(x, d, grad, func, beta=0.5, sigma=1e-4):
    alpha = 1.0
    while func(x[0] + alpha * d[0], x[1] + alpha * d[1]) > func(x[0], x[1]) + sigma * alpha * np.dot(grad, d):
        alpha *= beta  # Reduce alpha if the Armijo condition is not satisfied
    return alpha

def gradient_descent_armijo(x_init, func, max_iters=1000, tol=1e-6):
    path = [x_init]
    x = x_init
    for i in range(max_iters):
        grad = gradient(x)  # Compute the gradient at the current point
        d = -grad  # Descent direction (negative of gradient)
        alpha = armijo_rule(x, d, grad, func)  # Compute step size using Armijo rule
        x_new = x + alpha * d  # Update the point
        path.append(x_new)

        # Print values at each iteration
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {func(x_new[0], x_new[1])}")

        # Convergence check: If the change in x is small enough, stop the algorithm
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(path)

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

x_init = np.array([-1.2, 1.0])

path = gradient_descent_armijo(x_init, rosenbrock)

fig = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(121)
ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
ax1.set_title('Gradient Descent with Armijo Rule on Rosenbrock Function (2D)')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='jet', alpha=0.7)
ax2.set_title('Gradient Descent with Armijo Rule on Rosenbrock Function (3D)')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_zlabel('$f(x_1, x_2)$')

for i in range(len(path)):
    ax1.clear()
    ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    ax1.plot(path[:i+1, 0], path[:i+1, 1], 'ro-', markersize=4, label='Gradient Descent Path')
    ax1.set_title('Gradient Descent with Armijo Rule on Rosenbrock Function (2D)')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.legend()

    ax2.clear()
    ax2.plot_surface(X, Y, Z, cmap='jet', alpha=0.7)
    ax2.plot(path[:i+1, 0], path[:i+1, 1], [rosenbrock(x[0], x[1]) for x in path[:i+1]], 'ro-', markersize=4, label='Gradient Descent Path')
    ax2.set_title('Gradient Descent with Armijo Rule on Rosenbrock Function (3D)')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlabel('$f(x_1, x_2)$')
    ax2.legend()

    plt.pause(0.1)

plt.tight_layout()
plt.show()

