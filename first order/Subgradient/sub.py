# This script implements the Subgradient Optimization Algorithm to minimize the Rosenbrock function.
# The Subgradient optimization method is used for functions that are not differentiable everywhere.
# The optimizer uses a learning rate that decays over iterations for better convergence.
# The stopping criteria for optimization are either reaching a gradient norm smaller than EPSILON or hitting the maximum number of iterations.
# The algorithm tracks the best function value (f_best) and updates the current position iteratively.
# The script generates both 2D and 3D visualizations of the optimization path, with animation for a dynamic representation.
# The 2D plot uses a contour plot to visualize the function, and the 3D plot uses a surface plot for better representation.
# An animation is created to show the progress of the optimizer across iterations.

# Import necessary libraries
# numpy is used for numerical operations.
# matplotlib is used for plotting 2D and 3D visualizations of the function and optimization path.
    """
    Implementation of the Subgradient Optimization Algorithm with improvements for faster convergence.
    
    - alpha: The initial learning rate for each iteration (default is 0.1).
    - alpha_decay: The decay rate for the learning rate (default is 0.95).
    - f_best: Holds the best function value found during optimization.
    - K: A counter for maximum iterations (default is 500).
    
    Key Methods:
    - _reset(): Resets the learning rate, counter K, and iteration count at the end of optimization.
    - _next(): Updates the position using the subgradient method. Computes the new position based on the gradient and updates f_best if the new function value is better.
    - optimize(): Iteratively updates `x` using the subgradient method until the gradient norm is smaller than a threshold (EPSILON) or the maximum iterations (K) are reached.
    """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

EPSILON = 0.0001

class Subgradient:

    def __init__(self, alpha: float = 0.1, alpha_decay: float = 0.95) -> None:
        
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.f_best: float | None = None  # Will hold the best function value
        self.K = 500  # Maximum iterations
        self.num_iter = 0  # Iteration counter
        return

    def _reset(self) -> None:
        """
        Reset alpha, K, and iteration count for a new optimization.
        """
        self.alpha = 0.1
        self.K = 500
        self.num_iter = 0  # Reset iteration counter
        return

    def _next(self, x: np.ndarray, func_callback, grad_func_callback) -> np.ndarray:
     
        grad = grad_func_callback(x)
        x_new = x - self.alpha * grad / np.linalg.norm(grad)

        # Check if f_best needs initialization or update
        if self.f_best is None or func_callback(x_new) < self.f_best:
            self.f_best = func_callback(x_new)
            return x_new

        # Decrease iteration counter and learning rate if no improvement
        self.K -= 1
        self.alpha *= self.alpha_decay
        return x

    def optimize(
        self,
        x: np.ndarray,
        func_callback,
        grad_func_callback,
        is_plot: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    
        plot_points: list[np.ndarray] = [x]
        self.f_best = func_callback(x)

        # Iterate to perform subgradient optimization
        while np.linalg.norm(grad_func_callback(x)) > EPSILON and self.K > 0:
            self.num_iter += 1
            # Position Update Equation of the Subgradient Algorithm
            x = self._next(x, func_callback, grad_func_callback)
            if is_plot:
                plot_points.append(x)

        # Reset the parameters after optimization
        self._reset()
        if is_plot:
            return x, plot_points
        return x

# Rosenbrock function (benchmark function for optimization)
# The Rosenbrock function is commonly used to test optimization algorithms.
# It has a global minimum at (1, 1) where the value is 0.
rosenbrock = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
rosenbrock_grad = lambda x: np.array([
    -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
    200 * (x[1] - x[0]**2)
])

# Main program to run the Subgradient optimizer
if __name__ == "__main__":
    # Set an initial point that is farther from the optimal solution (1, 1)
    x0 = np.array([-1.2, 1.0])

    # Initialize and run Subgradient optimization
    optimizer = Subgradient(alpha=0.1, alpha_decay=0.95)
    result, path = optimizer.optimize(x=x0, func_callback=rosenbrock, grad_func_callback=rosenbrock_grad, is_plot=True)

    # Generate mesh grid for the Rosenbrock function
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    # Set up figure and subplots for 2D and 3D plots
    fig = plt.figure(figsize=(14, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    # Set up 2D contour plot to visualize the Rosenbrock function
    ax2d.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    # Extract x and y values from the path for plotting
    x_vals = [point[0] for point in path]
    y_vals = [point[1] for point in path]
    line_2d, = ax2d.plot(x_vals, y_vals, 'o-', color='red', label='Subgradient Path')
    goal_point_2d, = ax2d.plot(result[0], result[1], 'go', label='Final Point')
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title('2D Path of Subgradient Optimization')
    ax2d.legend()
    text_2d = ax2d.text(-1.5, 2.8, '', fontsize=12)

    # Set up 3D surface plot for the Rosenbrock function
    ax3d.plot_surface(X, Y, Z, cmap='jet', alpha=0.6, edgecolor='none')
    path_z = (1 - np.array(x_vals))**2 + 100 * (np.array(y_vals) - np.array(x_vals)**2)**2
    line_3d, = ax3d.plot(x_vals, y_vals, path_z, 'o-', color='red', label='Subgradient Path')
    goal_point_3d = ax3d.scatter(result[0], result[1], (1 - result[0])**2 + 100 * (result[1] - result[0]**2)**2, color='green', s=50, label='Final Point')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z (Rosenbrock)')
    ax3d.set_title('3D Path of Subgradient Optimization')
    ax3d.view_init(elev=30, azim=120)
    ax3d.legend()
    text_3d = ax3d.text2D(0.05, 0.9, '', transform=ax3d.transAxes, fontsize=12)

    # Animation update function for visualization
    def update(i):
        # Update 2D plot
        line_2d.set_data(x_vals[:i+1], y_vals[:i+1])
        goal_point_2d.set_data(result[0], result[1])
        text_2d.set_text(f"Iter: {i}")
        
        # Update 3D plot
        line_3d.set_data(x_vals[:i+1], y_vals[:i+1], path_z[:i+1])
        text_3d.set_text(f"Iter: {i}")

    # Animation for showing optimization progress
    ani = FuncAnimation(fig, update, frames=len(path), interval=100, repeat=False)
    plt.show()

