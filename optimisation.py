import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Rosenbrock function
def f(x, y):
    """
    Rosenbrock function: f(x, y) = 100(y - x^2)^2 + (1 - x)^2
    Global minimum at (1, 1) with f(1, 1) = 0
    """
    return 100 * (y - x**2)**2 + (1 - x)**2

### TASK 1: Visualize the Rosenbrock function
# Create a grid of x and y values
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)

# Calculate function values
Z = f(X, Y)

# Known minimum
x_min, y_min = 1, 1
z_min = f(x_min, y_min)

# Create figure with subplots
fig = plt.figure(figsize=(16, 6))

# 2D Contour Plot
ax1 = fig.add_subplot(121)
contour = ax1.contour(X, Y, Z, levels=50, cmap='viridis')
contourf = ax1.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
ax1.plot(x_min, y_min, 'r*', markersize=20, label=f'Minimum at ({x_min}, {y_min})')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('2D Contour Plot of Rosenbrock Function', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
fig.colorbar(contourf, ax=ax1, label='f(x, y)')

# 3D Surface Plot
ax2 = fig.add_subplot(122, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
ax2.scatter([x_min], [y_min], [z_min], color='red', s=100, marker='*', 
            label=f'Minimum at ({x_min}, {y_min}, {z_min})')
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_zlabel('f(x, y)', fontsize=10)
ax2.set_title('3D Surface Plot of Rosenbrock Function', fontsize=14)
ax2.legend(fontsize=9)
# Set view angle for better visualization
ax2.view_init(elev=25, azim=30)
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='f(x, y)')

plt.tight_layout()
plt.show()

# Print information about the minimum
print(f"Function: f(x, y) = 100(y - x²)² + (1 - x)²")
print(f"Global minimum located at: ({x_min}, {y_min})")
print(f"Function value at minimum: f({x_min}, {y_min}) = {z_min}")
