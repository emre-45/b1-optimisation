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

### TASK 2: Implement algorithms to find the minimum

# Gradient of the Rosenbrock function
def gradient(x, y):
    """
    Gradient of f(x, y) = 100(y - x^2)^2 + (1 - x)^2
    ∇f = [∂f/∂x, ∂f/∂y]
    ∂f/∂x = -400x(y - x^2) - 2(1 - x)
    ∂f/∂y = 200(y - x^2)
    """
    df_dx = -400 * x * (y - x**2) - 2 * (1 - x)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

# Hessian of the Rosenbrock function
def hessian(x, y):
    """
    Hessian matrix of f(x, y) = 100(y - x^2)^2 + (1 - x)^2
    H = [[∂²f/∂x², ∂²f/∂x∂y],
         [∂²f/∂y∂x, ∂²f/∂y²]]
    ∂²f/∂x² = -400(y - 3x^2) + 2
    ∂²f/∂x∂y = -400x
    ∂²f/∂y² = 200
    """
    d2f_dx2 = -400 * (y - 3 * x**2) + 2
    d2f_dxdy = -400 * x
    d2f_dy2 = 200
    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])

# Gradient Descent
def gradient_descent(x0, y0, learning_rate=0.001, max_iter=10000, tol=1e-6):
    """
    Gradient Descent optimization
    """
    path = [(x0, y0)]
    x, y = x0, y0
    
    for i in range(max_iter):
        grad = gradient(x, y)
        
        # Update parameters
        x_new = x - learning_rate * grad[0]
        y_new = y - learning_rate * grad[1]
        
        path.append((x_new, y_new))
        
        # Check convergence
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        
        x, y = x_new, y_new
    
    return np.array(path), i + 1

# Newton's Method
def newton_method(x0, y0, max_iter=100, tol=1e-6):
    """
    Newton's Method optimization
    """
    path = [(x0, y0)]
    x, y = x0, y0
    
    for i in range(max_iter):
        grad = gradient(x, y)
        hess = hessian(x, y)
        
        # Check if Hessian is singular
        try:
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            print("Singular Hessian matrix encountered")
            break
        
        # Update parameters
        x_new = x + delta[0]
        y_new = y + delta[1]
        
        path.append((x_new, y_new))
        
        # Check convergence
        if np.linalg.norm(delta) < tol:
            break
        
        x, y = x_new, y_new
    
    return np.array(path), i + 1

# Gauss-Newton Method
def gauss_newton(x0, y0, max_iter=100, tol=1e-6):
    """
    Gauss-Newton Method optimization
    For Rosenbrock: f(x,y) = 100(y - x^2)^2 + (1 - x)^2
    We can write it as sum of squares: r1^2 + r2^2 where
    r1 = 10(y - x^2)
    r2 = (1 - x)
    """
    path = [(x0, y0)]
    x, y = x0, y0
    
    for i in range(max_iter):
        # Residuals
        r1 = 10 * (y - x**2)
        r2 = 1 - x
        r = np.array([r1, r2])
        
        # Jacobian of residuals
        # dr1/dx = -20x, dr1/dy = 10
        # dr2/dx = -1, dr2/dy = 0
        J = np.array([[-20*x, 10],
                      [-1, 0]])
        
        # Gauss-Newton approximation: H ≈ J^T J
        JTJ = J.T @ J
        JTr = J.T @ r
        
        # Solve (J^T J) delta = -J^T r
        try:
            delta = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered")
            break
        
        # Update parameters
        x_new = x + delta[0]
        y_new = y + delta[1]
        
        path.append((x_new, y_new))
        
        # Check convergence
        if np.linalg.norm(delta) < tol:
            break
        
        x, y = x_new, y_new
    
    return np.array(path), i + 1

# Three random initializations
np.random.seed(42)
initial_points = [
    (-1.5, 2.0),
    (0.0, -1.0),
    (1.5, 1.5)
]

print("\n" + "="*80)
print("TASK 2: Optimization Methods")
print("="*80)

# Run all methods from all initial points
results = {}
for method_name, method_func in [('Gradient Descent', gradient_descent),
                                   ('Newton Method', newton_method),
                                   ('Gauss-Newton', gauss_newton)]:
    results[method_name] = []
    print(f"\n{method_name}:")
    print("-" * 40)
    
    for idx, (x0, y0) in enumerate(initial_points):
        if method_name == 'Gradient Descent':
            path, iters = method_func(x0, y0, learning_rate=0.001)
        else:
            path, iters = method_func(x0, y0)
        
        results[method_name].append(path)
        x_final, y_final = path[-1]
        f_final = f(x_final, y_final)
        
        print(f"  Init {idx+1}: ({x0:6.2f}, {y0:6.2f}) -> ({x_final:8.5f}, {y_final:8.5f}), "
              f"f = {f_final:10.7f}, iterations = {iters}")

# Visualize convergence paths
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']

for ax_idx, (method_name, ax) in enumerate(zip(['Gradient Descent', 'Newton Method', 'Gauss-Newton'], axes)):
    # Plot contours
    contourf = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    ax.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.3, linewidths=0.5)
    
    # Plot minimum
    ax.plot(x_min, y_min, 'r*', markersize=20, label='Global minimum', zorder=10)
    
    # Plot convergence paths
    for idx, path in enumerate(results[method_name]):
        ax.plot(path[:, 0], path[:, 1], 
                color=colors[idx], marker=markers[idx], markersize=4,
                linewidth=2, alpha=0.7, markevery=max(1, len(path)//20),
                label=f'Init {idx+1}: ({initial_points[idx][0]:.1f}, {initial_points[idx][1]:.1f})')
        
        # Mark start point
        ax.plot(path[0, 0], path[0, 1], marker='x', color=colors[idx], 
                markersize=12, markeredgewidth=3, zorder=9)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.show()

print("\n" + "="*80)

### TASK 3: Minimize Rosenbrock's function using the Nelder-Mead Simplex algorithm

from scipy.optimize import minimize

# Rosenbrock function for scipy (takes array input)
def rosenbrock_scipy(params):
    x, y = params
    return 100 * (y - x**2)**2 + (1 - x)**2

# Store path during optimization
def nelder_mead_with_path(x0, y0):
    """
    Nelder-Mead optimization using scipy with path tracking
    """
    path = [(x0, y0)]
    
    def callback(xk):
        path.append((xk[0], xk[1]))
    
    result = minimize(rosenbrock_scipy, 
                     x0=[x0, y0], 
                     method='Nelder-Mead',
                     callback=callback,
                     options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    return np.array(path), result.nit, result

# Use the same three random initializations
print("\n" + "="*80)
print("TASK 3: Nelder-Mead Simplex Algorithm")
print("="*80)

nelder_mead_results = []
print("\nNelder-Mead (scipy):")
print("-" * 40)

for idx, (x0, y0) in enumerate(initial_points):
    path, iters, result = nelder_mead_with_path(x0, y0)
    nelder_mead_results.append(path)
    x_final, y_final = result.x
    f_final = result.fun
    
    print(f"  Init {idx+1}: ({x0:6.2f}, {y0:6.2f}) -> ({x_final:8.5f}, {y_final:8.5f}), "
          f"f = {f_final:10.7f}, iterations = {iters}, success = {result.success}")

# Visualize Nelder-Mead convergence
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot contours
contourf = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
ax.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.3, linewidths=0.5)

# Plot minimum
ax.plot(x_min, y_min, 'r*', markersize=20, label='Global minimum', zorder=10)

# Plot convergence paths for all three initializations
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']

for idx, path in enumerate(nelder_mead_results):
    ax.plot(path[:, 0], path[:, 1], 
            color=colors[idx], marker=markers[idx], markersize=4,
            linewidth=2, alpha=0.7, markevery=max(1, len(path)//20),
            label=f'Init {idx+1}: ({initial_points[idx][0]:.1f}, {initial_points[idx][1]:.1f})')
    
    # Mark start point
    ax.plot(path[0, 0], path[0, 1], marker='x', color=colors[idx], 
            markersize=12, markeredgewidth=3, zorder=9)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Nelder-Mead Simplex Algorithm Convergence', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.colorbar(contourf, ax=ax, label='f(x, y)')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
