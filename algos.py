from typing import Callable, Literal, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking-Armijo", "Backtracking-Goldstein", "Bisection"],
) -> npt.NDArray[np.float64]:
    """
    Implements the Steepest Descent algorithm with inexact line search.
    
    Args:
        f: Function to minimize
        d_f: Gradient function
        initial_point: Starting point for optimization
        condition: Line search condition to use
        
    Returns:
        Optimized point
    """
    # Initialize
    x = initial_point.copy()
    max_iter = 10000
    epsilon = 1e-6
    
    alpha = 1.0  # Initial step size
    rho = 0.5    # Reduction factor for backtracking
    c1 = 0.1     # Armijo condition parameter
    c2 = 0.9     # Goldstein/Wolfe condition parameter
    
    f_vals = []
    grad_norms = []
    iterations = []
    points = [x.copy()]
    
    f_prev = float('inf')
    stagnation_counter = 0
    min_step_size = 1e-10
    
    if not hasattr(f, '__name__') or f.__name__ == '':
        f.__name__ = "function"
    
    for iter_count in range(max_iter):
        f_val = f(x)
        grad = d_f(x)
        grad_norm = np.linalg.norm(grad)
        
        f_vals.append(f_val)
        grad_norms.append(grad_norm)
        iterations.append(iter_count)
        
        if grad_norm < epsilon:
            print(f"Converged: Gradient norm {grad_norm} < {epsilon}")
            break
            
        if abs(f_prev - f_val) < 1e-10:
            stagnation_counter += 1
            if stagnation_counter > 100:
                print(f"Converged: Function value stagnated at {f_val}")
                break
        else:
            stagnation_counter = 0
            
        f_prev = f_val
            
        direction = -grad
        
        try:
            if condition == "Backtracking-Armijo":
                alpha = backtracking_armijo(f, d_f, x, direction, alpha, rho, c1)
            elif condition == "Backtracking-Goldstein":
                alpha = backtracking_goldstein(f, d_f, x, direction, alpha, rho, c1, c2)
            elif condition == "Bisection":
                alpha = bisection_wolfe(f, d_f, x, direction, c1, c2)
        except Exception as e:
            print(f"Line search failed: {e}")
            alpha = min_step_size
            
        if alpha < min_step_size:
            print(f"Converged: Step size {alpha} < {min_step_size}")
            break
        
        x_new = x + alpha * direction
        
        if np.allclose(x, x_new):
            print("Converged: No significant update to x")
            break
            
        x = x_new
        points.append(x.copy())
    
    if iter_count == max_iter - 1:
        print(f"Warning: Maximum iterations ({max_iter}) reached without convergence")
        
    try:
        create_plots(f, f_vals, grad_norms, iterations, points, initial_point, condition)
    except Exception as e:
        print(f"Failed to create plots: {e}")
    
    return x

def backtracking_armijo(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    alpha: float,
    rho: float,
    c1: float
) -> float:
    """
    Backtracking line search with Armijo condition.
    """
    f_x = f(x)
    grad_x = d_f(x)
    directional_derivative = np.dot(grad_x, direction)
    
    while f(x + alpha * direction) > f_x + c1 * alpha * directional_derivative:
        alpha *= rho
        
    return alpha

def backtracking_goldstein(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    alpha: float,
    rho: float,
    c1: float,
    c2: float
) -> float:
    """
    Backtracking line search with Armijo-Goldstein conditions.
    """
    f_x = f(x)
    grad_x = d_f(x)
    directional_derivative = np.dot(grad_x, direction)
    
    while True:
        f_new = f(x + alpha * direction)
        armijo = f_new <= f_x + c1 * alpha * directional_derivative
        goldstein = f_new >= f_x + c2 * alpha * directional_derivative
        
        if armijo and goldstein:
            break
            
        alpha *= rho
        
    return alpha

def bisection_wolfe(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    c1: float,
    c2: float
) -> float:
    """
    Bisection method with Wolfe conditions.
    """
    alpha_max = 10.0
    alpha_min = 0.0
    alpha = 1.0
    max_iter = 20
    
    f_x = f(x)
    grad_x = d_f(x)
    directional_derivative = np.dot(grad_x, direction)
    
    for _ in range(max_iter):
        f_new = f(x + alpha * direction)
        grad_new = d_f(x + alpha * direction)
        
        armijo = f_new <= f_x + c1 * alpha * directional_derivative
        
        wolfe = np.dot(grad_new, direction) >= c2 * directional_derivative
        
        if armijo and wolfe:
            return alpha
        
        if not armijo:
            alpha_max = alpha
        else:  
            alpha_min = alpha
            
        alpha = (alpha_min + alpha_max) / 2.0
    
    return alpha

def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:
    """
    Implements various variants of Newton's Method.
    
    Args:
        f: Function to minimize
        d_f: Gradient function
        d2_f: Hessian function
        initial_point: Starting point for optimization
        condition: Newton method variant to use
        
    Returns:
        Optimized point
    """
    x = initial_point.copy()
    max_iter = 10000
    epsilon = 1e-6
    
    alpha = 0.001  # For Damped Newton
    beta = 0.75    # For Damped Newton
    mu = 1.0       # For Levenberg-Marquardt (initial value)
    
    f_vals = []
    grad_norms = []
    iterations = []
    points = [x.copy()]
    
    f_prev = float('inf')
    stagnation_counter = 0
    min_step_size = 1e-10
    
    if not hasattr(f, '__name__') or f.__name__ == '':
        f.__name__ = "function"
    
    for iter_count in range(max_iter):
        f_val = f(x)
        grad = d_f(x)
        grad_norm = np.linalg.norm(grad)
        
        try:
            hessian = d2_f(x)
            if np.any(np.isnan(hessian)) or np.any(np.isinf(hessian)):
                raise ValueError("Hessian contains NaN or infinity values")
        except Exception as e:
            print(f"Error computing Hessian: {e}")
            hessian = np.eye(len(x))
        
        f_vals.append(f_val)
        grad_norms.append(grad_norm)
        iterations.append(iter_count)
        
        if grad_norm < epsilon:
            print(f"Converged: Gradient norm {grad_norm} < {epsilon}")
            break
            
        if abs(f_prev - f_val) < 1e-10:
            stagnation_counter += 1
            if stagnation_counter > 100: 
                print(f"Converged: Function value stagnated at {f_val}")
                break
        else:
            stagnation_counter = 0
            
        f_prev = f_val
        
        try:
            if condition == "Pure":
                try:
                    direction = -np.linalg.solve(hessian, grad)
                    step_size = 1.0
                except np.linalg.LinAlgError:
                    reg_factor = max(1e-6, 1e-4 * grad_norm)
                    hessian_reg = hessian + reg_factor * np.eye(len(x))
                    direction = -np.linalg.solve(hessian_reg, grad)
                    step_size = 1.0
            
            elif condition == "Damped":
                try:
                    direction = -np.linalg.solve(hessian, grad)
                    step_size = backtracking_line_search(f, x, direction, alpha, beta)
                except np.linalg.LinAlgError:
                    reg_factor = max(1e-6, 1e-4 * grad_norm)
                    hessian_reg = hessian + reg_factor * np.eye(len(x))
                    direction = -np.linalg.solve(hessian_reg, grad)
                    step_size = backtracking_line_search(f, x, direction, alpha, beta)
                    if step_size < min_step_size:
                        direction = -grad
                        step_size = backtracking_line_search(f, x, direction, alpha, beta)
            
            elif condition == "Levenberg-Marquardt":
                identity = np.eye(len(x))
                hessian_modified = hessian + mu * identity
                
                try:
                    direction = -np.linalg.solve(hessian_modified, grad)
                    
                    pred_reduction = -0.5 * np.dot(direction, np.dot(hessian, direction)) - np.dot(grad, direction)
                    actual_reduction = f(x) - f(x + direction)
                    
                    ratio = actual_reduction / (pred_reduction + 1e-10)  
                    
                    if ratio > 0.75:
                        mu = max(mu / 2, 1e-7)
                    elif ratio < 0.25:
                        mu = min(mu * 2, 1e7)
                        
                    step_size = 1.0
                except np.linalg.LinAlgError:
                    mu *= 10
                    direction = -grad  
                    step_size = 0.1
            
            elif condition == "Combined":
                identity = np.eye(len(x))
                hessian_modified = hessian + mu * identity
                
                try:
                    direction = -np.linalg.solve(hessian_modified, grad)
                    step_size = backtracking_line_search(f, x, direction, alpha, beta)
                    
                    pred_reduction = -0.5 * np.dot(direction, np.dot(hessian, direction)) - np.dot(grad, direction)
                    actual_reduction = f(x) - f(x + step_size * direction)
                    
                    ratio = actual_reduction / (pred_reduction + 1e-10)  
                    
                    if ratio > 0.75:
                        mu = max(mu / 2, 1e-7)
                    elif ratio < 0.25:
                        mu = min(mu * 2, 1e7)
                        
                    if step_size < min_step_size:
                        direction = -grad
                        step_size = backtracking_line_search(f, x, direction, alpha, beta)
                except np.linalg.LinAlgError:
                    mu *= 10
                    direction = -grad 
                    step_size = backtracking_line_search(f, x, direction, alpha, beta)
        except Exception as e:
            print(f"Error computing direction: {e}")
            direction = -grad
            step_size = 0.01
        
        if step_size < min_step_size:
            print(f"Converged: Step size {step_size} < {min_step_size}")
            break
        
        x_new = x + step_size * direction
        
        if np.allclose(x, x_new):
            print("Converged: No significant update to x")
            break
            
        x = x_new
        points.append(x.copy())
    
    if iter_count == max_iter - 1:
        print(f"Warning: Maximum iterations ({max_iter}) reached without convergence")
    
    try:
        create_plots(f, f_vals, grad_norms, iterations, points, initial_point, condition)
    except Exception as e:
        print(f"Failed to create plots: {e}")
    
    return x

def backtracking_line_search(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    x: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    alpha: float,
    beta: float
) -> float:
    """
    Simple backtracking line search for Newton methods.
    """
    step_size = 1.0
    f_x = f(x)
    
    while f(x + step_size * direction) > f_x - alpha * step_size * np.linalg.norm(direction)**2:
        step_size *= beta
        
    return step_size

def create_plots(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    f_vals: List[float],
    grad_norms: List[float],
    iterations: List[int],
    points: List[npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64],
    condition: str
):
    """
    Creates plots for function values, gradient norms, and contour plot.
    """
    initial_point_str = np.array2string(initial_point).replace('[', '').replace(']', '').replace(' ', '_')
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, f_vals, 'b-')
    plt.title(f"Function Value vs Iterations ({condition})")
    plt.xlabel("Iterations")
    plt.ylabel("Function Value")
    plt.grid(True)
    plt.savefig(f"plots/{f.__name__}_{initial_point_str}_{condition}_vals.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, grad_norms, 'r-')
    plt.title(f"Gradient Norm vs Iterations ({condition})")
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f"plots/{f.__name__}_{initial_point_str}_{condition}_grad.png")
    plt.close()
    
    if len(initial_point) == 2:
        points = np.array(points)
        
        # Determine range for contour plot
        x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
        y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute function values on grid
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
                
        # Plot contour
        plt.figure(figsize=(10, 8))
        contour = plt.contour(X, Y, Z, 20)
        plt.colorbar(contour)
        
        # Plot optimization path
        plt.plot(points[:, 0], points[:, 1], 'ro-', linewidth=1, markersize=3)
        plt.plot(points[0, 0], points[0, 1], 'go', markersize=5, label='Start')
        plt.plot(points[-1, 0], points[-1, 1], 'bo', markersize=5, label='End')
        
        # Add arrows to show direction
        for i in range(min(len(points)-1, 10)):  # Plot up to 10 arrows to avoid clutter
            idx = i * (len(points) // 10) if len(points) > 10 else i
            plt.arrow(points[idx, 0], points[idx, 1], 
                     points[idx+1, 0] - points[idx, 0], points[idx+1, 1] - points[idx, 1],
                     head_width=0.05, head_length=0.1, fc='k', ec='k')
        
        plt.title(f"Contour Plot with Optimization Path ({condition})")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/{f.__name__}_{initial_point_str}_{condition}_cont.png")
        plt.close()