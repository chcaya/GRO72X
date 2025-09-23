import numpy as np
import matplotlib.pyplot as plt

def polynomial_regression(x, y, degree, learning_rate, iterations):
    """
    Performs polynomial regression using gradient descent.
    
    Args:
        x (np.ndarray): Input feature values.
        y (np.ndarray): Target values.
        degree (int): The degree of the polynomial (N).
        learning_rate (float): The step size for gradient descent.
        iterations (int): The number of iterations to run.
        
    Returns:
        np.ndarray: The final optimized polynomial coefficients 'a'.
    """
    # Get the number of observations (I)
    num_samples = len(x)
    
    # 1. Prepare the feature matrix X based on the equation (15)
    # This creates a matrix where each row is [1, x, x^2, ..., x^N]
    X = np.column_stack([x**n for n in range(degree + 1)])
    
    # 2. Initialize the parameters 'a' to zeros
    # The number of parameters is N + 1
    a = np.zeros(degree + 1)
    
    # 3. Run the gradient descent loop
    for _ in range(iterations):
        # Calculate predictions ŷ = a * x^T (vectorized form)
        predictions = X @ a
        
        # Calculate the error (ŷ - y)
        error = predictions - y
        
        # Calculate the gradient based on equation (17)
        # Gradient = (2/I) * X^T * (ŷ - y)
        # Dividing by num_samples because using MSE instead of SSE
        gradient = (2 / num_samples) * X.T @ error
        
        # Update the parameters 'a'
        a = a - learning_rate * gradient
        
    return a

# --- Main script ---
if __name__ == "__main__":
    # --- Data Loading ---
    # Load the provided data into NumPy arrays
    x_train = np.array([-0.95, -0.82, -0.62, -0.43, -0.17, -0.07, 0.25, 0.38, 0.61, 0.79, 1.04])
    y_train = np.array([0.02, 0.03, -0.17, -0.12, -0.37, -0.25, -0.10, 0.14, 0.53, 0.71, 1.53])

    # --- Configuration ---
    degrees_to_test = [1, 2, 7]
    learning_rate = 0.01
    iterations = 1000

    # --- Plotting Setup ---
    plt.figure(figsize=(12, 8))
    # Plot the original data points
    plt.scatter(x_train, y_train, color='red', label='Données d\'entraînement')

    # Generate smooth x-values for plotting the final curves
    x_plot = np.linspace(-1.25, 1.25, 100)

    # --- Train and Plot for each degree ---
    for N in degrees_to_test:
        # Train the model to get the coefficients 'a'
        a = polynomial_regression(x_train, y_train, degree=N, learning_rate=learning_rate, iterations=iterations)
        
        # Prepare the feature matrix for the plotting values
        X_plot = np.column_stack([x_plot**n for n in range(N + 1)])
        
        # Calculate the predictions for the plot
        y_plot = X_plot @ a
        
        # Plot the resulting polynomial curve
        plt.plot(x_plot, y_plot, label=f'Régression d\'ordre {N}')
    
    # --- Finalize and Show Plot ---
    plt.title('Régression Polynomiale par Descente de Gradient')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.ylim(-1, 2) # Set y-axis limits for better visualization
    plt.show()
