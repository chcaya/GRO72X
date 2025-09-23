import numpy as np
import matplotlib.pyplot as plt


def invert_matrix_gradient_descent(A, learning_rate=0.01, iterations=1000):
    """
    Finds the inverse of a square matrix A using gradient descent
    and returns the history of the loss at each iteration.
    """
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Input matrix must be square.")

    I = np.identity(n)
    B = np.random.rand(n, n)
    
    # List to store the loss value at each iteration
    loss_history = []

    for i in range(iterations):
        error_matrix = B @ A - I
        loss = np.sum(error_matrix**2)
        loss_history.append(loss)
        
        gradient = 2 * error_matrix @ A.T
        B = B - learning_rate * gradient
        
    return B, loss_history

# --- Example Usage ---
if __name__ == "__main__":
    # Define the initial matrix A
    # A = np.array([
    #     [3, 4, 1],
    #     [5, 2, 3],
    #     [6, 2, 2]
    # ])

    # A = np.array([
    #     [3, 4, 1, 2, 1, 5],
    #     [5, 2, 3, 2, 2, 1],
    #     [6, 2, 2, 6, 4, 5],
    #     [1, 2, 1, 3, 1, 2],
    #     [1, 5, 2, 3, 3, 3],
    #     [1, 2, 2, 4, 2, 1]
    # ])

    A = np.array([
        [2, 1, 1, 2],
        [1, 2, 3, 2],
        [2, 1, 1, 2],
        [3, 1, 4, 1]
    ])

    # List of learning rates to test
    learning_rates = [0.001, 0.005, 0.01]
    iterations = 1000
    
    plt.figure(figsize=(12, 8))

    for lr in learning_rates:
        print(f"--- Running optimization with learning rate (α) = {lr} ---")
        
        # Calculate the inverse and get the loss history
        B_calculated, loss_history = invert_matrix_gradient_descent(A, learning_rate=lr, iterations=iterations)
        
        # Plot the loss history for the current learning rate
        plt.plot(range(iterations), loss_history, label=f'α = {lr}')

        # Print the calculated inverse matrix
        print("\nInverse B (calculated via Gradient Descent):\n", B_calculated)
        
        # Verify the result by multiplying B with A
        identity_check = B_calculated @ A
        print("Verification (B @ A):\n", identity_check, "\n")

    # --- Matplotlib plot configuration ---
    plt.title('Loss vs. Iterations for Different Learning Rates')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Cost)')
    plt.yscale('log') # Use a log scale to better see the differences
    plt.legend()
    plt.grid(True)
    plt.show()
