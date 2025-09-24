import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        super().__init__()

        # Initialize weights using He initialization (good for ReLU)
        self.W = np.random.normal(
            loc=0.0,
            scale=np.sqrt(2 / (input_count + output_count)),
            size=(output_count, input_count)
        )
        # Initialize biases to zero
        self.b = np.zeros((output_count,))
        
        # Gradients will be stored here after the backward pass
        self.grad_W = None
        self.grad_b = None

    def get_parameters(self):
        """Returns the learnable parameters (weights and biases)."""
        return {
            'w': self.W,
            'b': self.b
        }

    def get_buffers(self):
        """Returns non-learnable buffers (none for this layer)."""
        return {}

    def forward(self, x):
        """
        Computes the forward pass: Y = X @ W.T + b
        """
        # Cache the input 'x' for use in the backward pass
        cache = {'x': x}
        output = x @ self.W.T + self.b
        return (output, cache)

    def backward(self, output_grad, cache):
        """
        Computes the backward pass to find gradients.
        """
        # Retrieve the input from the forward pass
        x = cache['x']

        # Gradient with respect to the input of this layer (to be passed back)
        grad_x = output_grad @ self.W
        
        # Gradient with respect to the weights (store it)
        self.grad_W = output_grad.T @ x
        
        # Gradient with respect to the bias (store it)
        self.grad_b = np.sum(output_grad, axis=0)
        
        return (
            grad_x,
            {
                'w': self.grad_W,
                'b': self.grad_b
        })


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()

        # Learnable parameters: scale (gamma) and shift (beta)
        self.gamma = np.ones((1, input_count))
        self.beta = np.zeros((1, input_count))
        
        # Gradients for the learnable parameters
        self.grad_gamma = None
        self.grad_beta = None
        
        # Non-learnable buffers for running statistics (used during evaluation)
        self.global_mean = np.zeros((1, input_count))
        self.global_variance = np.zeros((1, input_count))
        
        # Hyperparameters
        self.alpha = alpha  # Momentum for the moving average
        self.epsilon = 1e-8 # Small value for numerical stability

    def get_parameters(self):
        """Returns the learnable parameters (gamma and beta)."""
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }

    def get_buffers(self):
        """Returns the non-learnable buffers (running mean and variance)."""
        return {
            'global_mean': self.global_mean,
            'global_variance': self.global_variance
        }

    def forward(self, x):
        """
        Directs the input to the correct forward pass based on the layer's mode.
        """
        if self._is_training:
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    def _forward_training(self, x):
        """
        Computes the forward pass during training.
        Normalizes using the current batch's statistics and updates running stats.
        """
        # 1. Calculate current batch mean and variance
        batch_mean = np.mean(x, axis=0, keepdims=True)
        batch_var = np.var(x, axis=0, keepdims=True)

        # 2. Update running mean and variance using exponential moving average
        self.global_mean = (1 - self.alpha) * self.global_mean + self.alpha * batch_mean
        self.global_variance = (1 - self.alpha) * self.global_variance + self.alpha * batch_var

        # 3. Normalize the input
        x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

        # 4. Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        # Cache necessary values for the backward pass
        cache = {'x': x, 'x_normalized': x_normalized, 'batch_mean': batch_mean, 'batch_var': batch_var}
        
        return (output, cache)

    def _forward_evaluation(self, x):
        """
        Computes the forward pass during evaluation/inference.
        Normalizes using the stored running statistics.
        """
        # 1. Normalize the input using the stored running mean and variance
        x_normalized = (x - self.global_mean) / np.sqrt(self.global_variance + self.epsilon)
        
        # 2. Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        # No cache is needed as backward pass is not performed during evaluation
        return (output, {})

    def backward(self, output_grad, cache):
        """
        Computes the backward pass for batch normalization.
        """
        # Retrieve cached values from the forward pass
        x = cache['x']
        x_normalized = cache['x_normalized']
        batch_mean = cache['batch_mean']
        batch_var = cache['batch_var']
        
        m = x.shape[0] # Number of samples in the batch
        
        # --- Gradients of learnable parameters ---
        # Gradient of gamma (scale)
        self.grad_gamma = np.sum(output_grad * x_normalized, axis=0, keepdims=True)
        
        # Gradient of beta (shift)
        self.grad_beta = np.sum(output_grad, axis=0, keepdims=True)
        
        # --- Gradient with respect to the input x (to be passed back) ---
        # inv_std = 1.0 / np.sqrt(batch_var + self.epsilon)
        
        # grad_x_normalized = output_grad * self.gamma
        
        # grad_var = np.sum(grad_x_normalized * (x - batch_mean) * -0.5 * inv_std**3, axis=0, keepdims=True)
        
        # grad_mean = np.sum(grad_x_normalized * -inv_std, axis=0, keepdims=True) + \
        #             grad_var * np.mean(-2.0 * (x - batch_mean), axis=0, keepdims=True)
        
        # grad_x = (grad_x_normalized * inv_std) + (grad_var * 2.0 * (x - batch_mean) / m) + (grad_mean / m)

        std_dev = np.sqrt(batch_var + self.epsilon)

        grad_x_normalized = output_grad * self.gamma

        grad_var = np.sum(grad_x_normalized * (x - batch_mean) * (-1/2)*(batch_var + self.epsilon)**(-3/2), axis=0, keepdims=True)

        grad_mean = -np.sum(grad_x_normalized / std_dev, axis=0, keepdims=True) + \
                    grad_var * np.mean(-2.0 * (x - batch_mean), axis=0, keepdims=True) # TODO

        grad_x = grad_x_normalized / std_dev + (2/m) * grad_var * (x - batch_mean) + (1/m) * grad_mean
    
        return (
            grad_x,
            {
                'gamma': self.grad_gamma,
                'beta': self.grad_beta
        })


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        """Sigmoid has no learnable parameters."""
        return {}

    def get_buffers(self):
        """Sigmoid has no non-learnable buffers."""
        return {}

    def forward(self, x):
        """Computes the sigmoid function and caches the output for backpropagation."""
        y = 1 / (1 + np.exp(-x))
        cache = {'y': y}
        return (y, cache)

    def backward(self, output_grad, cache):
        """Computes the gradient using the cached output 'y' from the forward pass."""
        y = cache['y']
        # The derivative of sigmoid is y * (1 - y)
        return (output_grad * ((1 - y) * y), {})


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        """ReLU has no learnable parameters."""
        return {}

    def get_buffers(self):
        """ReLU has no non-learnable buffers."""
        return {}

    def forward(self, x):
        """Computes the ReLU function and caches the input for backpropagation."""
        output = np.maximum(0, x)
        cache = {'x': x}
        return (output, cache)

    def backward(self, output_grad, cache):
        """Computes the gradient using the cached input 'x' from the forward pass."""
        x = cache['x']
        # The derivative of ReLU is 1 for positive inputs and 0 otherwise.
        return (output_grad * (x > 0), {})
