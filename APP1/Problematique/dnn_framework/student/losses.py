import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        num_samples = x.shape[0]
        
        # 1. Compute probabilities using the softmax function
        probabilities = softmax(x)
        
        # 2. Compute the cross-entropy loss
        # We select the probabilities corresponding to the correct class for each sample
        correct_class_probabilities = probabilities[range(num_samples), target]
        
        # Add a small epsilon for numerical stability to avoid log(0)
        epsilon = 1e-12
        loss = -np.mean(np.log(correct_class_probabilities + epsilon))
        
        # 3. Compute the gradient of the loss with respect to the input x
        # The gradient of softmax + cross-entropy has a very simple form: (probabilities - target_one_hot)
        # First, create a one-hot encoded version of the target
        target_one_hot = np.zeros_like(probabilities)
        target_one_hot[range(num_samples), target] = 1
        
        input_grad = (probabilities - target_one_hot) / num_samples
        
        return (loss, input_grad)


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    # Subtract the max for numerical stability to prevent overflow
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    # Normalize
    return exps / np.sum(exps, axis=1, keepdims=True)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        # 1. Calculate the error
        error = x - target
        
        # 2. Compute the mean squared error loss
        loss = np.mean(error**2)
        
        # 3. Compute the gradient of the loss with respect to the input x
        input_grad = (2 * error) / error.size
        
        return (loss, input_grad)
