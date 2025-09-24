from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """

    def __init__(self, parameters, learning_rate=0.01):
        """
        Initializes the optimizer.
        :param parameters: A list of the model's parameters (e.g., [W1, b1, W2, b2, ...]).
        :param learning_rate: The step size for the gradient descent updates.
        """
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        """
        Performs the SGD update for a single parameter.
        :param parameter: The parameter tensor (e.g., W1).
        :param parameter_grad: The gradient of the loss with respect to the parameter.
        :return: The updated parameter tensor.
        """
        # The core SGD update rule: parameter = parameter - learning_rate * gradient
        return parameter - self.learning_rate * parameter_grad
