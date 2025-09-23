#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------- Utils functions ------------------------------
def assert_almost_equal(a, b):
    assert np.mean(np.abs(a - b)) < 1e-5


def test_grad(input_shape, forward, backward, X=None, output_grad=None):
    h = 1e-5
    x = np.random.randn(*input_shape) if X is None else X
    output_grad = np.random.randn(*forward(x).shape) if output_grad is None else output_grad
    
    analytical_grad = backward(x, output_grad)
    numerical_grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        
        old_value = x[i]
        x[i] = old_value + h
        y1 = forward(x)
        x[i] = old_value - h
        y0 = forward(x)
        
        numerical_grad[i] = np.sum((y1 - y0) * output_grad) / (2 * h)
        it.iternext()

    error = np.mean(np.abs(analytical_grad - numerical_grad))
    assert_almost_equal(error, 0)        


# ------------------------------ Layer functions ------------------------------
def fully_connected_forward(W, b, X):
    """Computes the forward pass of a fully connected layer."""
    return X @ W.T + b

def fully_connected_backward(W, b, X, output_grad):
    """Computes the backward pass of a fully connected layer."""
    grad_X = output_grad @ W
    grad_W = output_grad.T @ X
    grad_b = np.sum(output_grad, axis=0)
    return grad_X, grad_W, grad_b

def relu_forward(X):
    """Computes the forward pass of a ReLU activation."""
    return np.maximum(0, X)

def relu_backward(X, output_grad):
    """Computes the backward pass of a ReLU activation."""
    return output_grad * (X > 0)

def sigmoid_forward(X):
    """Computes the forward pass of a sigmoid activation."""
    return 1 / (1 + np.exp(-X))

def sigmoid_backward(X, output_grad):
    """Computes the backward pass of a sigmoid activation."""
    s = sigmoid_forward(X)
    grad_sigmoid = s * (1 - s)
    return output_grad * grad_sigmoid

def bce_forward(y, target):
    """Computes the binary cross-entropy loss (forward pass)."""
    # Epsilon for numerical stability to avoid log(0)
    epsilon = 1e-12
    m = target.shape[0]
    loss = -1/m * np.sum(target * np.log(y + epsilon) + (1 - target) * np.log(1 - y + epsilon))
    return loss

def bce_backward(y, target):
    """Computes the gradient of the binary cross-entropy loss."""
    # Epsilon for numerical stability to avoid division by zero
    epsilon = 1e-12
    m = target.shape[0]
    
    # Gradient of BCE with respect to its input y
    grad = (y - target) / ((y * (1 - y)) + epsilon)
    
    # Normalize by the number of samples
    return grad / m


# ------------------------------ Test functions ------------------------------
def test_fully_connected_forward():
    print('------test_fully_connected_forward-------')
    W = np.array([[2, 3]])
    b = np.array([1])
    X = np.array([[-1.0, 0.5], [-2.0, 1.0]])
    Y = fully_connected_forward(W, b, X)
    
    assert_almost_equal(Y[0], 0.5)
    assert_almost_equal(Y[1], 0.0)
    print('\tOk')


def test_fully_connected_backward():
    print('------test_fully_connected_backward------')
    X = np.random.randn(2, 4)
    W = np.random.randn(10, 4)
    b = np.random.randn(10)
    def forward_X(X):
        return fully_connected_forward(W, b, X)
    def backward_X(X, output_grad):
        return fully_connected_backward(W, b, X, output_grad)[0]
    def forward_W(W):
        return fully_connected_forward(W, b, X)
    def backward_W(W, output_grad):
        return fully_connected_backward(W, b, X, output_grad)[1]
    def forward_b(b):
        return fully_connected_forward(W, b, X)
    def backward_b(b, output_grad):
        return fully_connected_backward(W, b, X, output_grad)[2]

    test_grad((2, 4), forward_X, backward_X)
    test_grad((10, 4), forward_W, backward_W)
    test_grad((10,), forward_b, backward_b)
    print('\tOk')


def test_relu_forward():
    print('------------test_relu_forward------------')
    X = np.array([-1.0, 0.5])
    Y = relu_forward(X)
    
    assert_almost_equal(X[0], -1.0)
    assert_almost_equal(X[1], 0.5)
    assert_almost_equal(Y[0], 0.0)
    assert_almost_equal(Y[1], 0.5)
    print('\tOk')


def test_relu_backward():
    print('------------test_relu_backward-----------')
    
    def forward(X):
        return relu_forward(X)
    def backward(X, output_grad):
        return relu_backward(X, output_grad)
    
    test_grad((2, 4), forward, backward)
    print('\tOk')


def test_sigmoid_forward():
    print('----------test_sigmoid_forward----------')    
    X = np.array([-1.0, 0.5])
    Y = sigmoid_forward(X)

    assert_almost_equal(Y[0], 0.2689414)
    assert_almost_equal(Y[1], 0.6224593)
    print('\tOk')


def test_sigmoid_backward():
    print('----------test_sigmoid_backward----------')
    
    def forward(X):
        return sigmoid_forward(X)
    def backward(X, output_grad):
        return sigmoid_backward(X, output_grad)
    
    test_grad((2, 4), forward, backward)
    print('\tOk')


def test_bce_forward():
    print('------------test_bce_forward-------------')
    x = np.array([0.1, 0.6])
    target = np.array([0.0, 1.0])
    y = bce_forward(x, target)

    assert_almost_equal(y, 0.3081)
    print('\tOk')


def test_bce_backward():
    print('------------test_bce_backward------------')
    X = np.array([0.1, 0.6])
    target = np.array([0.0, 1.0])
    output_grad = 1
    def forward(X):
        return bce_forward(X, target)
    def backward(X, output_grad):
        return bce_backward(X, target)

    test_grad((2,), forward, backward, X=X, output_grad=output_grad)
    print('\tOk')
    

def test():
    test_fully_connected_forward()
    test_fully_connected_backward()
    test_relu_forward()
    test_relu_backward()
    test_sigmoid_forward()
    test_sigmoid_backward()
    test_bce_forward()
    test_bce_backward()


# ---------------------------- Training functions -----------------------------
def train(x_train, target_train, x_val=None, target_val=None, epoch_count=100, learning_rate=0.04):
    counts = [2, 25, 25, 1]
    
    W1 = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / (counts[0] + counts[1])),
                          size=(counts[1], counts[0]))
    b1 = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / counts[1]),
                          size=(counts[1],))

    W2 = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / (counts[1] + counts[2])),
                          size=(counts[2], counts[1]))
    b2 = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / counts[2]),
                          size=(counts[2],))

    W3 = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / (counts[2] + counts[3])),
                          size=(counts[3], counts[2]))
    b3 = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / counts[3]),
                          size=(counts[3],))
    
    losses_train = []
    accuracies_train = []
    losses_val = []
    accuracies_val = []    

    for epoch in range(epoch_count):
        print('epoch={}'.format(epoch + 1))
        
        # --- Training: Forward pass (Matches your equations exactly) ---
        # u = W1x + b1
        u = fully_connected_forward(W1, b1, x_train)
        # g = R(u)
        g = relu_forward(u)
        # v = W2g + b2
        v = fully_connected_forward(W2, b2, g)
        # h = R(v)
        h = relu_forward(v)
        # w = W3h + b3
        w = fully_connected_forward(W3, b3, h)
        # y = S(w)
        y = sigmoid_forward(w)
        
        # L = C(y, ŷ)
        loss = bce_forward(y, target_train)

        # --- Training: Backward pass ---
        # Start from the end and propagate backward
        grad_L_y = bce_backward(y, target_train)
        grad_L_w = sigmoid_backward(w, grad_L_y)
        grad_L_h, grad_W3, grad_b3 = fully_connected_backward(W3, b3, h, grad_L_w)
        grad_L_v = relu_backward(v, grad_L_h)
        grad_L_g, grad_W2, grad_b2 = fully_connected_backward(W2, b2, g, grad_L_v)
        grad_L_u = relu_backward(u, grad_L_g)
        _, grad_W1, grad_b1 = fully_connected_backward(W1, b1, x_train, grad_L_u)
        
        # --- Training: Gradient Descent ---
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2
        W3 -= learning_rate * grad_W3
        b3 -= learning_rate * grad_b3
        
        # Training: Metrics
        losses_train.append(loss)        
        predicted_classes = (y > 0.5).astype(int)
        
        accuracy = np.sum((predicted_classes == target_train)) / target_train.size
        accuracies_train.append(accuracy)

        print('Training: loss={:.4f}, accuracy={:.4f}'.format(loss, accuracy))

        if x_val is not None and target_val is not None:

            # --- Validation: Forward pass ---
            # Propagate the validation data through the current network weights
            u_val = fully_connected_forward(W1, b1, x_val)
            g_val = relu_forward(u_val)
            v_val = fully_connected_forward(W2, b2, g_val)
            h_val = relu_forward(v_val)
            w_val = fully_connected_forward(W3, b3, h_val)
            y = sigmoid_forward(w_val)
            
            # Calculate the loss on the validation data
            loss = bce_forward(y, target_val)

            # Validation: Metrics
            losses_val.append(loss)        
            predicted_classes = (y > 0.5).astype(int)
            
            accuracy = np.sum((predicted_classes == target_val)) / target_val.size
            accuracies_val.append(accuracy)          
            
            print('Validation: loss={:.4f}, accuracy={:.4f}'.format(loss, accuracy))
        print()
        
    show_learning_curves(losses_train, accuracies_train, title='Training')
    show_classification(W1, b1, W2, b2, W3, b3, x_train, title='Training')

    if x_val is not None and target_val is not None:
        show_learning_curves(losses_val, accuracies_val, title='Validation')
        show_classification(W1, b1, W2, b2, W3, b3, x_val, title='Validation')

    show_decision_boundary(W1, b1, W2, b2, W3, b3)
    
    plt.show()
     
def show_learning_curves(losses, accuracies, title=''):
    fig = plt.figure(figsize=(10, 5), dpi=200)
    fig.suptitle(title)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    epochs = range(1, len(losses) + 1)
    ax1.plot(epochs, losses, '-o')
    ax1.set_title(u'Loss')
    ax1.set_xlabel(u'Epoch')
    ax1.set_ylabel(u'Loss')

    epochs = range(1, len(accuracies) + 1)
    ax2.plot(epochs, accuracies, '-o')
    ax2.set_title(u'Accuracy')
    ax2.set_xlabel(u'Epoch')
    ax2.set_ylabel(u'Accuracy')

    fig.show()
        

def show_decision_boundary(W1, b1, W2, b2, W3, b3):
    x1 = np.arange(-1, 1, 0.01)
    x2 = np.arange(1, -1, -0.01)
    
    data = np.array(np.meshgrid(x1, x2)).T.reshape(-1,2)
    
    # --- Forward pass to get predictions for each point on the grid ---
    u = fully_connected_forward(W1, b1, data)
    g = relu_forward(u)
    v = fully_connected_forward(W2, b2, g)
    h = relu_forward(v)
    w = fully_connected_forward(W3, b3, h)
    y = sigmoid_forward(w)
    
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(111)
    ax.imshow(1 - y.reshape(x1.size, x2.size).T, cmap='bwr', extent=[-1, 1, -1, 1], vmin=0, vmax=1)
    fig.show()


def show_classification(W1, b1, W2, b2, W3, b3, X, title=''):

    # --- Forward pass to get predictions for the input data --- 
    u = fully_connected_forward(W1, b1, X)
    g = relu_forward(u)
    v = fully_connected_forward(W2, b2, g)
    h = relu_forward(v)
    w = fully_connected_forward(W3, b3, h)
    y = sigmoid_forward(w)
    
    predicted_classes = (y > 0.5).astype(int)

    c1 = np.squeeze(predicted_classes==0)
    c2 = np.squeeze(predicted_classes==1)

    fig = plt.figure(figsize=(5,5), dpi=200)
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    ax.scatter(X[c1,0], X[c1,1], c='red')
    ax.scatter(X[c2,0], X[c2,1], c='blue')
    fig.show()

# ------------------------------------ main -----------------------------------
if __name__ == "__main__":
    mode = 'training'
    if mode == 'test':
        test()
    elif mode == 'overfitting':
        x_train = np.array([[0.5, 0.5], [0.75, 0.75], [0.25, 0.25], [0.1, 0.1]])
        target_train = np.array([[0], [0], [1], [1]], dtype=int)
        train(x_train, target_train)
    elif mode == 'training':
        x_train = np.genfromtxt('train.csv', delimiter=',')[:,slice(0,2)]
        target_train = np.expand_dims(np.genfromtxt('train.csv', delimiter=',')[:,2], axis=1)
        x_val = np.genfromtxt('val.csv', delimiter=',')[:,slice(0,2)]
        target_val = np.expand_dims(np.genfromtxt('val.csv', delimiter=',')[:,2], axis=1)
        train(x_train, target_train, x_val, target_val, epoch_count=20000, learning_rate=0.04)
