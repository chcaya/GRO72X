import argparse
import numpy as np
import matplotlib.pyplot as plt

# --- Import your project's classes ---
# (Adjust these imports to match your project structure)
from dnn_framework import Network, FullyConnectedLayer, BatchNormalization, ReLU
from mnist.dataset import MnistDataset

def create_network(checkpoint_path):
    """
    Creates the network architecture.
    IMPORTANT: This must be the EXACT same architecture as the trained model.
    """
    layers = [
        # Input layer: MNIST images are 28x28=784 pixels.
        # Hidden layer 1 with 128 neurons, Batch Norm, and ReLU activation.
        FullyConnectedLayer(input_count=784, output_count=128),
        BatchNormalization(input_count=128, alpha=0.1),
        ReLU(),

        # Hidden layer 2 with 32 neurons, Batch Norm, and ReLU activation.
        FullyConnectedLayer(input_count=128, output_count=32),
        BatchNormalization(input_count=32, alpha=0.1),
        ReLU(),

        # Output layer: 10 neurons for the 10 digits (0-9).
        # No activation function here, as CrossEntropyLoss expects raw logits.
        FullyConnectedLayer(input_count=32, output_count=10)
    ]
    network = Network(layers)
    if checkpoint_path is not None:
        network.load(checkpoint_path)

    return network

def softmax(x):
    """Compute softmax values for a set of scores in x."""
    e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

def main():
    parser = argparse.ArgumentParser(description='Test a single image with a trained network')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved model checkpoint file.')
    parser.add_argument('--image_index', type=int, default=43,
                        help='Index of the image to test from the MNIST test set.')
    args = parser.parse_args()

    # 1. Create the network and load the trained weights
    network = create_network(args.checkpoint_path)
    network.eval() # Set the network to evaluation mode

    # 2. Load the test dataset and select a single image
    print(f"Loading image at index {args.image_index} from the test set...")
    test_dataset = MnistDataset('testing')
    input_image, target_label = test_dataset[args.image_index] # Get image and label

    # 3. Visualize the input image
    plt.imshow(input_image.reshape(28, 28), cmap='gray')
    plt.title(f"Input Image: The actual number is {target_label}")
    plt.show()

    # 4. Prepare the image for the network
    # The network expects a batch of inputs, so we reshape (784,) to (1, 784)
    network_input = input_image.reshape(1, -1)

    # 5. Make a prediction
    logits = network.forward(network_input)
    probabilities = softmax(logits.squeeze()) # Squeeze to remove batch dimension
    prediction = np.argmax(probabilities)

    # 6. View the output
    print(f"\nModel Prediction: {prediction}")
    print(f"Actual Number: {target_label}")
    print("\n--- Output Probabilities ---")
    for i, prob in enumerate(probabilities):
        print(f"Digit {i}: {prob:.4f} ({prob*100:.2f}%)")

if __name__ == '__main__':
    main()
