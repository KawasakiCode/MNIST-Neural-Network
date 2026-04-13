import matplotlib.pyplot as plt

def plot_training_curves(loss_history, accuracy_history):
    """
    Takes your two history lists and graphs them side-by-side.
    """
    # Create a blank canvas with two side-by-side plots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: The Loss Curve
    ax1.plot(loss_history, color='red', linewidth=2)
    ax1.set_title("Categorical Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.7) # Adds a faint background grid

    # Plot 2: The Accuracy Curve
    ax2.plot(accuracy_history, color='blue', linewidth=2)
    ax2.set_title("Training Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Automatically adjust spacing so the titles don't overlap, then display
    plt.tight_layout()
    plt.show()

def show_prediction(image_array, true_label, predicted_label):
    """
    Takes a single row of pixel data, reshapes it, and draws it.
    """
    # Image is currently a flat array of 784 pixels
    # Matplotlib needs a 2D grid to draw a picture, so we reshape it to 28x28
    image_2d = image_array.reshape(28, 28)

    # Draw the image using a standard grayscale color map
    plt.imshow(image_2d, cmap='gray')

    # Color the title green if the network was right, red if it was wrong
    title_color = 'green' if true_label == predicted_label else 'red'
    
    plt.title(f"True Answer: {true_label} | Network Guess: {predicted_label}", 
              color=title_color, 
              fontsize=14, 
              fontweight='bold')

    plt.axis('off')
    plt.show()

def plot_accuracy_only(accuracy_history):
    fig, (ax1) = plt.subplot(1, figsize=(12, 5))

    # Plot 1: The Accuracy Curve
    ax1.plot(accuracy_history, color='blue', linewidth=2)
    ax1.set_title("Training Accuracy (%)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Automatically adjust spacing so the titles don't overlap, then display
    plt.tight_layout()
    plt.show()


