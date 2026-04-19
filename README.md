# NumPy Deep Learning: From Scratch

This project is a handwritten digit recognition neural network built entirely from scratch using pure Python and NumPy. The goal of this project was to understand the underlying calculus, linear algebra, and architecture of machine learning without relying on "black-box" frameworks like TensorFlow or PyTorch.

## The Journey & Architecture Evolution

This network was built iteratively, starting from a basic mathematical foundation and evolving into a deep learning architecture.

### Step 1: The Shallow Foundation
* **Architecture:** 1 Hidden Layer (128 nodes).
* **Details:** Implemented the core forward pass, backpropagation, and Gradient Descent algorithms. Flattened $28 \times 28$ pixel images into a 784-element 1D array.
* **Result:** Achieved **~89% accuracy** on unseen test data. Proved the math worked, but hit a clear performance ceiling.

### Step 2: Scaling Width
* **Architecture:** 1 Hidden Layer (512 nodes).
* **Details:** Increased the "memory" of the network to see if a wider layer could break the 90% barrier. 
* **Result:** Accuracy remained stagnant at **~89%**. Learned that simply adding capacity doesn't solve fundamental structural limitations; the network became better at memorizing but not necessarily better at generalizing the core concepts of digits.

### Step 3: Going Deep
* **Architecture:** 2 Hidden Layers (128 nodes $\rightarrow$ 128 nodes).
* **Details:** Transitioned from a Multi-Layer Perceptron to true Deep Learning. 
    * Implemented **He Initialization** (`np.random.randn(...) * np.sqrt(2.0 / inputs)`) to prevent variance explosion and dead neurons across multiple layers.
    * Derived and applied the **Chain Rule** twice to pull error gradients back through the middle layer.
* **Result:** Broke the barrier and achieved **~91% Test Accuracy**. The deep architecture successfully learned hierarchical, abstract representations of the numbers.

### Step 4
* **Architecture:** CNN (8 Filters $3 \times 3 \rightarrow$ Flatten $\rightarrow$ 128 nodes $\rightarrow$ 128 nodes $\rightarrow$ 10 nodes).
* **Details:** Cured the network's "Spatial Blindness" by replacing the raw pixel inputs with a 2D mathematical "magnifying glass."
    * Implemented Convolutional Filters to scan the image for spatial features (edges, loops, curves) before passing them to the dense layers.
    * Developed vectorized im2col (Image to Column) math to convert slow, nested sliding-window loops into high-speed matrix multiplications.
    * Integrated CuPy for GPU acceleration and optimized RAM usage, cutting training time from an estimated year down to minutes.
* **Result:** Achieved 97.20% Training Accuracy and 96.67% Test Accuracy. The network successfully reshaped its loss landscape, finding a much deeper local minimum by learning actual geometric features instead of memorizing pixel coordinates.

### Step 5: Adam Optimizer & Learning Rate Decay
* **Architecture:** Same network with Adam optimizer instead of SGD
* **Details:** In this phase, we upgraded the learning engine from standard Gradient Descent to a custom-built Adam Optimizer. By tracking both the exponentially weighted moving average of the gradients (momentum/velocity) and the squared gradients (friction), the network dynamically customizes the learning rate for every individual weight and bias. To prevent overshooting at the absolute bottom of the loss valley, we also implemented Learning Rate Step Decay to progressively shrink the step size during the final epochs.
* **Result:** Achieved 99.95% Training Accuracy and 98.09% Test Accuracy. This performance effectively hits the mathematical ceiling for a standard Convolutional Neural Network architecture on this dataset, demonstrating robust feature extraction with only a minor generalization gap left to solve via regularization.

### Step 6: Data Augmentation
* **Architecture:** Same network as Step 5 with data augmentation
* **Details:** In this phase, we tackled overfitting by implementing a custom, purely mathematical data augmentation engine. To prevent the network from memorizing static pixel coordinates, we introduced randomized horizontal and vertical pixel shifts (Translations) to the batches immediately before the forward pass. Using independent X and Y matrix slicing and coordinate logic, we dynamically shifted the matrices and padded them with zero-value (black) pixels. Because the images are augmented "on-the-fly," the network never sees the exact same image twice, forcing it to learn universal geometric shapes rather than positional anomalies.
* **Result:** Achieved 91.94% Training Accuracy and 98.59% Test Accuracy. This solved the problem of overfitting which means that the network no longer memorises the data set but actually learning on it. The higher testing accuracy is the byproduct of this learning.

## Key Technical Learnings
* **The Math is the Engine:** Fully translated the Chain Rule, Cross-Entropy Loss, Softmax, and ReLU derivatives into matrix operations.
* **Dimensionality:** Mastered matrix alignment—ensuring weights, biases, and dot products flow seamlessly backward and forward.
* **Optimization vs. Capacity:** Learned that falling into local minima of the loss function cannot always be fixed by throwing more nodes at the problem.
* **Spacial Invariance:** Learned how Parameter Sharing (using the same $3 \times 3$ filter weights across the whole image) allows the model to recognize a digit regardless of its position.
* **Hardware & Memory Management:** Discovered that software performance is heavily dictated by memory leaks, garbage collection, and how efficiently data is batched to the GPU.

## Roadmap: Step 7 (The Next Evolution)
The current CNN has proven the power of feature extraction, but it can be optimized further to push toward the 99% accuracy threshold.

# Next Step: Implement Max Pooling and Dropout Regularization.
Instead of passing every single convolution pixel forward, the next evolution will add pooling layers to shrink the spatial dimensions, making the network even more resilient to shifted or distorted digits. Introducing dropout regularization. Artifically turn off a random % of the neurons of the network to force some "lazy" neurons to wake up and work.