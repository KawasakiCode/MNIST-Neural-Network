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

### Step 3: Going Deep (Current State)
* **Architecture:** 2 Hidden Layers (128 nodes $\rightarrow$ 128 nodes).
* **Details:** Transitioned from a Multi-Layer Perceptron to true Deep Learning. 
    * Implemented **He Initialization** (`np.random.randn(...) * np.sqrt(2.0 / inputs)`) to prevent variance explosion and dead neurons across multiple layers.
    * Derived and applied the **Chain Rule** twice to pull error gradients back through the middle layer.
* **Result:** Broke the barrier and achieved **~91% Test Accuracy**. The deep architecture successfully learned hierarchical, abstract representations of the numbers.

## Key Technical Learnings
* **The Math is the Engine:** Fully translated the Chain Rule, Cross-Entropy Loss, Softmax, and ReLU derivatives into matrix operations.
* **Dimensionality:** Mastered matrix alignment—ensuring weights, biases, and dot products flow seamlessly backward and forward.
* **Optimization vs. Capacity:** Learned that falling into local minima of the loss function cannot always be fixed by throwing more nodes at the problem.

## Roadmap: Step 4 (The Next Evolution)
The current Multi-Layer Perceptron is suffering from **Spatial Blindness**. By flattening a 2D image into a 1D array, the network loses all spatial relationships between pixels (curves, edges, and loops). 

**Next Step:** Implement a **Convolutional Neural Network (CNN)**.
Instead of connecting every pixel to every node, the next evolution will build a 2D mathematical "magnifying glass" (convolutional filters) to scan the image for spatial features, with the goal of breaking the **95%+ accuracy barrier**.