# Animal Classification with Convolutional Neural Network (CNN)

This project involves building and training a Convolutional Neural Network (CNN) for classifying images of animals. The model is designed to classify images from a dataset containing various animal categories, using TensorFlow and Keras for model construction and training.

## Dataset

The dataset used in this project is the *Animals with Attributes 2* dataset, which contains images of various animals. The selected animal classes for this project include:

- Collie
- Dolphin
- Elephant
- Fox
- Moose
- Rabbit
- Sheep
- Squirrel
- Giant Panda
- Polar Bear

The images are preprocessed by resizing them to 224x224 pixels, and the dataset is split into training, validation, and test sets.

## Project Workflow

1. **Data Preprocessing:**
    - Selected a subset of animal classes.
    - Resized images to 224x224 pixels for input to the CNN model.
    - Split the data into training (80%) and testing (20%) sets.

2. **Model Architecture:**
    - Built a CNN model with 4 convolutional layers, followed by a flattening layer, a fully connected layer, and a softmax output layer for multi-class classification.
    - The layers in the model are described as follows:

      - **Convolutional Layers (Conv2D):** 
        - Used 4 convolutional layers with increasing numbers of filters (32, 64, 128, 256) to capture complex features in the images. 
        - The kernel size was set to 3x3, a common choice for capturing fine-grained features in images.
        - The activation function used in the convolutional layers is **Mish**. This activation function has been shown to improve training performance over ReLU in some cases, as it allows for smoother gradients and avoids dead neurons.
      
      - **Max Pooling Layers (MaxPooling2D):**
        - After each convolutional layer, a max pooling layer with a pool size of 2x2 was applied. This reduces the spatial dimensions of the feature maps and helps in downsampling, making the model more computationally efficient and helping with feature extraction.

      - **Flatten Layer:**
        - After passing through the convolutional and pooling layers, the 3D feature maps are flattened into a 1D vector to feed into the fully connected layers.

      - **Dense Layer:**
        - A fully connected dense layer with 256 neurons was added after the flatten layer. This layer uses **Mish** as the activation function, allowing the model to learn non-linear combinations of the features learned by the convolutional layers.

      - **Dropout Layer:**
        - A **Dropout** layer with a rate of 0.3 was included after the dense layer to help prevent overfitting by randomly setting a fraction of the input units to zero during training.

      - **Output Layer (Dense):**
        - The output layer uses **softmax** activation to predict the class probabilities for the 10 animal categories. Softmax is suitable for multi-class classification problems, as it produces a probability distribution over the output classes.

    - The model has a total of 9,828,426 trainable parameters.

3. **Data Augmentation:**
    - Applied various data augmentation techniques, including rotation, width/height shifts, zooming, and horizontal flipping to improve generalization.

4. **Model Training:**
    - The model was compiled with the **Adam optimizer** and **categorical cross-entropy** loss function. 
    - **Adam** optimizer was chosen for its ability to adaptively adjust learning rates during training, making it more efficient and faster for training deep neural networks.
    - The **categorical cross-entropy** loss function is ideal for multi-class classification problems, as it measures the difference between the predicted class probabilities and the true class labels.
    - Used early stopping to prevent overfitting, with patience set to 5 epochs. If the validation loss does not improve after 5 epochs, training is stopped early.
    - Implemented a learning rate scheduler that reduces the learning rate by 50% after 3 epochs with no improvement in the validation loss.

5. **Evaluation:**
    - The model was evaluated on the test dataset, and the final test accuracy was 66%.
    - The test loss was 105.35%, indicating that further improvements could be made in the model's generalization.

## Requirements

To run this project, the following Python libraries are required:

- TensorFlow
- Keras
- OpenCV
- Matplotlib
- NumPy
- Scikit-learn

These dependencies can be installed using `pip`:

```bash
pip install tensorflow opencv-python matplotlib numpy scikit-learn
