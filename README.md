# Competitive Neural Network for Image Classification

This Python code implements a competitive neural network for image classification using a self-organizing map (SOM) approach. The network is trained on images of Saturn and Earth to classify input images into one of the two classes.

## Features

- Utilizes OpenCV and NumPy for image processing and neural network operations.
- Implements convolutional and pooling layers for feature extraction.
- Uses a competitive learning algorithm to classify images based on their features.
- Provides functions to train the network, extract features from images, and classify input images.

## Usage

To use the code:
1. Ensure you have the necessary dependencies installed (OpenCV and NumPy).
2. Run the `start()` function to initialize the weights of the neural network.
3. Call the `compititive()` function with an input image to classify it as either Saturn or Earth.
