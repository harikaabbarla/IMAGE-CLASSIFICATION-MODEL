# IMAGE-CLASSIFICATION-MODEL

COMPANY: CODTECH IT SOLUTIONS

NAME: ABBARLA HARIKA

INTERN ID: CT06DN519

DOMAIN: MACHINE LEARNING

DURATION: 6 WEEKS

MENTOR: NEELA SANTHOSH KUMAR

DESCRIPTION:

Convolutional Neural Network is a specialized neural network used primarily for image recognition and computer vision tasks like image classification, object detection and facial recognition. It automatically

learns spatial hierarchies of features of images.

CNN(Convolutional Neural Network) Architecture consists of several layers:

1.Input Layer: It takes image as input, here images are represented as 3D matrices.

2.Convolutional Layer: This layer applies kernel/filter to the images to extract features like edges,corners etc..

3.Activation Layer: (ReLU) It adds non linearity to the images and makes the model to learn complex patterns (ReLU-Rectified Linear Unit)

4.Pooling layer: In this layer model reduces the spatial size of the feature maps. There are two types of pooling techniques Max Pooling and Average Pooling techniques this layer helps in reducing overfitting and computation load on the model.

5.Flatten Layer: This layer converts 2D feature maps into 1D feature map. So that it can be an input to a fully connected input layer

6.Fully Connected (Dense Layer): Each neuron is connected to all other neurons of the previous layer.

Applications of CNN:

Object Detection.

Image Classification.

Medical Image Analysis.

Self Driving Cars.

The library used to bulid Convolutional Neural Network is "TensorFlow". This library is an open source machine learning frame work. It is one of the widely used tools in deep learning and machine learning.

It provides both low-level API for custom model building and high-level API for easier and faster model deployment.

In this program first I build an Aritificial Neural Network for image classification and then I built a Convolutional Neural Network to check the difference between them and i foiund that using Artificial neural

network the accuracy if around 50% and when using the Convolutional Neural Network the accuracy is excellent that the model has got an accuracy of around 80%.

Methods,libraries and functions used in this program:

Importing required libraries: tensorflow an open source deep learning library and used for defining and training and evaluting neural networks

matplotlib.pyplot used for plotting data

numpy is used for reshaping arrays and working with predictions

tensorflow.keras.datasets used to import CIFAR-10 data set for image classification

tensorflow.keras.layers used to include layers like Dense,Conv2D,Flatten,MaxPooling2D

tensorflow.keras.models used to define sequential model(models.Sequential())

Reshaping the labels: .reshape() function is used to reshape for model training

3.Plotting Function: plt.imshow() displays the image plt.xlabel() displays the class name label

4.Building an Artificial Neural Network and compiling the ANN: SGD(Stochastic Gradient Descent) optimizer and loss is (sparse_categorical_crossentropy) and accuracy

5.Printing the classification report of ANN: classification_report(y_pred) this method prints precision recall f1score and support of the model.

6.Building and evaluting the Convolutional Neural Network: this method also involves the same dense and flatten and Conv2D layers for convolution layer and it uses ReLU activation and Softmax. It uses adam optimizer and loss is same sparse_categorical_crossentropy

7.Classification report of CNN: this is final step of the project it prints the classification report of CNN which shows the best as compared to ANN

Dataset:

The dataset i used is CIFAR-10 dataset which is widely used for image claddification tasks. It contains 60,000 color images of size 32*32 and are divided into 10 classes

The classes includes objects like airplane,automobile,birds,cats,deer,dogs,frogs,horses,ships,trucks.

CIFAR-10 is used to train and evaluate computer vision models, especially convolutional neural networks (CNNs).

OUTPUT:
