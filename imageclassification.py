import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import datasets
from tensorflow.keras import layers,models
# download required dataset from te tensorflow
(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
x_train.shape
# labelling the classes required for the dataset
classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# reshaping the data in the dataset from 2dimensional to 1dimensional array
y_train=y_train.reshape(-1, )
y_train[:5]
# function to plot the image from the dataset for the given index with the label
def plot_image(x,y,index):
  plt.figure(figsize=(15,2))
  plt.imshow(x[index])
  plt.xlabel(classes[y[index]])
plot_image(x_train,y_train,7)
# normalizing the pixels for the dataset
x_train= x_train/255
x_test=x_test/255
# building an Artificial Neural Network ANN
ann=models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(4000,activation='relu'),
    layers.Dense(1000,activation='relu'),
    layers.Dense(10,activation='sigmoid')
])
ann.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
ann.fit(x_train,y_train,epochs=7)
ann.evaluate(x_test,y_test)
y_pred=ann.predict(x_test)
y_pred_classes=[np.argmax(element) for element in y_pred]
print('ANN Classification report is : ',classification_report(y_test,y_pred_classes))

OUTPUT :

![Image](https://github.com/user-attachments/assets/c9dd843f-9a39-4f6b-a7d8-0fb73343e50d)

# building a Convolutional Neural Network for the given dataset
cnn=models.Sequential([
    # cnn layers
    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    # dense layers for cnn
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
])
cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
cnn.fit(x_train,y_train,epochs=10)
cnn.evaluate(x_test,y_test)
# reshaping the y_test to flatten a shapped y_test
y_test=y_test.reshape(-1)
plot_image(x_test,y_test,1)
# predicting the outcome for CNN
y_pred=cnn.predict(x_test)
y_pred[:5]
y_classes=[np.argmax(element) for element in y_pred]
y_test[:5]
plot_image(x_test,y_test,8)
classes[y_classes[8]]
print("CNN classification report is: ",classification_report(y_test,y_classes))


https://github.com/user-attachments/assets/88005aa8-4d82-4dfa-9fbb-ba7b6da023eb
