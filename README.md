# Convolutional Deep Neural Network for Digit Classification

## AIM

### To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

### MNIST Handwritten Digit Classification Dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

### The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.


![image](https://github.com/PSriVarshan/mnist-classification/assets/114944059/4ba57e18-ede4-40e0-8543-d94c9793b639)



## Neural Network Model


![MNISTDiagram](https://github.com/PSriVarshan/mnist-classification/assets/114944059/9bb645f3-16be-43a0-ac79-22b17408076b)


## DESIGN STEPS

### STEP 1:

#### Import tensorflow and preprocessing libraries

### STEP 2:

#### Build a CNN model

### STEP 3:
#### Compile and fit the model and then predict

## PROGRAM & OUTPUT

### Name: Sri Varshan P
### Register Number: 212222240104

### Importing necessary libraries and the training set

```py


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

```
### Adding layers of network

```py

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32, kernel_size=(5,5),  activation='relu')),
model.add(layers.MaxPool2D(pool_size=(2, 2))),
model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')),
model.add(layers.MaxPool2D(pool_size=(2, 2))),
model.add(layers.Flatten()),
model.add(layers.Dense(64, activation='relu')),
model.add(layers.Dense(10, activation='softmax'))
```
### Modelling

```py

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=10,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
```

### Evaluation 

```py

metrics = pd.DataFrame(model.history.history)

metrics.head()
```
### Visualization
#### Validation Loss Vs Iteration Plot
```py
metrics[['accuracy','val_accuracy']].plot()
```

![image](https://github.com/PSriVarshan/mnist-classification/assets/114944059/797acf6a-f29c-4634-838c-093e29f54d08)


#### Training Loss
```py
metrics[['loss','val_loss']].plot()
```

![image](https://github.com/PSriVarshan/mnist-classification/assets/114944059/25466b9a-14c1-4d0b-a911-f6df2def1c01)

### Confusion Matrix

```py
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))
```

![image](https://github.com/PSriVarshan/mnist-classification/assets/114944059/4daeb4e8-130a-4794-848c-10abaa0de249)


### Classification Report
```py
print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/mnist2.png')

```

![image](https://github.com/PSriVarshan/mnist-classification/assets/114944059/e2d581d2-89c8-472f-b0f4-28b9c5408652)


### New Sample Data Prediction
```py
type(img)

img = image.load_img('/content/mnist2.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_inverted_scaled.reshape(28,28),cmap='gray')


```
![image](https://github.com/PSriVarshan/mnist-classification/assets/114944059/a8891e9f-c1f7-46eb-bbeb-451322283431)



## RESULT
### Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
