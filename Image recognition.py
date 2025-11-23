1 method 
# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# Import the MNIST dataset from Keras
from keras.datasets import mnist

# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Check lengths of training and test datasets
print("Length of x_train:", len(x_train))
print("Length of x_test:", len(x_test))

# Check the shape of the first training example
print("Shape of x_train[0]:", x_train[0].shape)

# Display the pixel values of the first image (optional)
print("First image array data:\n", x_train[0])

# Visualize the first image from the dataset
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

3 method 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Import the mnist object from keras.datasets
from keras.datasets import mnist

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Check data sizes
len(x_train)
len(x_test)

# Display shape of one image
x_train[0].shape

# Display pixel values of one image
x_train[0]

# Display image
plt.matshow(x_train[1])
plt.title(f"Label: {y_train[1]}")
plt.show()

# Display labels
y_train[2]
y_train[:5]

# Normalize the data
x_train = x_train / 255
x_test = x_test / 255
# FLATTEN THE ARRAY
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_train_flattened.shape

x_test_flattened = x_test.reshape(len(x_test), 28*28)
x_test_flattened.shape

# Convert an image to array
arr = np.array(x_train[0])

# Display flattened first image array
x_train_flattened[0]

# Build the model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train_flattened, y_train, epochs=5)

# MODEL EVALUATION ON TESTING
model.evaluate(x_test_flattened, y_test)

# Display one test image
plt.matshow(x_test[0])
plt.title(f"Actual Label: {y_test[0]}")
plt.show()
# Predict on test data
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

y_pred = model.predict(x_test_flattened)

# Convert predictions to class labels
y_pred_classes = [np.argmax(i) for i in y_pred]

# Create confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10,7))
sn.heatmap(conf_mat, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()         





