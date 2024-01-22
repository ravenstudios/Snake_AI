import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models

# Step 1: Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Step 2: Data Preprocessing
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Step 3: Build a Neural Network Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 4: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the Model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Step 7: Make Predictions
# predictions = model.predict("2.png")
# print(predictions)
from PIL import Image
import numpy as np

# Load the PNG image
image_path = '2.png'
png_image = Image.open(image_path)

# Preprocess the image
resized_image = png_image.resize((28, 28), Image.ANTIALIAS)  # Corrected line
grayscale_image = resized_image.convert('L')  # Convert to grayscale
image_array = np.array(grayscale_image)
normalized_image = image_array.reshape((1, 28, 28, 1)).astype('float32') / 255

# Make predictions
predictions = model.predict(normalized_image)

# Print the predictions
print(f'Predictions: {predictions}')
