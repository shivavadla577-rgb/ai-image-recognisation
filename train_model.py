
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Create models folder
os.makedirs("models", exist_ok=True)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# Normalize image pixels (0-255 becomes 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Image categories
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Create CNN model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # First convolution layer
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Second convolution layer
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Third convolution layer
    layers.Conv2D(64, (3, 3), activation="relu"),

    # Convert feature maps into a vector
    layers.Flatten(),

    # Fully connected layers
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    # 10 output categories
    layers.Dense(10, activation="softmax")
])

# Configure model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(
    x_test, y_test
)

print("Test Accuracy:", test_accuracy * 100, "%")

# Save trained model
model.save("models/image_classifier.keras")

print("Model saved successfully!")
