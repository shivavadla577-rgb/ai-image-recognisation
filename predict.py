
import tensorflow as tf
import numpy as np
from PIL import Image
import sys

# Load trained model
model = tf.keras.models.load_model(
    "models/image_classifier.keras"
)

# Categories
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

def predict_image(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize image to 32x32
    image = image.resize((32, 32))

    # Convert image into array
    image_array = np.array(image)

    # Normalize image
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    # Find predicted category
    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    print("Predicted Image:", predicted_class)
    print("Confidence:", round(float(confidence), 2), "%")

    # Display top 3 predictions
    top_indices = np.argsort(predictions[0])[::-1][:3]

    print("\nTop 3 Predictions:")

    for index in top_indices:
        print(
            class_names[index],
            "-",
            round(float(predictions[0][index] * 100), 2),
            "%"
        )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict.py image.jpg")
    else:
        predict_image(sys.argv[1])
