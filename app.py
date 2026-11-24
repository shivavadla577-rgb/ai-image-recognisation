
import os
from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
app = Flask(__name__)

# Upload configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Load trained model
model = tf.keras.models.load_model(
    "models/image_classifier.keras"
)

# Image categories
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# HTML design
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Image Classification</title>

    <style>
        body {
            font-family: Arial;
            background: #eef2ff;
            text-align: center;
            padding: 40px;
        }

        .container {
            background: white;
            max-width: 550px;
            margin: auto;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px #cccccc;
        }

        h1 {
            color: #3730a3;
        }

        input {
            margin: 20px;
        }

        button {
            background: #4f46e5;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }

        img {
            max-width: 250px;
            max-height: 250px;
            margin-top: 20px;
            border-radius: 10px;
        }

        .result {
            background: #dcfce7;
            padding: 15px;
            margin-top: 20px;
            border-radius: 10px;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>AI Image Classification</h1>

    <p>Upload an image to identify its category.</p>

    <form method="POST" enctype="multipart/form-data">

        <input type="file" name="file"
               accept="image/*" required>

        <br>

        <button type="submit">Classify Image</button>

    </form>

    {% if image_url %}

        <img src="{{ image_url }}" alt="Uploaded image">

    {% endif %}

    {% if result %}

        <div class="result">

            <h2>Prediction: {{ result }}</h2>

            <h3>Confidence: {{ confidence }}%</h3>

            <h4>Top 3 Predictions</h4>

            {% for name, score in top_predictions %}
                <p>{{ name }} - {{ score }}%</p>
            {% endfor %}

        </div>

    {% endif %}

    {% if error %}
        <p style="color:red">{{ error }}</p>
    {% endif %}

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    top_predictions = []
    image_url = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Please select an image."

        else:
            try:
                # Validate image
                image = Image.open(file.stream).convert("RGB")

                # Save uploaded image
                filename = str(uuid.uuid4()) + ".jpg"
                filepath = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    filename
                )

                image.save(filepath)

                # Preprocess image
                image = image.resize((32, 32))

                image_array = np.array(image)
                image_array = image_array.astype("float32") / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                # Predict
                predictions = model.predict(
                    image_array, verbose=0
                )[0]

                predicted_index = np.argmax(predictions)

                result = class_names[predicted_index]

                confidence = round(
                    float(predictions[predicted_index] * 100), 2
                )

                # Top 3 results
                top_indices = np.argsort(predictions)[::-1][:3]

                top_predictions = [
                    (
                        class_names[i],
                        round(float(predictions[i] * 100), 2)
                    )
                    for i in top_indices
                ]

                image_url = "/" + filepath.replace("\\", "/")

            except Exception:
                error = "Invalid image. Please upload a valid image."

    return render_template_string(
        HTML_PAGE,
        result=result,
        confidence=confidence,
        top_predictions=top_predictions,
        image_url=image_url,
        error=error
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
