from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# === CONFIG ===
IMG_SIZE = 224  # Same size used in training
CLASS_NAMES = ['dress', 'high_heel', 'handbag', 'skirt', 'outerwear', 'boot']

# Load model
model = tf.keras.models.load_model("../models/model_mobilenet_tl/best_model.h5", compile=False)

# === FLASK APP ===
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === ROUTES ===

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)

            # Preprocess image
            img = Image.open(image_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_tensor)[0]
            probabilities = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
            return render_template("result.html", image_path=image_path, predictions=probabilities)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
