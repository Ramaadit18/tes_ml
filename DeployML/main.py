import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model_pest_detection.h5')
label = ["ants", "bees", "beetle", "caterpillar", "earthworms", "earwig", "grasshopper", "moth", "slug", "snail", "wasp", "weevil"]

@app.route('/predict', methods=['POST'])
def index():
    file = request.files['file']
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})
    
    # Read the image bytes
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Get prediction
    pred_img = predict_label(img)
    return jsonify({"prediction": pred_img})

def predict_label(img):
    # Preprocess the image
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Make predictions
    pred = model.predict(x)
    result = label[np.argmax(pred)]
    return result

if __name__ == "__main__":
    app.run(debug=True)
