from flask import Flask, jsonify, request
import numpy as np
import PIL
from PIL import Image
from keras.models import load_model

app = Flask(__name__)

#model = load_model('./my_model.h5')
image_size = 224
@app.route('/predict', methods=["POST"])
def predict_image():
        # Preprocess the image so that it matches the training input
        image = request.files['file']
        image = Image.open(image)
        image = np.asarray(image.resize((image_size,image_size)))
        image = image.reshape(1,image_size,image_size,3)

        # Use the loaded model to generate a prediction.
        pred = model.predict(image)
        pred = pred[0][0]
        # Prepare and send the response.
        return jsonify(pred)

if __name__ == "__main__":
        app.run(host='0.0.0.0', port=8998)
