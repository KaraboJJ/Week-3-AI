from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)
model = load_model('model/mnist_cnn.h5')

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    image = np.array(image)
    image = 255 - image  # invert colors
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        image_bytes = file.read()
        image = preprocess_image(image_bytes)
        pred = model.predict(image)
        prediction = np.argmax(pred)
        
        encoded_img = base64.b64encode(image_bytes).decode('utf-8')
        return render_template('index.html', prediction=prediction, image_data=encoded_img)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
