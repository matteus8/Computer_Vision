from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    # Get the uploaded file from the request
    uploaded_file = request.files['file']

    # Get the absolute path to the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Save the uploaded file to the 'uploads' directory
    file_path = os.path.join(current_directory, 'uploads', uploaded_file.filename)
    uploaded_file.save(file_path)

    # Load the saved image and preprocess it
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Load the pre-trained model
    model_path = os.path.join(current_directory, 'dog_cat_classification_model.h5')
    model = tf.keras.models.load_model(model_path)
    # model = tf.keras.models.load_model('dog_cat_classification_model.h5')

    # Perform the classification
    result = model.predict(image)
    if result[0][0] > 0.5:
        classification = 'Dog'
    else:
        classification = 'Cat'

    # Render the result template with the classification
    return render_template('result1.html', classification=classification)

@app.route('/result')
def result():
    return render_template('result1.html')

if __name__ == '__main__':
    app.run(debug=True)
