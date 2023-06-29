import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = tf.keras.models.load_model('dog_cat_classification_model.h5')

# Define the target image size
target_size = (224, 224)

# List of image paths to classify
image_paths = ['C:\\Users\\mcamacho\\OneDrive - ISSAC Corp\\Desktop\\RufusfinalIOM-M.jpg', 'C:\\Users\\mcamacho\\OneDrive - ISSAC Corp\\Desktop\\istockphoto-1255354804-612x612.jpg', 'C:\\Users\\mcamacho\\OneDrive - ISSAC Corp\\Desktop\\Picture1.jpg', 'C:\\Users\\mcamacho\\OneDrive - ISSAC Corp\\Desktop\\closeup-maine-coon-cat-portrait-isolated-on-black-background-sergey-taran.jpg', 'C:\\Users\\mcamacho\\OneDrive - ISSAC Corp\\Desktop\\lookie.jpg']

# Perform classification for each image
for image_path in image_paths:
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = img_array / 255.0  # Normalize the image

    prediction = model.predict(preprocessed_img)

    if prediction[0][0] > 0.5:
        classification = 'Dog'
    else:
        classification = 'Cat'

    print(f"The image '{image_path}' is classified as: {classification}")
