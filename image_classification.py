import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the preprocessed image data and labels
preprocessed_images = np.load("preprocessed_images.npy")
labels = np.load("labels.npy")

# Split the data into training and testing/validation sets
X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_images, labels, test_size=0.2, random_state=42)

# Load a pre-trained model
base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

# Freeze the pre-trained model's layers
base_model.trainable = False

# Add new layers on top of the pre-trained model for classification
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=30, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the trained model
model.save("image_classification_model.h5")
