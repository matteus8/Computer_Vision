import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Set the parent directory
parent_dir = 'C:/Users/mcamacho/OneDrive - ISSAC Corp/Desktop/CV-Project/Pre-CV-Project/new'

# Define the target image size
target_size = (224, 224)

# Create an image data generator for preprocessing the images
datagen = ImageDataGenerator(rescale=1./255)


########################SPLIT DOGS########################
# Split the filenames into training and testing sets for dogs
train_dog_filenames, val_dog_filenames = train_test_split(os.listdir(os.path.join(parent_dir, 'PetImages', 'dog')), test_size=0.2, random_state=42)
# Print the number of files used for training and validation for dogs
print("There are", len(train_dog_filenames),"files that will be used for training dogs.")
print("There are", len(val_dog_filenames),"files that will be used for validating dogs.")

########################SPLIT CATS########################
# Split the filenames into training and testing sets for cats
train_cat_filenames, val_cat_filenames = train_test_split(os.listdir(os.path.join(parent_dir, 'PetImages', 'cat')), test_size=0.2, random_state=42)
# Print the number of files used for training and validation for cats
print("There are", len(train_cat_filenames),"files that will be used for training cats.")
print("There are", len(val_cat_filenames),"files that will be used for validating cats.")


# Combine the dog and cat filenames for training and validation sets
train_filenames = train_dog_filenames + train_cat_filenames
val_filenames = val_dog_filenames + val_cat_filenames

# Print the number of images in each set
print("Number of training images:", len(train_filenames))
print("Number of validation images:", len(val_filenames))

# Create the training and validation data generators
train_generator = datagen.flow_from_directory(
    os.path.join(parent_dir, 'PetImages', 'train'),
    target_size=target_size,
    batch_size = int(20),
    class_mode='binary',
    shuffle=True
)
val_generator = datagen.flow_from_directory(
    os.path.join(parent_dir, 'PetImages', 'val'),
    target_size=target_size,
    batch_size = int(20),
    class_mode='binary',
    shuffle=False
)


# Load a pre-trained model (e.g., VGG16)
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3)) #tagetsize is 244,244 with 3 color channels

# Freeze the pre-trained model's layers
base_model.trainable = False

# Create the model architecture
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'), #relu - negative num's become zero and positive num's remain unchanged
    tf.keras.layers.Dense(1, activation='sigmoid') #squishification to 0-1
])

# Compile the model --- training the model --- model learning
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=3, validation_data=val_generator)

# Save the trained model
model.save('dog_cat_classification_model.h5')
