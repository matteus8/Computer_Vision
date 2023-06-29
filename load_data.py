import os
from PIL import Image

def load_images_from_folder(folder, num_images=None, valid_extensions=(".jpg", ".jpeg", ".png")):
    images = []
    count = 0  # Keep track of the number of loaded images
    for filename in os.listdir(folder):
        if num_images is not None and count == num_images:
            break
        try:
            # Check if the file has a valid image extension
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(folder, filename)
                # Check if the file size is non-zero (greater than 0)
                if os.path.getsize(file_path) > 0:
                    # Load the image using PIL
                    try:
                        image = Image.open(file_path)
                        image.load()
                        # Convert the image to RGB mode if it's not already
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        # Append the loaded image to the list
                        images.append(image)
                        count += 1  # Increment the count
                    except (IOError, OSError) as e:
                        print(f"Corrupted image: {file_path}")
                        os.remove(file_path)  # Delete the file
                else:
                    print(f"Ignoring 0KB image: {file_path}")
                    os.remove(file_path)  # Delete the file
            else:
                print(f"Ignoring file with unsupported extension: {file_path}")
                os.remove(file_path)  # Delete the file
        except Exception as e:
            print(f"Error loading image {file_path}: {str(e)}")
            os.remove(file_path)  # Delete the file
            continue
    return images

# Define the paths to the cat and dog image folders
cat_folder = "C:\\Users\\mcamacho\\OneDrive - ISSAC Corp\\Desktop\\CV-Project\\Pre-CV-Project\\new\\PetImages\\Cat"
dog_folder = "C:\\Users\\mcamacho\\OneDrive - ISSAC Corp\\Desktop\\CV-Project\\Pre-CV-Project\\new\\PetImages\\Dog"

# Call the function to load the cat images
loaded_cat_images = load_images_from_folder(cat_folder, num_images=1200)

# Check if any cat images were successfully loaded
if len(loaded_cat_images) > 0:
    print(f"Successfully loaded {len(loaded_cat_images)} cat images.")
else:
    print("No cat images were loaded.")

# Call the function to load the dog images
loaded_dog_images = load_images_from_folder(dog_folder, num_images=1200)

# Check if any dog images were successfully loaded
if len(loaded_dog_images) > 0:
    print(f"Successfully loaded {len(loaded_dog_images)} dog images.")
else:
    print("No dog images were loaded.")
