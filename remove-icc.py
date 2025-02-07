from PIL import Image

def remove_icc_profile(image_path):
    # Open the image using Pillow
    image = Image.open(image_path)
    
    # Save the image without the ICC profile (just save it normally)
    image.save(image_path, "PNG", icc_profile=None)

# Example: Apply this function to all PNG files in your dataset
import os

data_dir = 'pokemon-dataset'  # Directory with your images

for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".png"):  # Process only PNG images
            image_path = os.path.join(subdir, file)
            remove_icc_profile(image_path)
