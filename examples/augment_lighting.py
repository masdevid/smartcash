
import os
import cv2
import shutil
import random
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance


def apply_lighting_augmentation(image_path, output_dir_base):
    """
    Applies a random lighting augmentation to an image and saves it.

    Args:
        image_path (str): The path to the image file.
        output_dir_base (str): The base directory for the output images.
    """
    try:
        img = Image.open(image_path).convert("RGB")

        # Define a set of lighting augmentations with more extreme brightness
        augmentations = [
            # Extreme brightness variations
            lambda i: ImageEnhance.Brightness(i).enhance(random.uniform(0.2, 0.5)),  # Dark
            lambda i: ImageEnhance.Brightness(i).enhance(random.uniform(1.5, 2.5)),  # Bright

            # Other variations
            lambda i: ImageEnhance.Contrast(i).enhance(random.uniform(0.7, 1.3)),
            lambda i: gamma_correction(i, random.uniform(0.7, 1.3)),
            lambda i: hsv_augmentation(i, hue_delta=random.uniform(-15, 15), sat_scale=random.uniform(0.8, 1.2))
        ]

        # Apply a random augmentation
        augmentation = random.choice(augmentations)
        augmented_img = augmentation(img)

        # Save the augmented image
        output_dir = Path(output_dir_base) / "images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / Path(image_path).name
        augmented_img.save(output_path, "JPEG", quality=95)

        # Copy the corresponding label file
        label_path = str(image_path).replace("images", "labels").replace(".jpg", ".txt")
        if os.path.exists(label_path):
            output_label_dir = Path(output_dir_base) / "labels"
            os.makedirs(output_label_dir, exist_ok=True)
            shutil.copy2(label_path, output_label_dir / Path(label_path).name)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def gamma_correction(img, gamma):
    """Applies gamma correction to an image."""
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.power(img_array, 1.0 / gamma)
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def hsv_augmentation(img, hue_delta, sat_scale):
    """Applies HSV augmentation to an image."""
    img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * sat_scale, 0, 255)
    return Image.fromarray(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))

def main():
    """
    Augments images with lighting variations in parallel.
    """
    source_dir = "/Users/masdevid/Projects/smartcash/data/preprocessed/test"
    output_dir = "/Users/masdevid/Projects/smartcash/data/preprocessed/test_lighting"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    image_dir = Path(source_dir) / "images"
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return

    image_paths = list(image_dir.glob("*.jpg"))

    with Pool(8) as p:
        p.starmap(apply_lighting_augmentation, [(path, output_dir) for path in image_paths])

if __name__ == "__main__":
    main()
