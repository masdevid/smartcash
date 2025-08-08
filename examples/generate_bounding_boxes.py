
import os
import cv2
import shutil
from multiprocessing import Pool
from pathlib import Path
from functools import partial

# Define a list of colors for different classes
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
]

def draw_bounding_box(image_path, output_dir_base):
    """
    Draws bounding boxes on an image based on a corresponding label file.

    Args:
        image_path (str): The path to the image file.
        output_dir_base (str): The base directory for the output images.
    """
    image_path = Path(image_path)
    label_path = image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"
    image = cv2.imread(str(image_path))
    h, w, _ = image.shape

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                color = COLORS[class_id % len(COLORS)]

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    str(class_id),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

    source_folder_name = image_path.parent.parent.name
    output_dir = Path(output_dir_base) / source_folder_name / 'images'
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)

def main():
    """
    Generates images with bounding boxes in parallel from multiple sources.
    """
    output_dir_base = "/Users/masdevid/Projects/smartcash/data/evaluation/samples"
    if os.path.exists(output_dir_base):
        shutil.rmtree(output_dir_base)
    os.makedirs(output_dir_base)

    source_dirs = [
        "/Users/masdevid/Projects/smartcash/data/evaluation/lighting_variation",
        "/Users/masdevid/Projects/smartcash/data/evaluation/position_variation",
        "/Users/masdevid/Projects/smartcash/data/valid",
        "/Users/masdevid/Projects/smartcash/data/test",
        "/Users/masdevid/Projects/smartcash/data/test_lighting",
        "/Users/masdevid/Projects/smartcash/data/augmented/train",
    ]

    for source_dir in source_dirs:
        image_dir = Path(source_dir) / "images"
        if not image_dir.exists():
            print(f"Image directory not found: {image_dir}")
            continue

        image_paths = list(image_dir.glob("*.jpg"))
        
        # Create a partial function with the output_dir_base argument fixed
        draw_func = partial(draw_bounding_box, output_dir_base=output_dir_base)

        with Pool(8) as p:
            p.map(draw_func, image_paths)

if __name__ == "__main__":
    main()
