# 📊 Dataset Technical Documentation

## 📋 Overview

SmartCash dataset terdiri dari gambar uang kertas Rupiah dalam berbagai kondisi pengambilan gambar. Dataset ini dirancang untuk melatih model deteksi yang robust terhadap variasi pencahayaan, rotasi, dan oklusi.

## 🏗️ Dataset Structure

### 1. Directory Layout
```
data/
├── raw/                    # Data mentah
│   ├── images/            # Gambar original
│   └── labels/            # Label YOLO format
├── processed/             # Data terproses
│   ├── train/            # Training split (70%)
│   │   ├── images/      
│   │   └── labels/      
│   ├── valid/            # Validation split (15%)
│   │   ├── images/      
│   │   └── labels/      
│   └── test/             # Testing split (15%)
│       ├── images/      
│       └── labels/      
└── augmented/            # Data hasil augmentasi
    └── train/            # Hanya untuk training
        ├── images/      
        └── labels/      
```

### 2. File Format

#### Images
- Format: JPG/JPEG
- Resolusi: 640x640 pixels
- Channels: RGB
- Metadata: Exif (optional)

#### Labels (YOLO format)
```
<class_id> <x_center> <y_center> <width> <height>
```
Example:
```
2 0.523 0.642 0.456 0.178
```

## 🔄 Preprocessing Pipeline

### 1. Image Processing
```python
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess single image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image array
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (640, 640))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    return image
```

### 2. Label Processing
```python
def process_label(
    label_path: str,
    img_size: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """
    Process YOLO format label.
    
    Args:
        label_path: Path to label file
        img_size: Target image size
        
    Returns:
        Processed label array
    """
    # Read label
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f]
    
    # Convert to array
    labels = np.array(labels, dtype=np.float32)
    
    # Normalize coordinates
    if len(labels):
        labels[:, [1,3]] /= img_size[0]  # x
        labels[:, [2,4]] /= img_size[1]  # y
    
    return labels
```
