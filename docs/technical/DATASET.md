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

## 🔍 Data Validation

### 1. Image Validation
```python
def validate_image(image_path: str) -> bool:
    """Validate image file."""
    try:
        # Check file exists
        if not os.path.exists(image_path):
            return False
            
        # Check file size
        if os.path.getsize(image_path) < 1024:  # Min 1KB
            return False
            
        # Try loading image
        img = Image.open(image_path)
        img.verify()
        
        return True
        
    except Exception:
        return False
```

### 2. Label Validation
```python
def validate_label(
    label_path: str,
    num_classes: int = 7
) -> bool:
    """Validate label file."""
    try:
        # Check file exists
        if not os.path.exists(label_path):
            return False
            
        # Read labels
        labels = np.loadtxt(label_path)
        
        # Check format
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)
            
        if labels.shape[1] != 5:  # class, x, y, w, h
            return False
            
        # Check values
        if not (labels[:,0] < num_classes).all():  # class_id
            return False
            
        if not (labels[:,1:] <= 1.0).all():  # normalized coords
            return False
            
        return True
        
    except Exception:
        return False
```

## 📊 Dataset Statistics

### 1. Class Distribution
```
Class     Count    Percentage
--------------------------------
1000      2500     14.3%
2000      2500     14.3%
5000      2500     14.3%
10000     2500     14.3%
20000     2500     14.3%
50000     2500     14.3%
100000    2500     14.3%
--------------------------------
Total     17500    100%
```

### 2. Image Properties
- Mean Resolution: 640x640
- Mean File Size: 145KB
- Color Space: RGB
- Bit Depth: 8-bit

### 3. Augmentation Stats
- Original Images: 17,500
- Augmented Images: 52,500
- Total Dataset: 70,000

## 🛠️ Data Pipeline

### 1. Data Loading
```python
class RupiahDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Setup paths
        split_dir = self.data_dir / split
        self.img_dir = split_dir / 'images'
        self.lbl_dir = split_dir / 'labels'
        
        # Get files
        self.img_files = sorted(self.img_dir.glob('*.jpg'))
        self.lbl_files = [
            self.lbl_dir / f"{img.stem}.txt"
            for img in self.img_files
        ]
```

### 2. Augmentation Pipeline
```python
train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels']
))
```

## 🔄 Data Versioning

### 1. Version Control
- Dataset versioning dengan DVC
- Metadata tracking
- Experiment logging

### 2. Version Format
```
v{major}.{minor}.{patch}
Example: v1.2.3
```

### 3. Changelog Format
```markdown
# Changelog

## [1.2.3] - 2025-02-20
### Added
- 500 new images for each class
- Lighting variation dataset

### Changed
- Updated label format
- Improved image quality

### Fixed
- Incorrect labels in test set
- Duplicate images removed
```

## 📈 Quality Metrics

### 1. Image Quality
- Resolution check
- Blur detection
- Noise level
- Contrast ratio

### 2. Label Quality
- Bounding box overlap
- Class balance
- Annotation consistency
- Coordinate validation

### 3. Dataset Balance
- Class distribution
- Image conditions
- Size variations
- Rotation diversity