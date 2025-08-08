"""
SmartCash YOLOv5 Training Example
Complete training workflow with both YOLOv5s and EfficientNet-B4 backbones
"""

import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from smartcash.model.architectures.smartcash_yolov5 import create_smartcash_yolov5
from smartcash.model.training.direct_training_manager import DirectTrainingManager
from smartcash.common.logger import SmartCashLogger

logger = SmartCashLogger(__name__)


def create_real_dataloader(data_dir, batch_size=8, limit_samples=100, shuffle=True):
    """Create data loader from real preprocessed data"""
    class SmartCashDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, limit_samples=None):
            self.data_dir = Path(data_dir)
            self.images_dir = self.data_dir / "images"
            self.labels_dir = self.data_dir / "labels"
            
            # Find all image files (.npy)
            self.image_files = sorted(list(self.images_dir.glob("*.npy")))
            
            # Apply sample limit if specified
            if limit_samples:
                self.image_files = self.image_files[:limit_samples]
            
            logger.info(f"📊 Found {len(self.image_files)} image files in {data_dir}")
            
        def __len__(self):
            return len(self.image_files)
            
        def __getitem__(self, idx):
            # Load image (.npy file)
            image_path = self.image_files[idx]
            image = np.load(image_path).astype(np.float32)
            
            # Convert from HWC to CHW if needed and ensure proper shape
            if image.shape[-1] == 3:  # HWC format
                image = image.transpose(2, 0, 1)  # Convert to CHW
            
            # Normalize to [0, 1] if not already
            if image.max() > 1.0:
                image = image / 255.0
                
            image_tensor = torch.from_numpy(image)
            
            # Load corresponding label file
            label_path = self.labels_dir / (image_path.stem + ".txt")
            
            labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            parts = line.split()
                            if len(parts) >= 5:
                                cls = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                labels.append([cls, x, y, w, h])
            
            # Convert labels to tensor, handle empty case
            if labels:
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
            else:
                # Empty tensor for images with no labels
                labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
                
            return image_tensor, labels_tensor
    
    dataset = SmartCashDataset(data_dir, limit_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)


def custom_collate_fn(batch):
    """Custom collate function to handle variable number of labels per image"""
    images = []
    all_labels = []
    
    for i, (image, labels) in enumerate(batch):
        images.append(image)
        
        # Add batch index to labels (required for YOLOv5 training)
        if len(labels) > 0:
            batch_labels = torch.cat([
                torch.full((len(labels), 1), i),  # Batch index
                labels
            ], dim=1)
            all_labels.append(batch_labels)
    
    # Stack images
    images = torch.stack(images)
    
    # Concatenate all labels
    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)
    else:
        # Empty batch
        all_labels = torch.zeros((0, 6), dtype=torch.float32)
        
    return images, all_labels


def train_yolov5s_example():
    """Example: Training YOLOv5s backbone"""
    logger.info("🚀 Training YOLOv5s Example")
    
    # Create model
    model = create_smartcash_yolov5(
        backbone="yolov5s",
        pretrained=False,
        device="cpu"
    )
    
    # Create data loaders from real preprocessed data
    train_loader = create_real_dataloader(
        data_dir="data/preprocessed/train", 
        batch_size=4, 
        limit_samples=80,  # 80 samples for training
        shuffle=True
    )
    val_loader = create_real_dataloader(
        data_dir="data/preprocessed/valid", 
        batch_size=4, 
        limit_samples=20,  # 20 samples for validation
        shuffle=False
    )
    
    # Create training manager
    trainer = DirectTrainingManager(
        model=model.model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="data/checkpoints/yolov5s"
    )
    
    logger.info("📊 Model Info:")
    logger.info(f"   Total parameters: {model.get_model_info()['total_params']:,}")
    logger.info(f"   Trainable parameters: {model.get_model_info()['trainable_params']:,}")
    
    # Phase 1: Head-only training
    logger.info("🔒 Phase 1: Head localization training")
    trainer.setup_phase_1(learning_rate=1e-3)
    
    phase1_history = trainer.train_phase_1(
        epochs=5,  # Short demo
        save_best=True,
        patience=3
    )
    
    logger.info(f"✅ Phase 1 completed: {len(phase1_history['train_loss'])} epochs")
    
    # Phase 2: Full model fine-tuning
    logger.info("🔓 Phase 2: Full model fine-tuning")
    trainer.setup_phase_2(learning_rate=1e-4)
    
    phase2_history = trainer.train_phase_2(
        epochs=5,  # Short demo
        save_best=True,
        patience=3,
        load_phase1_weights=True
    )
    
    logger.info(f"✅ Phase 2 completed: {len(phase2_history['train_loss'])} epochs")
    
    # Get training summary
    summary = trainer.get_training_summary()
    logger.info("📋 Training Summary:")
    logger.info(f"   Best mAP: {summary['best_map']:.4f}")
    logger.info(f"   Final phase: {summary['current_phase']}")
    logger.info(f"   Total epochs: {summary['current_epoch']}")
    
    return model, trainer


def train_efficientnet_b4_example():
    """Example: Training EfficientNet-B4 backbone"""
    logger.info("🚀 Training EfficientNet-B4 Example")
    
    # Create model
    model = create_smartcash_yolov5(
        backbone="efficientnet_b4",
        pretrained=True,  # Use pretrained backbone
        device="cpu"
    )
    
    # Create data loaders from real preprocessed data
    train_loader = create_real_dataloader(
        data_dir="data/preprocessed/train", 
        batch_size=2,  # Smaller batch for larger model
        limit_samples=80,  # 80 samples for training
        shuffle=True
    )
    val_loader = create_real_dataloader(
        data_dir="data/preprocessed/valid", 
        batch_size=2, 
        limit_samples=20,  # 20 samples for validation
        shuffle=False
    )
    
    # Create training manager
    trainer = DirectTrainingManager(
        model=model.model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="data/checkpoints/efficientnet_b4"
    )
    
    logger.info("📊 Model Info:")
    logger.info(f"   Total parameters: {model.get_model_info()['total_params']:,}")
    logger.info(f"   Trainable parameters: {model.get_model_info()['trainable_params']:,}")
    
    # Phase 1: Head-only training
    logger.info("🔒 Phase 1: Head localization training")
    trainer.setup_phase_1(learning_rate=5e-4)  # Lower LR for pretrained backbone
    
    phase1_history = trainer.train_phase_1(
        epochs=3,  # Short demo
        save_best=True,
        patience=2
    )
    
    logger.info(f"✅ Phase 1 completed: {len(phase1_history['train_loss'])} epochs")
    
    # Phase 2: Full model fine-tuning
    logger.info("🔓 Phase 2: Full model fine-tuning")
    trainer.setup_phase_2(learning_rate=1e-5)  # Very low LR for pretrained backbone
    
    phase2_history = trainer.train_phase_2(
        epochs=3,  # Short demo
        save_best=True,
        patience=2,
        load_phase1_weights=True
    )
    
    logger.info(f"✅ Phase 2 completed: {len(phase2_history['train_loss'])} epochs")
    
    # Get training summary
    summary = trainer.get_training_summary()
    logger.info("📋 Training Summary:")
    logger.info(f"   Best mAP: {summary['best_map']:.4f}")
    logger.info(f"   Final phase: {summary['current_phase']}")
    logger.info(f"   Total epochs: {summary['current_epoch']}")
    
    return model, trainer


def compare_backbones():
    """Compare training performance between backbones"""
    logger.info("⚖️ Backbone Comparison Training")
    
    backbones = ["yolov5s", "efficientnet_b4"]
    results = {}
    
    for backbone in backbones:
        logger.info(f"\n🧪 Training {backbone}")
        
        # Create model
        model = create_smartcash_yolov5(
            backbone=backbone,
            pretrained=(backbone == "efficientnet_b4"),
            device="cpu"
        )
        
        # Quick training test with real data
        train_loader = create_real_dataloader(
            data_dir="data/preprocessed/train", 
            batch_size=2, 
            limit_samples=50,  # 50 samples for quick comparison
            shuffle=True
        )
        val_loader = create_real_dataloader(
            data_dir="data/preprocessed/valid", 
            batch_size=2, 
            limit_samples=10,  # 10 samples for quick validation
            shuffle=False
        )
        
        trainer = DirectTrainingManager(
            model=model.model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=f"data/checkpoints/{backbone}_comparison"
        )
        
        # Quick phase 1 training
        trainer.setup_phase_1()
        history = trainer.train_phase_1(epochs=2, save_best=False, patience=10)
        
        model_info = model.get_model_info()
        results[backbone] = {
            'total_params': model_info['total_params'],
            'trainable_params': model_info['trainable_params'],
            'final_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'training_speed': len(history['train_loss'])  # epochs completed
        }
        
        logger.info(f"✅ {backbone}: {results[backbone]['total_params']:,} params, "
                   f"loss={results[backbone]['final_loss']:.4f}")
    
    # Summary comparison
    logger.info("\n📊 Backbone Comparison Results:")
    for backbone, result in results.items():
        logger.info(f"   {backbone}:")
        logger.info(f"     Parameters: {result['total_params']:,}")
        logger.info(f"     Trainable (P1): {result['trainable_params']:,}")
        logger.info(f"     Final Loss: {result['final_loss']:.4f}")
    
    return results


if __name__ == "__main__":
    try:
        # Example 1: YOLOv5s training
        logger.info("=" * 60)
        logger.info("EXAMPLE 1: YOLOv5s Training")
        logger.info("=" * 60)
        
        yolov5s_model, yolov5s_trainer = train_yolov5s_example()
        
        # Example 2: EfficientNet-B4 training  
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 2: EfficientNet-B4 Training")
        logger.info("=" * 60)
        
        efficientnet_model, efficientnet_trainer = train_efficientnet_b4_example()
        
        # Example 3: Backbone comparison
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 3: Backbone Comparison")
        logger.info("=" * 60)
        
        comparison_results = compare_backbones()
        
        logger.info("\n🎉 All training examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training example failed: {e}")
        import traceback
        traceback.print_exc()