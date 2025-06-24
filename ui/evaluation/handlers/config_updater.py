from typing import Dict, Any

def update_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI dari config"""
    # Update checkpoint config
    checkpoint_config = config.get('checkpoint', {})
    if 'auto_select_best' in ui_components and 'auto_select_best' in checkpoint_config:
        ui_components['auto_select_best'].value = checkpoint_config['auto_select_best']
    if 'custom_checkpoint_path' in ui_components and 'custom_checkpoint_path' in checkpoint_config:
        ui_components['custom_checkpoint_path'].value = checkpoint_config['custom_checkpoint_path']
    if 'validation_metrics' in ui_components and 'validation_metrics' in checkpoint_config:
        ui_components['validation_metrics'].value = checkpoint_config['validation_metrics']
    
    # Update test data config
    test_data_config = config.get('test_data', {})
    if 'test_folder' in ui_components and 'test_folder' in test_data_config:
        ui_components['test_folder'].value = test_data_config['test_folder']
    if 'apply_augmentation' in ui_components and 'apply_augmentation' in test_data_config:
        ui_components['apply_augmentation'].value = test_data_config['apply_augmentation']
    if 'batch_size' in ui_components and 'batch_size' in test_data_config:
        ui_components['batch_size'].value = test_data_config['batch_size']
    if 'image_size' in ui_components and 'image_size' in test_data_config:
        ui_components['image_size'].value = test_data_config['image_size']
    if 'confidence_threshold' in ui_components and 'confidence_threshold' in test_data_config:
        ui_components['confidence_threshold'].value = test_data_config['confidence_threshold']
    if 'iou_threshold' in ui_components and 'iou_threshold' in test_data_config:
        ui_components['iou_threshold'].value = test_data_config['iou_threshold']