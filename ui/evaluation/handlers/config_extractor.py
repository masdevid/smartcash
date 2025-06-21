from typing import Dict, Any

def extract_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components"""
    config = {}
    # Ekstrak checkpoint config
    checkpoint_config = {}
    if 'auto_select_best' in ui_components:
        checkpoint_config['auto_select_best'] = ui_components['auto_select_best'].value
    if 'custom_checkpoint_path' in ui_components:
        checkpoint_config['custom_checkpoint_path'] = ui_components['custom_checkpoint_path'].value
    if 'validation_metrics' in ui_components:
        checkpoint_config['validation_metrics'] = ui_components['validation_metrics'].value
        
    # Ekstrak test data config
    test_data_config = {}
    if 'test_folder' in ui_components:
        test_data_config['test_folder'] = ui_components['test_folder'].value
    if 'apply_augmentation' in ui_components:
        test_data_config['apply_augmentation'] = ui_components['apply_augmentation'].value
    if 'batch_size' in ui_components:
        test_data_config['batch_size'] = ui_components['batch_size'].value
    if 'image_size' in ui_components:
        test_data_config['image_size'] = ui_components['image_size'].value
    if 'confidence_threshold' in ui_components:
        test_data_config['confidence_threshold'] = ui_components['confidence_threshold'].value
    if 'iou_threshold' in ui_components:
        test_data_config['iou_threshold'] = ui_components['iou_threshold'].value
        
    # Combine configs
    config['checkpoint'] = checkpoint_config
    config['test_data'] = test_data_config
    
    return config