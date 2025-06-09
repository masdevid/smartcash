"""
File: smartcash/dataset/augmentor/examples/usage_examples.py
Deskripsi: Contoh penggunaan augmentation service dengan progress tracker integration
"""

from smartcash.dataset.augmentor import create_augmentor, augment_and_normalize
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker

# Example 1: Basic usage dengan dual progress tracker
def example_with_dual_tracker():
    """Contoh penggunaan dengan dual progress tracker"""
    
    # Create progress tracker (dual)
    progress_tracker = create_dual_progress_tracker("Augmentation")
    
    # Configuration
    config = {
        'data': {'dir': 'data'},
        'augmentation': {
            'types': ['combined'],  # Default research type
            'num_variations': 3,
            'target_count': 500,
            'intensity': 0.7
        },
        'preprocessing': {
            'normalization': {
                'method': 'minmax',     # Default
                'denormalize': False,   # Default - save as normalized
                'target_size': [640, 640]  # Fixed for YOLO
            }
        }
    }
    
    # Run complete pipeline
    result = augment_and_normalize(
        config=config,
        target_split='train',
        progress_tracker=progress_tracker
    )
    
    return result

# Example 2: Manual callback untuk custom progress handling  
def example_with_custom_callback():
    """Contoh dengan custom progress callback"""
    
    def custom_progress_callback(level, current, total, message):
        print(f"[{level.upper()}] {current}/{total}: {message}")
    
    config = {
        'data': {'dir': '/content/data'},
        'augmentation': {
            'types': ['lighting'],  # Hanya lighting variations
            'num_variations': 2
        }
    }
    
    # Run dengan custom callback
    result = augment_and_normalize(
        config=config,
        target_split='train',
        progress_callback=custom_progress_callback
    )
    
    return result

# Example 3: Step-by-step dengan manual control
def example_stepwise_control():
    """Contoh kontrol manual step-by-step"""
    
    config = {
        'data': {'dir': 'data'},
        'augmentation': {
            'types': ['position', 'lighting'],  # Multiple types
            'num_variations': 2
        }
    }
    
    # Create service
    service = create_augmentor(config)
    
    # Step 1: Run augmentation only
    aug_result = service.engine.augment_split('train')
    print(f"✅ Augmentation: {aug_result['total_generated']} files")
    
    # Step 2: Run normalization only  
    norm_result = service.normalizer.normalize_augmented_files(
        service.path_resolver.get_augmented_path('train'),
        service.path_resolver.get_preprocessed_path('train')
    )
    print(f"✅ Normalization: {norm_result['total_normalized']} files")
    
    # Step 3: Create symlinks
    symlink_result = service._create_symlinks('train')
    print(f"✅ Symlinks: {symlink_result['total_created']} created")
    
    return {
        'augmentation': aug_result,
        'normalization': norm_result, 
        'symlinks': symlink_result
    }

# Example 4: Cleanup operations
def example_cleanup():
    """Contoh cleanup operations"""
    
    config = {'data': {'dir': 'data'}}
    
    # Cleanup specific split
    result = cleanup_augmented_data(config, target_split='train')
    print(f"🧹 Cleanup train: {result['total_removed']} files removed")
    
    # Cleanup all splits
    result = cleanup_augmented_data(config)
    print(f"🧹 Cleanup all: {result['total_removed']} files removed")
    
    return result

# Example 5: Configuration variations
def example_config_variations():
    """Contoh berbagai konfigurasi"""
    
    # Research config dengan normalization variations
    configs = {
        'minmax_normalized': {
            'preprocessing': {
                'normalization': {
                    'method': 'minmax',
                    'denormalize': False  # Save as float32 [0,1]
                }
            }
        },
        
        'minmax_denormalized': {
            'preprocessing': {
                'normalization': {
                    'method': 'minmax', 
                    'denormalize': True   # Save as uint8 [0,255]
                }
            }
        },
        
        'standard_normalized': {
            'preprocessing': {
                'normalization': {
                    'method': 'standard',
                    'denormalize': False  # Save as z-score
                }
            }
        },
        
        'imagenet_normalized': {
            'preprocessing': {
                'normalization': {
                    'method': 'imagenet',
                    'denormalize': False  # Save with ImageNet normalization
                }
            }
        }
    }
    
    results = {}
    
    for config_name, norm_config in configs.items():
        full_config = {
            'data': {'dir': 'data'},
            'augmentation': {'types': ['combined']},
            **norm_config
        }
        
        print(f"🧪 Testing {config_name}")
        result = augment_and_normalize(full_config, target_split='valid')
        results[config_name] = result
        
        print(f"   Generated: {result.get('total_generated', 0)}")
        print(f"   Normalized: {result.get('total_normalized', 0)}")
    
    return results

# Example 6: Integration dengan UI components  
def example_ui_integration(ui_components):
    """Contoh integrasi dengan UI components"""
    
    # Extract progress tracker dari UI components
    progress_tracker = ui_components.get('progress_tracker')
    
    # Extract config dari UI form values
    config = {
        'data': {'dir': getattr(ui_components.get('data_dir'), 'value', 'data')},
        'augmentation': {
            'types': list(getattr(ui_components.get('aug_types'), 'value', ['combined'])),
            'num_variations': getattr(ui_components.get('num_variations'), 'value', 2),
            'target_count': getattr(ui_components.get('target_count'), 'value', 500),
            'intensity': getattr(ui_components.get('intensity'), 'value', 0.7)
        },
        'preprocessing': {
            'normalization': {
                'method': getattr(ui_components.get('norm_method'), 'value', 'minmax'),
                'denormalize': getattr(ui_components.get('denormalize'), 'value', False)
            }
        }
    }
    
    # Custom callback untuk update UI status
    def ui_progress_callback(level, current, total, message):
        status_widget = ui_components.get('status_output')
        if status_widget:
            with status_widget:
                from IPython.display import display, HTML
                progress_pct = int((current / total) * 100) if total > 0 else 0
                html = f"""
                <div style="color: #28a745;">
                    📊 {message} ({progress_pct}%)
                </div>
                """
                display(HTML(html))
    
    # Run dengan UI integration
    result = augment_and_normalize(
        config=config,
        target_split='train',
        progress_tracker=progress_tracker,
        progress_callback=ui_progress_callback
    )
    
    # Update final status
    if result['status'] == 'success':
        ui_components.get('final_status', lambda x: None)(
            f"✅ Pipeline selesai: {result['total_generated']} generated, "
            f"{result['total_normalized']} normalized"
        )
    
    return result

if __name__ == "__main__":
    # Run examples
    print("🚀 Running augmentation examples...")
    
    # Basic example
    result1 = example_with_dual_tracker()
    print(f"Example 1: {result1['status']}")
    
    # Custom callback example
    result2 = example_with_custom_callback()
    print(f"Example 2: {result2['status']}")
    
    # Stepwise control
    result3 = example_stepwise_control()
    print(f"Example 3: Multiple phases completed")
    
    print("✅ All examples completed")