"""
File: smartcash/ui/dataset/split_config_visualization.py
Deskripsi: Komponen visualisasi dataset untuk split config dengan format kelas yang dioptimalkan dan clearing loading message
"""

from typing import Dict, Any, Tuple, Optional
import os
from pathlib import Path
from IPython.display import display, HTML, clear_output

class DatasetStatsManager:
    """Manages dataset statistics and visualization operations."""
    
    @staticmethod
    def get_paths(config: Dict[str, Any], env=None) -> Tuple[str, str]:
        """Get raw and preprocessed dataset paths."""
        drive_mounted = env and getattr(env, 'is_drive_mounted', False)
        base_path = '/content/drive/MyDrive/SmartCash/data' if drive_mounted else 'data'
        
        # Default paths
        dataset_path = base_path
        preprocessed_path = f'{base_path}/preprocessed'
        
        # Get from config if available
        if config and isinstance(config, dict) and 'data' in config and isinstance(config['data'], dict):
            dataset_path = config['data'].get('dataset_path', base_path)
            preprocessed_path = config['data'].get('preprocessed_path', f'{base_path}/preprocessed')
            
        return (dataset_path, preprocessed_path)

    @staticmethod
    def count_files(dataset_dir: str) -> Dict[str, Dict[str, int]]:
        """Count files in YOLO dataset structure."""
        stats = {}
        
        # Check if directory exists
        if not os.path.exists(dataset_dir): return stats
            
        for split in ['train', 'valid', 'test']:
            split_dir = Path(dataset_dir) / split
            
            # Initialize counters
            stats[split] = {'images': 0, 'labels': 0, 'valid': False}
            
            # Count files if directories exist
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if images_dir.exists(): stats[split]['images'] = len(list(images_dir.glob('*.*')))
            if labels_dir.exists(): stats[split]['labels'] = len(list(labels_dir.glob('*.txt')))
            stats[split]['valid'] = stats[split]['images'] > 0 and stats[split]['labels'] > 0
            
        return stats

    @staticmethod
    def get_stats(config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
        """Get dataset statistics for raw and preprocessed data."""
        try:
            # Default empty stats
            stats = {'raw': {'exists': False, 'stats': {}}, 'preprocessed': {'exists': False, 'stats': {}}}
            
            # Get paths safely
            if config is None: config = {}
            raw_path, preprocessed_path = DatasetStatsManager.get_paths(config, env)
            
            # Update stats with existence info
            stats['raw']['exists'] = os.path.exists(raw_path)
            stats['preprocessed']['exists'] = os.path.exists(preprocessed_path)
            
            # Count files if directories exist
            if stats['raw']['exists']:
                stats['raw']['stats'] = DatasetStatsManager.count_files(raw_path)
                if logger: logger.info(f"📊 Raw dataset stats computed: {raw_path}")
                
            if stats['preprocessed']['exists']:
                stats['preprocessed']['stats'] = DatasetStatsManager.count_files(preprocessed_path)
                if logger: logger.info(f"📊 Preprocessed dataset stats computed: {preprocessed_path}")
                
            return stats
        except Exception as e:
            if logger: logger.error(f"❌ Error getting dataset stats: {str(e)}")
            return {'raw': {'exists': False, 'stats': {}}, 'preprocessed': {'exists': False, 'stats': {}}}

    @staticmethod
    def get_class_distribution(config: Dict[str, Any], env=None, logger=None) -> Dict[str, Dict[str, int]]:
        """Get class distribution from dataset labels."""
        try:
            dataset_path, _ = DatasetStatsManager.get_paths(config, env)
            if not os.path.exists(dataset_path):
                if logger: logger.warning(f"⚠️ Dataset path not found: {dataset_path}")
                return DatasetStatsManager._dummy_distribution()

            class_stats = {}
            for split in ['train', 'valid', 'test']:
                labels_dir = Path(dataset_path) / split / 'labels'
                if not labels_dir.exists():
                    class_stats[split] = {}
                    continue

                split_classes = {}
                for label_file in labels_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.split()[0])
                                    split_classes[class_id] = split_classes.get(class_id, 0) + 1
                    except Exception as e:
                        if logger: logger.debug(f"⚠️ Error reading label {label_file}: {str(e)}")

                class_stats[split] = DatasetStatsManager._map_class_names(split_classes, logger)
            if logger: logger.info("📊 Class distribution analyzed successfully")
            return class_stats
        except Exception as e:
            if logger: logger.error(f"❌ Error analyzing class distribution: {str(e)}")
            return DatasetStatsManager._dummy_distribution()

    @staticmethod
    def _map_class_names(split_classes: Dict[int, int], logger=None) -> Dict[str, int]:
        """Map class IDs to names with formatted ID notation."""
        try:
            from smartcash.common.layer_config import get_layer_config_manager
            class_map = get_layer_config_manager().get_class_map() if get_layer_config_manager() else {}
            # Format: '{id}:Class_name (id_name)'
            return {f"{cid}:{class_map.get(cid, f'Class {cid}')} (l2_{str(cid).zfill(3)})": count 
                   for cid, count in split_classes.items()}
        except ImportError:
            return {f"{cid}:Class {cid} (l2_{str(cid).zfill(3)})": count 
                   for cid, count in split_classes.items()}

    @staticmethod
    def _dummy_distribution() -> Dict[str, Dict[str, int]]:
        """Return dummy class distribution with ID-prefixed class names."""
        return {
            'train': {'0:Rp1000 (l2_000)': 120, '1:Rp2000 (l2_001)': 110, '2:Rp5000 (l2_002)': 130, 
                     '3:Rp10000 (l2_003)': 140, '4:Rp20000 (l2_004)': 125, '5:Rp50000 (l2_005)': 115, 
                     '6:Rp100000 (l2_006)': 135},
            'valid': {'0:Rp1000 (l2_000)': 30, '1:Rp2000 (l2_001)': 25, '2:Rp5000 (l2_002)': 35, 
                     '3:Rp10000 (l2_003)': 40, '4:Rp20000 (l2_004)': 35, '5:Rp50000 (l2_005)': 28, 
                     '6:Rp100000 (l2_006)': 32},
            'test': {'0:Rp1000 (l2_000)': 30, '1:Rp2000 (l2_001)': 25, '2:Rp5000 (l2_002)': 35, 
                    '3:Rp10000 (l2_003)': 35, '4:Rp20000 (l2_004)': 30, '5:Rp50000 (l2_005)': 27, 
                    '6:Rp100000 (l2_006)': 33}
        }

def update_stats_cards(html_component, stats: Dict[str, Any], colors: Dict[str, str]) -> None:
    """Update HTML component with dataset stats cards."""
    from smartcash.ui.utils.constants import ICONS
    
    # Safe access to stats
    if not stats: stats = {'raw': {'stats': {}}, 'preprocessed': {'stats': {}}}
    raw, preprocessed = stats.get('raw', {'stats': {}}), stats.get('preprocessed', {'stats': {}})
    
    def generate_card(title: str, icon: str, color: str, data: Dict[str, Any]) -> str:
        # Safe access to stats data
        stats_data = data.get('stats', {})
        
        images = sum(s.get('images', 0) for s in stats_data.values())
        labels = sum(s.get('labels', 0) for s in stats_data.values())
        
        card = (
            f'<div style="flex:1; min-width:220px; border:1px solid {color}; border-radius:5px; padding:10px; background-color:{colors["light"]}">'
            f'<h4 style="margin-top:0; color:{color}">{icon} {title}</h4>'
            f'<p style="margin:5px 0; font-weight:bold; font-size:1.2em; color:{colors["dark"]}">{images} images / {labels} labels</p>'
            '<div style="display:flex; flex-wrap:wrap; gap:5px;">'
        )
        
        for split, sdata in stats_data.items():
            split_color = colors['success'] if sdata.get('valid', False) else colors['danger']
            card += (
                f'<div style="padding:5px; margin:2px; border-radius:3px; background-color:{colors["light"]}; border:1px solid {split_color}">'
                f'<strong style="color:{split_color}">{split.capitalize()}</strong>: <span style="color:#3795BD">{sdata.get("images", 0)}</span>'
                '</div>'
            )
            
        return card + '</div></div>'

    html_component.value = (
        f'<h3 style="margin:10px 0; color:{colors["dark"]}">{ICONS["dataset"]} Dataset Statistics</h3>'
        '<div style="display:flex; flex-wrap:wrap; gap:15px; margin-bottom:15px">'
        f'{generate_card("Raw Dataset", ICONS["folder"], colors["primary"], raw)}'
        f'{generate_card("Preprocessed Dataset", ICONS["processing"], colors["secondary"], preprocessed)}'
        '</div>'
    )

def show_distribution_visualization(output_widget, config: Dict[str, Any], env=None, logger=None) -> None:
    """Display class distribution visualization."""
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Penting: clear output terlebih dahulu sebelum menampilkan loading
    with output_widget:
        clear_output(wait=True)
        display(HTML(f'<div style="text-align:center; padding:10px;"><h3 style="color:{COLORS["dark"]}">{ICONS["chart"]} Loading dataset visualization...</h3></div>'))
        
        try:
            dataset_path, _ = DatasetStatsManager.get_paths(config, env)
            if not os.path.exists(dataset_path):
                clear_output(wait=True)  # Clear loading message
                display(HTML(
                    f'<div style="padding:10px; background-color:{COLORS["alert_warning_bg"]}; '
                    f'border-left:4px solid {COLORS["alert_warning_text"]}; color:{COLORS["alert_warning_text"]}; border-radius:4px;">'
                    f'<p>{ICONS["warning"]} Dataset not found at: {dataset_path}</p>'
                    f'<p>Ensure dataset is downloaded or path is correct.</p></div>'
                ))
                return

            class_distribution = DatasetStatsManager.get_class_distribution(config, env, logger)
            # Penting: clear loading message sebelum menampilkan plot
            clear_output(wait=True)
            _plot_distribution(class_distribution, dataset_path, logger)
        except Exception as e:
            if logger: logger.error(f"❌ Error during visualization: {str(e)}")
            # Penting: clear loading message sebelum menampilkan error
            clear_output(wait=True)
            display(HTML(
                f'<div style="padding:10px; background-color:{COLORS["alert_danger_bg"]}; '
                f'color:{COLORS["alert_danger_text"]}; border-radius:4px;">'
                f'<p>{ICONS["error"]} Visualization error: {str(e)}</p></div>'
            ))

def _plot_distribution(class_distribution: Dict[str, Dict[str, int]], dataset_path: str, logger=None) -> None:
    """Plot class distribution using matplotlib."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from smartcash.ui.utils.constants import COLORS, ICONS

        # Extract and prepare data from distribution
        all_classes = sorted(set().union(*[set(d.keys()) for d in class_distribution.values()]))
        
        # Create DataFrame with class as index
        df = pd.DataFrame([
            {'Class': cls, 'Train': class_distribution.get('train', {}).get(cls, 0),
             'Valid': class_distribution.get('valid', {}).get(cls, 0),
             'Test': class_distribution.get('test', {}).get(cls, 0)}
            for cls in all_classes
        ])

        # Plot the distribution
        plt.figure(figsize=(12, 6))
        bar_width = 0.25
        r1, r2, r3 = np.arange(len(df)), [x + bar_width for x in range(len(df))], [x + 2 * bar_width for x in range(len(df))]
        plt.bar(r1, df['Train'], width=bar_width, label='Train', color=COLORS['primary'])
        plt.bar(r2, df['Valid'], width=bar_width, label='Valid', color=COLORS['success'])
        plt.bar(r3, df['Test'], width=bar_width, label='Test', color=COLORS['warning'])
        plt.xlabel('Class'), plt.ylabel('Sample Count'), plt.title('Class Distribution per Dataset Split')
        plt.xticks([r + bar_width for r in range(len(df))], df['Class'], rotation=45, ha='right')
        plt.legend(), plt.tight_layout(), plt.show()

        # Show additional info after plot
        display(HTML(
            f'<div style="padding:10px; background-color:{COLORS["alert_info_bg"]}; '
            f'color:{COLORS["alert_info_text"]}; border-radius:4px; margin-top:15px;">'
            f'<p>{ICONS["info"]} <strong>Dataset Info:</strong> Visualization shows class distribution across splits.</p>'
            f'<p>Dataset path: <code>{dataset_path}</code></p></div>'
            f'<h3>{ICONS["chart"]} Class Distribution Table</h3>'
        ))
        display(df.style.background_gradient(cmap='Blues', subset=['Train', 'Valid', 'Test']))
    except Exception as e:
        if logger: logger.error(f"❌ Error creating visualization: {str(e)}")
        display(HTML(
            f'<div style="padding:10px; background-color:{COLORS["alert_danger_bg"]}; '
            f'color:{COLORS["alert_danger_text"]}; border-radius:4px;">'
            f'<p>{ICONS["error"]} Visualization creation error: {str(e)}</p></div>'
        ))

def load_and_display_dataset_stats(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> None:
    """Load and display basic dataset stats."""
    try:
        # Validasi input
        if ui_components is None:
            if logger: logger.error("❌ ui_components tidak boleh None")
            return
            
        if config is None: config = {}
            
        # Get stats
        stats = DatasetStatsManager.get_stats(config, env, logger)
        
        # Update UI jika ada
        if 'current_stats_html' in ui_components:
            from smartcash.ui.utils.constants import COLORS
            update_stats_cards(ui_components['current_stats_html'], stats, COLORS)
            
        if logger: logger.info("✅ Basic dataset stats displayed successfully")
    except Exception as e:
        if logger: logger.error(f"❌ Error displaying dataset stats: {str(e)}")
        
        # Tampilkan error jika ada output widget
        if ui_components is not None and 'output_box' in ui_components:
            with ui_components['output_box']:
                from smartcash.ui.utils.constants import ICONS, COLORS
                clear_output(wait=True)  # Pastikan loading message dihapus
                display(HTML(
                    f'<div style="padding:10px; background-color:{COLORS["alert_danger_bg"]}; '
                    f'color:{COLORS["alert_danger_text"]}; border-radius:4px;">'
                    f'<p>{ICONS["error"]} Error displaying stats: {str(e)}</p></div>'
                ))