"""
File: smartcash/ui/dataset/split_config_visualization.py
Deskripsi: Komponen visualisasi dataset untuk split config yang menampilkan data mentah dan preprocessed secara on-demand
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
        # from smartcash.common.constants import DRIVE_DATASET_PATH, DRIVE_PREPROCESSED_PATH
        drive_mounted = env and getattr(env, 'is_drive_mounted', False)
        base_path = '/content/drive/MyDrive/SmartCash/data' if drive_mounted else 'data'
        return (
            config.get('data', {}).get('dataset_path', base_path),
            config.get('data', {}).get('preprocessed_path', f'{base_path}/preprocessed')
        )

    @staticmethod
    def count_files(dataset_dir: str) -> Dict[str, Dict[str, int]]:
        """Count files in YOLO dataset structure."""
        stats = {}
        for split in ['train', 'valid', 'test']:
            split_dir = Path(dataset_dir) / split
            stats[split] = {
                'images': len(list((split_dir / 'images').glob('*.*'))) if (split_dir / 'images').exists() else 0,
                'labels': len(list((split_dir / 'labels').glob('*.txt'))) if (split_dir / 'labels').exists() else 0
            }
            stats[split]['valid'] = stats[split]['images'] > 0 and stats[split]['labels'] > 0
        return stats

    @staticmethod
    def get_stats(config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
        """Get dataset statistics for raw and preprocessed data."""
        try:
            raw_path, preprocessed_path = DatasetStatsManager.get_paths(config, env)
            stats = {
                'raw': {'exists': os.path.exists(raw_path), 'stats': {}},
                'preprocessed': {'exists': os.path.exists(preprocessed_path), 'stats': {}}
            }
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
        """Map class IDs to names."""
        try:
            from smartcash.common.layer_config import get_layer_config_manager
            class_map = get_layer_config_manager().get_class_map() if get_layer_config_manager() else {}
            return {class_map.get(cid, f"Class {cid}"): count for cid, count in split_classes.items()}
        except ImportError:
            return {f"Class {cid}": count for cid, count in split_classes.items()}

    @staticmethod
    def _dummy_distribution() -> Dict[str, Dict[str, int]]:
        """Return dummy class distribution."""
        return {
            'train': {'Rp1000': 120, 'Rp2000': 110, 'Rp5000': 130, 'Rp10000': 140, 'Rp20000': 125, 'Rp50000': 115, 'Rp100000': 135},
            'valid': {'Rp1000': 30, 'Rp2000': 25, 'Rp5000': 35, 'Rp10000': 40, 'Rp20000': 35, 'Rp50000': 28, 'Rp100000': 32},
            'test': {'Rp1000': 30, 'Rp2000': 25, 'Rp5000': 35, 'Rp10000': 35, 'Rp20000': 30, 'Rp50000': 27, 'Rp100000': 33}
        }

def update_stats_cards(html_component, stats: Dict[str, Any], colors: Dict[str, str]) -> None:
    """Update HTML component with dataset stats cards."""
    from smartcash.ui.utils.constants import ICONS
    raw, preprocessed = stats.get('raw', {'stats': {}}), stats.get('preprocessed', {'stats': {}})
    
    def generate_card(title: str, icon: str, color: str, data: Dict[str, Any]) -> str:
        images = sum(s.get('images', 0) for s in data['stats'].values())
        labels = sum(s.get('labels', 0) for s in data['stats'].values())
        card = (
            f'<div style="flex:1; min-width:220px; border:1px solid {color}; border-radius:5px; padding:10px; background-color:{colors["light"]}">'
            f'<h4 style="margin-top:0; color:{color}">{icon} {title}</h4>'
            f'<p style="margin:5px 0; font-weight:bold; font-size:1.2em; color:{colors["dark"]}">{images} images / {labels} labels</p>'
            '<div style="display:flex; flex-wrap:wrap; gap:5px;">'
        )
        for split, sdata in data['stats'].items():
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
    
    with output_widget:
        clear_output(wait=True)
        display(HTML(f'<div style="text-align:center; padding:10px;"><h3 style="color:{COLORS["dark"]}">{ICONS["chart"]} Loading dataset visualization...</h3></div>'))
        
        try:
            dataset_path, _ = DatasetStatsManager.get_paths(config, env)
            if not os.path.exists(dataset_path):
                display(HTML(
                    f'<div style="padding:10px; background-color:{COLORS["alert_warning_bg"]}; '
                    f'border-left:4px solid {COLORS["alert_warning_text"]}; color:{COLORS["alert_warning_text"]}; border-radius:4px;">'
                    f'<p>{ICONS["warning"]} Dataset not found at: {dataset_path}</p>'
                    f'<p>Ensure dataset is downloaded or path is correct.</p></div>'
                ))
                return

            class_distribution = DatasetStatsManager.get_class_distribution(config, env, logger)
            DatasetStatsManager._plot_distribution(class_distribution, dataset_path, logger)
        except Exception as e:
            if logger: logger.error(f"❌ Error during visualization: {str(e)}")
            display(HTML(
                f'<div style="padding:10px; background-color:{COLORS["alert_danger_bg"]}; '
                f'color:{COLORS["alert_danger_text"]}; border-radius:4px;">'
                f'<p>{ICONS["error"]} Visualization error: {str(e)}</p></div>'
            ))
def load_and_display_dataset_stats(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> None:
    """Load and display basic dataset stats."""
    try:
        stats = DatasetStatsManager.get_stats(config, env, logger)
        if 'current_stats_html' in ui_components:
            from smartcash.ui.utils.constants import COLORS
            update_stats_cards(ui_components['current_stats_html'], stats, COLORS)
        if logger: logger.info("✅ Basic dataset stats displayed successfully")
    except Exception as e:
        if logger: logger.error(f"❌ Error displaying dataset stats: {str(e)}")
        if 'output_box' in ui_components:
            with ui_components['output_box']:
                from smartcash.ui.utils.constants import ICONS, COLORS
                clear_output(wait=True)
                display(HTML(
                    f'<div style="padding:10px; background-color:{COLORS["alert_danger_bg"]}; '
                    f'color:{COLORS["alert_danger_text"]}; border-radius:4px;">'
                    f'<p>{ICONS["error"]} Error displaying stats: {str(e)}</p></div>'
                ))