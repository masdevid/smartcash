"""
File: smartcash/ui/dataset/preprocessing_visualization_integration.py
Deskripsi: Integrasi visualisasi distribusi kelas ke modul preprocessing dengan pendekatan modular
"""

from typing import Dict, Any
from IPython.display import display

def setup_preprocessing_visualization(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup integrasi visualisasi distribusi kelas untuk modul preprocessing.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Import visualisasi integrator untuk shared functionality
    try:
        from smartcash.ui.dataset.visualization_integrator import setup_visualization_handlers
        from smartcash.ui.utils.constants import ICONS
        
        # Setup handlers visualisasi dengan kustomisasi untuk preprocessing
        ui_components = setup_visualization_handlers(
            ui_components, 
            env, 
            config,
            context="preprocessing"  # Tambahkan context untuk kustomisasi
        )
        
        # Tambahkan tombol visualisasi distribusi ke container yang tepat
        if 'visualization_button_container' in ui_components:
            # Jika sudah ada container tombol visualisasi, gunakan itu
            button_container = ui_components['visualization_button_container']
            
            # Cek apakah sudah ada tombol distribusi kelas
            if 'distribution_button' in ui_components:
                # Jika sudah ada, pindahkan tombol ke container jika belum ada di sana
                distribution_button = ui_components['distribution_button']
                
                # Pastikan distribusi button berada dalam container
                if not any(child is distribution_button for child in button_container.children):
                    # Add button to container
                    import ipywidgets as widgets
                    new_children = list(button_container.children) + [distribution_button]
                    button_container.children = tuple(new_children)
                    
                    if logger: logger.info(f"{ICONS.get('success', '✅')} Tombol distribusi kelas ditambahkan ke container tombol visualisasi")
        
        # Tambahkan kustomisasi khusus preprocessing jika ada
        # (misal, korelasi antara preprocessing dan distribusi kelas)
        if logger: logger.info(f"{ICONS.get('success', '✅')} Visualisasi distribusi kelas berhasil diintegrasikan ke preprocessing")
        
        return ui_components
    
    except ImportError as e:
        if logger: 
            logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak dapat mengimport visualization_integrator: {str(e)}")
        return ui_components
    except Exception as e:
        if logger:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat integrasi distribusi kelas: {str(e)}")
        return ui_components
        
def analyze_preprocessing_effect(ui_components: Dict[str, Any], dataset_dir: str) -> None:
    """
    Analisis efek preprocessing terhadap distribusi kelas.
    
    Args:
        ui_components: Dictionary komponen UI
        dataset_dir: Path direktori dataset
    """
    try:
        from smartcash.ui.helpers.class_distribution_analyzer import (
            analyze_class_distribution,
            analyze_class_distribution_by_prefix,
            count_files_by_prefix
        )
        
        import matplotlib.pyplot as plt
        
        # Lakukan analisis untuk raw dan preprocessed
        # TODO: Implementasi pembandingan distribusi kelas sebelum dan sesudah preprocessing
        # (Anda bisa mengimplementasikan logika ini lebih lanjut sesuai kebutuhan)
        
        # Contoh sederhana: tampilkan visualisasi distribusi
        try:
            from smartcash.ui.dataset.visualization_integrator import create_distribution_visualizations
            
            # Dapatkan output widget
            output_widget = ui_components.get('visualization_container', ui_components.get('status'))
            if output_widget:
                # Tampilkan visualisasi distribusi kelas
                create_distribution_visualizations(
                    ui_components=ui_components,
                    dataset_dir=dataset_dir,
                    split_name='train',  # Fokus pada split train
                    aug_prefix='aug',    # Default prefix untuk augmentasi
                    orig_prefix='rp',    # Default prefix untuk original
                    target_count=1000    # Target count untuk balancing
                )
        except ImportError:
            # Jika visualization_integrator tidak tersedia, gunakan matplotlib langsung
            class_counts = analyze_class_distribution(dataset_dir, 'train')
            
            # Buat visualisasi sederhana jika ada data
            if class_counts:
                plt.figure(figsize=(10, 6))
                plt.bar(list(class_counts.keys()), list(class_counts.values()))
                plt.title('Distribusi Kelas Setelah Preprocessing')
                plt.xlabel('Kelas')
                plt.ylabel('Jumlah Instance')
                plt.grid(axis='y', alpha=0.3)
                plt.show()
    
    except ImportError as e:
        logger = ui_components.get('logger')
        if logger:
            from smartcash.ui.utils.constants import ICONS
            logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak dapat menganalisis efek preprocessing: {str(e)}")
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            from smartcash.ui.utils.constants import ICONS
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error analisis efek preprocessing: {str(e)}")