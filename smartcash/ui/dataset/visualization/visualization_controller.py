"""
File: smartcash/ui/dataset/visualization/visualization_controller.py
Deskripsi: Controller utama untuk visualisasi dataset
"""

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor import get_preprocessing_stats
import ipywidgets as widgets
from IPython.display import display

logger = get_logger(__name__)

class VisualizationController:
    """Controller utama untuk visualisasi dataset"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inisialisasi controller visualisasi
        
        Args:
            config: Konfigurasi visualisasi
        """
        self.config = config or {}
        self.ui_components = {}
        self.current_dataset = None
        self.dataset_stats = {}
        
        # Inisialisasi komponen
        self.stats_component = DatasetStatsComponent(
            config=self.config.get('stats', {})
        )
        
        # Inisialisasi visualizer augmentasi (akan diinisialisasi setelah memuat dataset)
        self.augmentation_visualizer = None
        
        # Inisialisasi handler visualisasi
        from smartcash.ui.dataset.visualization.handlers.visualization_handler import DatasetVisualizationHandler
        self.handler = DatasetVisualizationHandler(self.config)
        
        # Inisialisasi komponen visualisasi
        self.dataset_comparator = dataset_comparator.DatasetComparator(self.handler)
        
    def load_dataset(self, dataset_name: str):
        """Memuat dataset dan statistik terkait"""
        self.current_dataset = dataset_name
        
        # Ambil statistik preprocessing
        self.dataset_stats = get_preprocessing_stats(dataset_name)
        
        # Update komponen statistik
        self.stats_component.update_stats(self.dataset_stats)
        
        # Inisialisasi visualizer augmentasi jika diperlukan
        if 'augmentation' in self.config:
            dataset_path = os.path.join("data", "processed", dataset_name)
            self.augmentation_visualizer = AugmentationVisualizer(
                dataset_path=dataset_path,
                config=self.config['augmentation']
            )
            
    def render(self):
        """Render UI visualisasi"""
        # Tampilkan pilihan dataset
        datasets = list_available_datasets()
        dataset_selector = widgets.Dropdown(
            options=datasets,
            description='Pilih Dataset:',
            disabled=False
        )
        dataset_selector.observe(self._on_dataset_change, names='value')
        
        # Tampilkan komponen
        display(dataset_selector)
        self.stats_component.render()
        
        # Render visualisasi dataset
        self.handler.render()
        
    def _on_dataset_change(self, change):
        """Handler ketika dataset berubah"""
        if change['new']:
            self.load_dataset(change['new'])
            clear_output(wait=True)
            self.render()
    
    def _create_stats_tab(self) -> widgets.Widget:
        """Buat tab statistik dataset"""
        stats_tab = widgets.VBox([
            widgets.HTML("<h3>Statistik Dataset</h3>"),
            self.stats_component.get_ui_components()['main_container']
        ])
        return stats_tab
    
    def _create_advanced_viz_tab(self) -> widgets.Widget:
        """Buat tab visualisasi lanjut"""
        advanced_viz_tab = widgets.VBox([
            widgets.HTML("<h3>Visualisasi Lanjut</h3>"),
            self.advanced_visualizer.get_ui_components()['main_container']
        ])
        return advanced_viz_tab
    
    def _create_main_tabs(self) -> widgets.Widget:
        """Buat tab utama untuk visualisasi"""
        self.tabs = widgets.Tab()
        self.tabs.children = [
            self._create_stats_tab(),
            self._create_advanced_viz_tab(),
            self.dataset_comparator.create_comparison_ui()  # Tab perbandingan
        ]
        self.tabs.set_title(0, 'Statistik')
        self.tabs.set_title(1, 'Visualisasi Lanjut')
        self.tabs.set_title(2, 'Perbandingan')
        
        return self.tabs
    
    def _update_main_tabs(self) -> None:
        """Perbarui konten tab utama"""
        # Perbarui tab statistik
        stats_tab = self._create_stats_tab()
        
        # Perbarui tab visualisasi lanjut
        advanced_viz_tab = self._create_advanced_viz_tab()
        
        # Perbarui tab perbandingan
        comparison_tab = self.dataset_comparator.create_comparison_ui()
        
        # Perbarui tab
        self.tabs.children = [stats_tab, advanced_viz_tab, comparison_tab]
    
    def _create_ui(self) -> widgets.Widget:
        """Buat UI lengkap"""
        # Buat header
        header = widgets.HTML("""
        <div style='background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 15px;'>
            <h1 style='margin: 0;'>Visualisasi Dataset SmartCash</h1>
            <p style='margin: 5px 0 0 0; color: #666;'>
                Visualisasi dataset dan hasil preprocessing/augmentasi
            </p>
        </div>
        """)
        
        # Buat selector dataset
        datasets = list_available_datasets()
        dataset_selector = widgets.Dropdown(
            options=datasets,
            description='Pilih Dataset:',
            disabled=False
        )
        dataset_selector.observe(self._on_dataset_change, names='value')
        
        # Buat tab utama
        main_tabs = self._create_main_tabs()
        
        # Gabungkan semua komponen
        main_container = widgets.VBox([
            header,
            dataset_selector,
            main_tabs
        ])
        
        return main_container
    
    def _on_load_dataset(self, btn) -> None:
        """Handler untuk tombol muat dataset"""
        dataset_name = self.dataset_dropdown.value
        
        with self.status_output:
            clear_output(wait=True)
            print(f"Memuat dataset '{dataset_name}'...")
            
            if self.load_dataset(dataset_name):
                # Perbarui tab utama dengan data baru
                self._update_main_tabs()
                print(f"{ICONS.get('success', '✅')} Dataset berhasil dimuat")
            else:
                print(f"{ICONS.get('error', '❌')} Gagal memuat dataset")
    
    def display(self) -> None:
        """Tampilkan UI visualisasi"""
        if not hasattr(self, 'main_container') or self.main_container is None:
            self.main_container = self._create_ui()
        
        display(self.main_container)
        
    def get_ui_components(self) -> Dict[str, Any]:
        """Dapatkan komponen UI
        
        Returns:
            Dict berisi komponen UI
        """
        if not hasattr(self, 'main_container') or self.main_container is None:
            self.main_container = self._create_ui()
            
        return {
            'main_container': self.main_container,
            'stats_component': self.stats_component,
            'augmentation_visualizer': self.augmentation_visualizer
        }
