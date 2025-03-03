# File: smartcash/interface/menu/__init__.py
# Author: Alfrida Sabar
# Deskripsi: Menu utama pelatihan model dengan integrasi submenu

from smartcash.interface.menu.base import BaseMenu, MenuItem
from smartcash.interface.menu.training.detection_mode import DetectionModeMenu
from smartcash.interface.menu.training.backbone import BackboneMenu
from smartcash.interface.menu.training.parameters import TrainingParamsMenu
from smartcash.handlers.training_pipeline import TrainingPipeline
from pathlib import Path


class TrainingMenu(BaseMenu):
    """Menu utama pelatihan model."""
    
    def __init__(self, app, config_manager, display):
        self.app = app
        self.config_manager = config_manager
        self.display = display
        
        # Setup submenus
        self.detection_menu = DetectionModeMenu(app, config_manager, display)
        self.backbone_menu = BackboneMenu(app, config_manager, display)
        self.params_menu = TrainingParamsMenu(app, config_manager, display)
        
        # Create menu items
        items = [
            MenuItem(
                title="Pilih Sumber Data",
                action=self.show_data_source_menu,
                description="Pilih sumber dataset untuk pelatihan",
                category="Konfigurasi"
            ),
            MenuItem(
                title="Pilih Mode Deteksi",
                action=self.show_detection_mode_menu,
                description="Pilih mode deteksi lapis tunggal atau banyak",
                category="Konfigurasi"
            ),
            MenuItem(
                title="Pilih Arsitektur Model",
                action=self.show_backbone_menu,
                description="Pilih arsitektur backbone model",
                category="Konfigurasi"
            ),
            MenuItem(
                title="Konfigurasi Parameter",
                action=self.show_training_params_menu,
                description="Atur parameter pelatihan",
                category="Konfigurasi"
            ),
            MenuItem(
                title="Mulai Pelatihan",
                action=self.start_training,
                description="Mulai proses pelatihan model",
                category="Aksi",
                enabled=False  # Will be enabled when config is complete
            ),
            MenuItem(
                title="Kembali",
                action=lambda: False,
                category="Navigasi"
            )
        ]
        
        super().__init__("Menu Pelatihan Model", items)
        self._update_menu_state()
        
    def _update_menu_state(self) -> None:
        """Update status menu berdasarkan konfigurasi saat ini."""
        config = self.config_manager.current_config
        
        # Enable/disable start training based on config completeness
        can_train = all([
            config.get('data_source'),
            config.get('detection_mode'),
            config.get('backbone')
        ])
        
        if can_train:
            self.enable_item("Mulai Pelatihan")
        else:
            self.disable_item("Mulai Pelatihan")
            
        # Update submenus state jika perlu
        self.backbone_menu = BackboneMenu(self.app, self.config_manager, self.display)
        
    def show_data_source_menu(self) -> bool:
        """Tampilkan menu pemilihan sumber data."""
        items = [
            MenuItem(
                title="Dataset Lokal",
                action=lambda: self._set_data_source('local'),
                description=(
                    "Gunakan dataset yang tersimpan lokal"
                ),
                category="Sumber Data"
            ),
            MenuItem(
                title="Dataset Roboflow",
                action=lambda: self._set_data_source('roboflow'),
                description=(
                    "Unduh dan gunakan dataset dari Roboflow"
                ),
                category="Sumber Data"
            ),
            MenuItem(
                title="Kembali",
                action=lambda: False,
                category="Navigasi"
            )
        ]
        
        submenu = BaseMenu("Pilih Sumber Data", items, parent=self)
        
        # Show submenu
        while True:
            self.app.stdscr.clear()
            submenu.draw(self.app.stdscr, 2)
            self.display.show_config_status(self.config_manager.current_config)
            self.app.stdscr.refresh()
            
            key = self.app.stdscr.getch()
            if key == ord('q'):
                return True
                
            result = submenu.handle_input(key)
            if result is False:
                return True
    
    def _set_data_source(self, source: str) -> bool:
        """Set sumber data dan update menu."""
        try:
            self.config_manager.update('data_source', source)
            
            # Additional config for Roboflow
            if source == 'roboflow':
                roboflow_config = {
                    'workspace': 'detection-twl6q',
                    'project': 'rupiah_emisi-baru',
                    'version': '3'
                }
                self.config_manager.update('roboflow', roboflow_config)
            
            self.config_manager.save()
            self._update_menu_state()
            
            self.display.show_success(f"Sumber data diatur ke: {source}")
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal mengatur sumber data: {str(e)}")
            return True
    
    def show_detection_mode_menu(self) -> bool:
        """Tampilkan menu mode deteksi."""
        # Show detection mode submenu
        while True:
            self.app.stdscr.clear()
            self.detection_menu.draw(self.app.stdscr, 2)
            self.display.show_config_status(self.config_manager.current_config)
            self.app.stdscr.refresh()
            
            key = self.app.stdscr.getch()
            if key == ord('q'):
                return True
                
            result = self.detection_menu.handle_input(key)
            if result is False:
                self._update_menu_state()  # Update setelah kembali
                return True
    
    def show_backbone_menu(self) -> bool:
        """Tampilkan menu pemilihan backbone."""
        while True:
            self.app.stdscr.clear()
            self.backbone_menu.draw(self.app.stdscr, 2)
            self.display.show_config_status(self.config_manager.current_config)
            self.app.stdscr.refresh()
            
            key = self.app.stdscr.getch()
            if key == ord('q'):
                return True
                
            result = self.backbone_menu.handle_input(key)
            if result is False:
                self._update_menu_state()
                return True
    
    def show_training_params_menu(self) -> bool:
        """Tampilkan menu parameter pelatihan."""
        while True:
            self.app.stdscr.clear()
            self.params_menu.draw(self.app.stdscr, 2)
            self.display.show_config_status(self.config_manager.current_config)
            self.app.stdscr.refresh()
            
            key = self.app.stdscr.getch()
            if key == ord('q'):
                return True
                
            result = self.params_menu.handle_input(key)
            if result is False:
                return True
    
    def start_training(self) -> bool:
        """Mulai proses pelatihan model."""
        try:
            # Validasi konfigurasi
            if not self._validate_training_config():
                return True

            # Konfirmasi mulai training
            # if not self.display.show_dialog(
            #     "Konfirmasi",
            #     "Apakah Anda yakin ingin memulai pelatihan?",
            #     {"y": "Ya", "n": "Tidak"}
            # ):
            #     return True
                
            # Setup training pipeline dengan path yang benar
            config = self.config_manager.current_config
            
            # Pastikan path dataset dan output ada
            data_dir = Path('data')
            if not data_dir.exists():
                self.display.show_error("❌ Direktori data tidak ditemukan")
                return True
                
            # Set path dataset berdasarkan sumber data
            if config['data_source'] == 'local':
                config.update({
                    'train_data_path': str(data_dir / 'train'),
                    'val_data_path': str(data_dir / 'valid'),
                    'test_data_path': str(data_dir / 'test')
                })
            elif config['data_source'] == 'roboflow':
                # Akan dihandle oleh RoboflowHandler
                pass
            
            # Set path output
            output_dir = Path('runs/train')
            output_dir.mkdir(parents=True, exist_ok=True)
            config['output_dir'] = str(output_dir)
            
            # Clear screen dan tampilkan status training
            self.app.stdscr.clear()
            self.display.show_success("🚀 Memulai pelatihan model...")
            self.app.stdscr.refresh()
            
            # Run training dengan config yang sudah divalidasi
            pipeline = TrainingPipeline(config=config)
            results = pipeline.train(display_manager=self.display)
            if results == {}:
                return True
            
            # Show completion dialog
            self.display.show_dialog(
                "Pelatihan Selesai",
                f"Model tersimpan di: {results['train_dir']}\n\n"
                f"Metrik terbaik:\n"
                f"• Akurasi: {results['best_metrics']['accuracy']:.4f}\n"
                f"• F1-Score: {results['best_metrics']['f1']:.4f}\n"
                f"• mAP: {results['best_metrics']['mAP']:.4f}",
                {"o": "OK"}
            )
            
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal memulai pelatihan: {str(e)}")
            return False

    def _validate_training_config(self) -> bool:
        """Validasi konfigurasi sebelum training."""
        config = self.config_manager.current_config
        
        # Required configs
        required = {
            'data_source': 'Sumber data',
            'detection_mode': 'Mode deteksi',
            'backbone': 'Arsitektur backbone',
            'training.batch_size': 'Ukuran batch',
            'training.learning_rate': 'Learning rate',
            'training.epochs': 'Jumlah epoch'
        }
        
        # Check each required config
        for key, label in required.items():
            value = config
            for k in key.split('.'):
                value = value.get(k)
                if value is None:
                    self.display.show_error(f"❌ {label} belum dikonfigurasi")
                    return False
        
        return True