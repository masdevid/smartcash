# File: src/interfaces/export_interface.py
# Author: Alfrida Sabar
# Deskripsi: Antarmuka untuk ekspor model ke berbagai format

from pathlib import Path
from .base_interface import BaseInterface
import yaml
class ExportInterface(BaseInterface):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.export_dir = Path('exports')
        self.export_dir.mkdir(exist_ok=True)

    def tampilkan_menu(self):
        menu = """
📦 Menu Ekspor Model:

1. Ekspor ke ONNX
2. Ekspor ke TensorRT
3. Ekspor ke TFLite
4. Ekspor Konfigurasi Model

0. Kembali ke Menu Utama
"""
        return self.prompt(menu, color='cyan')

    def handle_menu(self):
        while True:
            pilihan = self.tampilkan_menu()
            
            try:
                if pilihan == '0':
                    break
                elif pilihan == '1':
                    self.ekspor_ke_onnx()
                elif pilihan == '2':
                    self.ekspor_ke_tensorrt()
                elif pilihan == '3':
                    self.ekspor_ke_tflite()
                elif pilihan == '4':
                    self.ekspor_konfigurasi()
                else:
                    self.show_error("❌ Pilihan menu tidak valid!")
            except Exception as e:
                self.show_error(f"❌ Terjadi kesalahan: {str(e)}")

    def ekspor_ke_onnx(self):
        """Ekspor model ke format ONNX"""
        self.logger.info("📦 Konfigurasi ekspor ONNX...")
        
        # Konfigurasi ekspor
        nama_file = self.prompt("Nama file output", default="model.onnx")
        opset = int(self.prompt("ONNX opset version", default="12"))
        
        if self.confirm("Mulai proses ekspor?"):
            try:
                output_path = self.export_dir / nama_file
                self.detector.export_onnx(
                    output_path=str(output_path),
                    opset=opset
                )
                self.show_success(f"✨ Model berhasil diekspor ke: {output_path}")
            except Exception as e:
                self.show_error(f"❌ Gagal mengekspor model: {str(e)}")

    def ekspor_ke_tensorrt(self):
        """Ekspor model ke format TensorRT"""
        self.logger.info("⚡ Ekspor ke TensorRT akan tersedia segera!")
        self.logger.info("Fitur ini masih dalam pengembangan")

    def ekspor_ke_tflite(self):
        """Ekspor model ke format TFLite"""
        self.logger.info("📱 Ekspor ke TFLite akan tersedia segera!")
        self.logger.info("Fitur ini masih dalam pengembangan")

    def ekspor_konfigurasi(self):
        """Ekspor konfigurasi model"""
        self.logger.info("⚙️ Mengekspor konfigurasi model...")
        
        try:
            # Dapatkan konfigurasi
            config = {
                'model': {
                    'type': self.detector.model_type,
                    'img_size': self.cfg.model.img_size,
                    'nc': self.cfg.model.nc,
                    'backbone': self.detector.backbone_info
                },
                'inference': {
                    'conf_thres': self.cfg.model.conf_thres,
                    'iou_thres': 0.45,
                    'max_det': 300
                },
                'classes': self.cfg.data.class_names
            }
            
            # Simpan konfigurasi
            output_path = self.export_dir / 'model_config.yaml'
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            self.show_success(f"✨ Konfigurasi berhasil diekspor ke: {output_path}")
        except Exception as e:
            self.show_error(f"❌ Gagal mengekspor konfigurasi: {str(e)}")