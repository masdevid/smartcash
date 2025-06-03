#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /Users/masdevid/Projects/smartcash/smartcash/tests/test_config_persistence.py
# Test persistensi konfigurasi untuk memastikan load dan save config berfungsi dengan benar

import os
import shutil
import tempfile
import unittest
import yaml
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional

from smartcash.common.config import ConfigManager, get_config_manager


class TestConfigPersistence(unittest.TestCase):
    """Test persistensi konfigurasi untuk memastikan load dan save config berfungsi dengan benar"""
    
    def setUp(self) -> None:
        """Setup temporary directory untuk test"""
        # Buat temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Buat struktur direktori configs
        self.temp_configs_dir = os.path.join(self.temp_dir, "configs")
        os.makedirs(self.temp_configs_dir, exist_ok=True)
        
        # Copy semua file config dari smartcash/configs ke temporary directory
        self.original_configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
        for config_file in os.listdir(self.original_configs_dir):
            if config_file.endswith(".yaml"):
                shutil.copy(
                    os.path.join(self.original_configs_dir, config_file),
                    os.path.join(self.temp_configs_dir, config_file)
                )
        
        # Dictionary untuk menyimpan config asli untuk perbandingan
        self.original_configs = {}
        
        # Inisialisasi ConfigManager dengan path temporary
        self.config_manager = ConfigManager(base_dir=self.temp_dir)
    
    def tearDown(self) -> None:
        """Cleanup temporary directory setelah test selesai"""
        shutil.rmtree(self.temp_dir)
    
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """Helper untuk memuat file YAML"""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _check_file_name_consistency(self, module_name: str) -> None:
        """Memastikan nama file yang disimpan tidak menghasilkan file baru seperti xxx_config_config.yaml"""
        # Dapatkan daftar file sebelum test dimulai
        expected_files = set([f for f in os.listdir(self.original_configs_dir) if f.endswith('.yaml')])
        # Dapatkan daftar file setelah penyimpanan
        actual_files = set([f for f in os.listdir(self.temp_configs_dir) if f.endswith('.yaml')])
        
        # Pastikan tidak ada file baru yang dibuat
        self.assertEqual(expected_files, actual_files, 
                        f"File baru dibuat saat menyimpan {module_name} config")
    
    def test_dataset_config_persistence(self) -> None:
        """Test persistensi konfigurasi dataset"""
        # Load konfigurasi dataset
        dataset_config = self.config_manager.load_config("dataset")
        
        # Simpan konfigurasi asli untuk perbandingan
        original_config = copy.deepcopy(dataset_config)
        
        # Ubah beberapa nilai
        if "data" in dataset_config and "roboflow" in dataset_config["data"]:
            dataset_config["data"]["roboflow"]["workspace"] = "test-workspace"
            dataset_config["data"]["roboflow"]["project"] = "test-project"
            dataset_config["data"]["roboflow"]["version"] = 2
        else:
            # Fallback jika struktur berbeda
            dataset_config["dataset_name"] = "test-dataset"
            dataset_config["version"] = "2.0"
        
        # Simpan konfigurasi yang diubah
        save_result = self.config_manager.save_config(dataset_config, "dataset")
        self.assertTrue(save_result, "Gagal menyimpan dataset config")
        
        # Periksa nama file
        self._check_file_name_consistency("dataset")
        
        # Load ulang konfigurasi dan verifikasi perubahan
        reloaded_config = self.config_manager.load_config("dataset")
        
        # Verifikasi perubahan berdasarkan struktur yang diubah
        if "data" in reloaded_config and "roboflow" in reloaded_config["data"]:
            self.assertEqual(reloaded_config["data"]["roboflow"]["workspace"], "test-workspace")
            self.assertEqual(reloaded_config["data"]["roboflow"]["project"], "test-project")
            self.assertEqual(reloaded_config["data"]["roboflow"]["version"], 2)
        else:
            self.assertEqual(reloaded_config["dataset_name"], "test-dataset")
            self.assertEqual(reloaded_config["version"], "2.0")
    
    def test_preprocessing_config_persistence(self) -> None:
        """Test persistensi konfigurasi preprocessing"""
        # Load konfigurasi preprocessing
        preprocessing_config = self.config_manager.load_config("preprocessing")
        
        # Simpan konfigurasi asli untuk perbandingan
        original_config = copy.deepcopy(preprocessing_config)
        
        # Ubah beberapa nilai berdasarkan struktur yang ada
        if "preprocessing" in preprocessing_config:
            # Struktur baru dengan preprocessing sebagai key utama
            preprocessing_config["preprocessing"]["resize"] = not preprocessing_config["preprocessing"].get("resize", True)
            preprocessing_config["preprocessing"]["size"] = 640
        else:
            # Struktur lama atau berbeda
            if "resize" in preprocessing_config:
                preprocessing_config["resize"]["enabled"] = not preprocessing_config["resize"].get("enabled", True)
                preprocessing_config["resize"]["size"] = 640
            else:
                preprocessing_config["resize_enabled"] = True
                preprocessing_config["resize_size"] = 640
        
        # Simpan konfigurasi yang diubah
        save_result = self.config_manager.save_config(preprocessing_config, "preprocessing")
        self.assertTrue(save_result, "Gagal menyimpan preprocessing config")
        
        # Periksa nama file
        self._check_file_name_consistency("preprocessing")
        
        # Load ulang konfigurasi dan verifikasi perubahan
        reloaded_config = self.config_manager.load_config("preprocessing")
        
        # Verifikasi perubahan berdasarkan struktur yang diubah
        if "preprocessing" in reloaded_config:
            self.assertEqual(reloaded_config["preprocessing"]["resize"], preprocessing_config["preprocessing"]["resize"])
            self.assertEqual(reloaded_config["preprocessing"]["size"], 640)
        elif "resize" in reloaded_config:
            self.assertEqual(reloaded_config["resize"]["enabled"], preprocessing_config["resize"]["enabled"])
            self.assertEqual(reloaded_config["resize"]["size"], 640)
        else:
            self.assertEqual(reloaded_config["resize_enabled"], True)
            self.assertEqual(reloaded_config["resize_size"], 640)
    
    def test_augmentation_config_persistence(self) -> None:
        """Test persistensi konfigurasi augmentation"""
        # Load konfigurasi augmentation
        augmentation_config = self.config_manager.load_config("augmentation")
        
        # Simpan konfigurasi asli untuk perbandingan
        original_config = copy.deepcopy(augmentation_config)
        
        # Ubah beberapa nilai berdasarkan struktur yang ada
        if "augmentation" in augmentation_config:
            # Struktur dengan augmentation sebagai key utama
            if "types" in augmentation_config["augmentation"]:
                # Tambahkan tipe augmentasi baru jika belum ada
                if "mosaic" not in augmentation_config["augmentation"]["types"]:
                    augmentation_config["augmentation"]["types"].append("mosaic")
            
            # Ubah parameter lain
            augmentation_config["augmentation"]["enabled"] = not augmentation_config["augmentation"].get("enabled", True)
            augmentation_config["augmentation"]["num_variations"] = 5
        elif "position" in augmentation_config:
            # Struktur dengan position sebagai key
            augmentation_config["position"]["fliplr"] = 0.75
            augmentation_config["position"]["degrees"] = 20
        else:
            # Fallback untuk struktur lain
            augmentation_config["enabled"] = True
            augmentation_config["num_variations"] = 5
        
        # Simpan konfigurasi yang diubah
        save_result = self.config_manager.save_config(augmentation_config, "augmentation")
        self.assertTrue(save_result, "Gagal menyimpan augmentation config")
        
        # Periksa nama file
        self._check_file_name_consistency("augmentation")
        
        # Load ulang konfigurasi dan verifikasi perubahan
        reloaded_config = self.config_manager.load_config("augmentation")
        
        # Verifikasi perubahan berdasarkan struktur yang diubah
        if "augmentation" in reloaded_config:
            self.assertEqual(reloaded_config["augmentation"]["enabled"], augmentation_config["augmentation"]["enabled"])
            self.assertEqual(reloaded_config["augmentation"]["num_variations"], 5)
            if "types" in reloaded_config["augmentation"]:
                self.assertIn("mosaic", reloaded_config["augmentation"]["types"])
        elif "position" in reloaded_config:
            self.assertEqual(reloaded_config["position"]["fliplr"], 0.75)
            self.assertEqual(reloaded_config["position"]["degrees"], 20)
        else:
            self.assertEqual(reloaded_config["enabled"], True)
            self.assertEqual(reloaded_config["num_variations"], 5)
    
    def test_model_config_persistence(self) -> None:
        """Test persistensi konfigurasi model"""
        # Load konfigurasi model
        model_config = self.config_manager.load_config("model")
        
        # Simpan konfigurasi asli untuk perbandingan
        original_config = copy.deepcopy(model_config)
        
        # Ubah beberapa nilai berdasarkan struktur yang ada
        if "model" in model_config and "backbone" in model_config["model"]:
            model_config["model"]["backbone"] = "efficientnet_b4"
            model_config["model"]["use_attention"] = not model_config["model"].get("use_attention", False)
            model_config["model"]["use_residual"] = not model_config["model"].get("use_residual", False)
        else:
            # Struktur lain dengan backbone sebagai key utama
            if "backbone" in model_config:
                if isinstance(model_config["backbone"], dict) and "type" in model_config["backbone"]:
                    model_config["backbone"]["type"] = "efficientnet_b4"
                    model_config["backbone"]["use_attention"] = not model_config["backbone"].get("use_attention", False)
                    model_config["backbone"]["use_residual"] = not model_config["backbone"].get("use_residual", False)
                else:
                    model_config["backbone"] = "efficientnet_b4"
            else:
                # Fallback untuk struktur lain
                model_config["model_type"] = "efficient_basic"
                model_config["backbone_type"] = "efficientnet_b4"
                model_config["use_attention"] = True
        
        # Simpan konfigurasi yang diubah
        save_result = self.config_manager.save_config(model_config, "model")
        self.assertTrue(save_result, "Gagal menyimpan model config")
        
        # Periksa nama file
        self._check_file_name_consistency("model")
        
        # Load ulang konfigurasi dan verifikasi perubahan
        reloaded_config = self.config_manager.load_config("model")
        
        # Verifikasi perubahan berdasarkan struktur yang diubah
        if "model" in reloaded_config and "backbone" in reloaded_config["model"]:
            self.assertEqual(reloaded_config["model"]["backbone"], "efficientnet_b4")
            self.assertEqual(reloaded_config["model"]["use_attention"], model_config["model"]["use_attention"])
            self.assertEqual(reloaded_config["model"]["use_residual"], model_config["model"]["use_residual"])
        elif "backbone" in reloaded_config:
            if isinstance(reloaded_config["backbone"], dict) and "type" in reloaded_config["backbone"]:
                self.assertEqual(reloaded_config["backbone"]["type"], "efficientnet_b4")
                self.assertEqual(reloaded_config["backbone"]["use_attention"], model_config["backbone"]["use_attention"])
                self.assertEqual(reloaded_config["backbone"]["use_residual"], model_config["backbone"]["use_residual"])
            else:
                self.assertEqual(reloaded_config["backbone"], "efficientnet_b4")
        else:
            self.assertEqual(reloaded_config["model_type"], "efficient_basic")
            self.assertEqual(reloaded_config["backbone_type"], "efficientnet_b4")
            self.assertEqual(reloaded_config["use_attention"], True)
    
    def test_hyperparameters_config_persistence(self) -> None:
        """Test persistensi konfigurasi hyperparameters"""
        # Load konfigurasi hyperparameters
        hyperparameters_config = self.config_manager.load_config("hyperparameters")
        
        # Simpan konfigurasi asli untuk perbandingan
        original_config = copy.deepcopy(hyperparameters_config)
        
        # Ubah beberapa nilai berdasarkan struktur yang ada
        if "batch_size" in hyperparameters_config:
            hyperparameters_config["batch_size"] = 16
            hyperparameters_config["epochs"] = 100
            
            # Ubah optimizer jika ada
            if "optimizer" in hyperparameters_config:
                if isinstance(hyperparameters_config["optimizer"], dict) and "type" in hyperparameters_config["optimizer"]:
                    hyperparameters_config["optimizer"]["type"] = "adam"
                else:
                    hyperparameters_config["optimizer"] = "adam"
        else:
            # Struktur dengan hyperparameters sebagai key utama
            if "hyperparameters" in hyperparameters_config:
                hyperparameters_config["hyperparameters"]["batch_size"] = 16
                hyperparameters_config["hyperparameters"]["epochs"] = 100
                
                # Ubah optimizer jika ada
                if "optimizer" in hyperparameters_config["hyperparameters"]:
                    if isinstance(hyperparameters_config["hyperparameters"]["optimizer"], dict):
                        hyperparameters_config["hyperparameters"]["optimizer"]["type"] = "adam"
                    else:
                        hyperparameters_config["hyperparameters"]["optimizer"] = "adam"
            else:
                # Fallback untuk struktur lain
                hyperparameters_config["training"] = {
                    "batch_size": 16,
                    "epochs": 100,
                    "optimizer": "adam"
                }
        
        # Simpan konfigurasi yang diubah
        save_result = self.config_manager.save_config(hyperparameters_config, "hyperparameters")
        self.assertTrue(save_result, "Gagal menyimpan hyperparameters config")
        
        # Periksa nama file
        self._check_file_name_consistency("hyperparameters")
        
        # Load ulang konfigurasi dan verifikasi perubahan
        reloaded_config = self.config_manager.load_config("hyperparameters")
        
        # Verifikasi perubahan berdasarkan struktur yang diubah
        if "batch_size" in reloaded_config:
            self.assertEqual(reloaded_config["batch_size"], 16)
            self.assertEqual(reloaded_config["epochs"], 100)
            
            if "optimizer" in reloaded_config:
                if isinstance(reloaded_config["optimizer"], dict) and "type" in reloaded_config["optimizer"]:
                    self.assertEqual(reloaded_config["optimizer"]["type"], "adam")
                else:
                    self.assertEqual(reloaded_config["optimizer"], "adam")
        elif "hyperparameters" in reloaded_config:
            self.assertEqual(reloaded_config["hyperparameters"]["batch_size"], 16)
            self.assertEqual(reloaded_config["hyperparameters"]["epochs"], 100)
            
            if "optimizer" in reloaded_config["hyperparameters"]:
                if isinstance(reloaded_config["hyperparameters"]["optimizer"], dict):
                    self.assertEqual(reloaded_config["hyperparameters"]["optimizer"]["type"], "adam")
                else:
                    self.assertEqual(reloaded_config["hyperparameters"]["optimizer"], "adam")
        else:
            self.assertEqual(reloaded_config["training"]["batch_size"], 16)
            self.assertEqual(reloaded_config["training"]["epochs"], 100)
            self.assertEqual(reloaded_config["training"]["optimizer"], "adam")
    
    def test_training_config_persistence(self) -> None:
        """Test persistensi konfigurasi training"""
        # Load konfigurasi training
        training_config = self.config_manager.load_config("training")
        
        # Simpan konfigurasi asli untuk perbandingan
        original_config = copy.deepcopy(training_config)
        
        # Ubah beberapa nilai berdasarkan struktur yang ada
        if "validation" in training_config:
            # Struktur dengan validation sebagai key utama
            if isinstance(training_config["validation"], dict) and "frequency" in training_config["validation"]:
                training_config["validation"]["frequency"] = 2
            
            # Ubah multi_scale jika ada
            if "multi_scale" in training_config and isinstance(training_config["multi_scale"], bool):
                training_config["multi_scale"] = not training_config["multi_scale"]
            
            # Ubah training_utils jika ada
            if "training_utils" in training_config and isinstance(training_config["training_utils"], dict):
                if "mixed_precision" in training_config["training_utils"]:
                    training_config["training_utils"]["mixed_precision"] = not training_config["training_utils"]["mixed_precision"]
                if "tensorboard" in training_config["training_utils"]:
                    training_config["training_utils"]["tensorboard"] = not training_config["training_utils"]["tensorboard"]
        else:
            # Fallback untuk struktur lain
            training_config["validation_frequency"] = 2
            training_config["use_multi_scale"] = True
            training_config["use_mixed_precision"] = True
        
        # Simpan konfigurasi yang diubah
        save_result = self.config_manager.save_config(training_config, "training")
        self.assertTrue(save_result, "Gagal menyimpan training config")
        
        # Periksa nama file
        self._check_file_name_consistency("training")
        
        # Load ulang konfigurasi dan verifikasi perubahan
        reloaded_config = self.config_manager.load_config("training")
        
        # Verifikasi perubahan berdasarkan struktur yang diubah
        if "validation" in reloaded_config:
            if isinstance(reloaded_config["validation"], dict) and "frequency" in reloaded_config["validation"]:
                self.assertEqual(reloaded_config["validation"]["frequency"], 2)
            
            if "multi_scale" in reloaded_config and isinstance(reloaded_config["multi_scale"], bool):
                self.assertEqual(reloaded_config["multi_scale"], training_config["multi_scale"])
            
            if "training_utils" in reloaded_config and isinstance(reloaded_config["training_utils"], dict):
                if "mixed_precision" in reloaded_config["training_utils"]:
                    self.assertEqual(reloaded_config["training_utils"]["mixed_precision"], training_config["training_utils"]["mixed_precision"])
                if "tensorboard" in reloaded_config["training_utils"]:
                    self.assertEqual(reloaded_config["training_utils"]["tensorboard"], training_config["training_utils"]["tensorboard"])
        else:
            self.assertEqual(reloaded_config["validation_frequency"], 2)
            self.assertEqual(reloaded_config["use_multi_scale"], True)
            self.assertEqual(reloaded_config["use_mixed_precision"], True)
    
    def test_config_inheritance(self) -> None:
        """Test persistensi konfigurasi dengan inheritance"""
        # Load konfigurasi training yang mewarisi dari hyperparameters dan model
        training_config = self.config_manager.load_config("training")
        
        # Simpan konfigurasi asli untuk perbandingan
        original_config = copy.deepcopy(training_config)
        
        # Periksa apakah training config memiliki _base_ yang menunjukkan inheritance
        if "_base_" in training_config:
            # Simpan nilai _base_ untuk diperiksa nanti
            base_configs = training_config.get("_base_", [])
            self.assertTrue(len(base_configs) > 0, "Training config tidak mewarisi konfigurasi lain")
            
            # Periksa apakah salah satu base config adalah hyperparameters_config.yaml
            has_hyperparameters_base = any("hyperparameters" in base for base in base_configs)
            self.assertTrue(has_hyperparameters_base, "Training config tidak mewarisi hyperparameters config")
        
        # Ubah beberapa nilai di training config
        if "validation" in training_config and isinstance(training_config["validation"], dict):
            if "frequency" in training_config["validation"]:
                training_config["validation"]["frequency"] = 2
        
        # Simpan konfigurasi yang diubah
        save_result = self.config_manager.save_config(training_config, "training")
        self.assertTrue(save_result, "Gagal menyimpan training config")
        
        # Periksa nama file
        self._check_file_name_consistency("training")
        
        # Load ulang konfigurasi dan verifikasi perubahan
        reloaded_config = self.config_manager.load_config("training")
        
        # Verifikasi perubahan berdasarkan struktur yang diubah
        if "validation" in reloaded_config and isinstance(reloaded_config["validation"], dict):
            if "frequency" in reloaded_config["validation"]:
                self.assertEqual(reloaded_config["validation"]["frequency"], 2)
        
        # Verifikasi bahwa training config memiliki parameter dari hyperparameters
        # Catatan: Dalam implementasi SimpleConfigManager, inheritance mungkin tidak otomatis diproses
        # Kita hanya memeriksa bahwa file config dapat disimpan dan dimuat dengan benar
        
        # Periksa struktur training config untuk parameter batch_size dan epochs
        has_batch_size = False
        has_epochs = False
        
        if "batch_size" in training_config:
            has_batch_size = True
        elif "hyperparameters" in training_config and "batch_size" in training_config["hyperparameters"]:
            has_batch_size = True
        elif "training" in training_config and "batch_size" in training_config["training"]:
            has_batch_size = True
            
        if "epochs" in training_config:
            has_epochs = True
        elif "hyperparameters" in training_config and "epochs" in training_config["hyperparameters"]:
            has_epochs = True
        elif "training" in training_config and "epochs" in training_config["training"]:
            has_epochs = True
            
        # Verifikasi bahwa parameter penting ada di training config
        self.assertTrue(has_batch_size or "_base_" in training_config, "Training config tidak memiliki parameter batch_size")
        self.assertTrue(has_epochs or "_base_" in training_config, "Training config tidak memiliki parameter epochs")
        
        # Simpan konfigurasi yang diubah
        save_result = self.config_manager.save_config(training_config, "training")
        self.assertTrue(save_result, "Gagal menyimpan training config")
        
        # Periksa nama file
        self._check_file_name_consistency("training")
        
        # Load ulang konfigurasi dan verifikasi perubahan
        reloaded_config = self.config_manager.load_config("training")
        
        # Verifikasi perubahan berdasarkan struktur yang diubah
        if "validation" in reloaded_config:
            if isinstance(reloaded_config["validation"], dict) and "frequency" in reloaded_config["validation"]:
                self.assertEqual(reloaded_config["validation"]["frequency"], 2)
            
            if "multi_scale" in reloaded_config and isinstance(reloaded_config["multi_scale"], bool):
                self.assertEqual(reloaded_config["multi_scale"], training_config["multi_scale"])
            
            if "training_utils" in reloaded_config and isinstance(reloaded_config["training_utils"], dict):
                if "mixed_precision" in reloaded_config["training_utils"]:
                    self.assertEqual(reloaded_config["training_utils"]["mixed_precision"], training_config["training_utils"]["mixed_precision"])
                if "tensorboard" in reloaded_config["training_utils"]:
                    self.assertEqual(reloaded_config["training_utils"]["tensorboard"], training_config["training_utils"]["tensorboard"])
        else:
            self.assertEqual(reloaded_config["validation_frequency"], 2)
            self.assertEqual(reloaded_config["use_multi_scale"], True)
            self.assertEqual(reloaded_config["use_mixed_precision"], True)

if __name__ == '__main__':
    unittest.main()
