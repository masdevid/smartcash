"""
File: smartcash/ui/training/tests/verify_ui_components.py
Deskripsi: Script verifikasi untuk memastikan komponen UI training berfungsi dengan benar
"""

import sys
import os
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import time

def verify_components():
    """
    Verifikasi komponen UI training
    
    Returns:
        Dict berisi hasil verifikasi
    """
    print("🔍 Memulai verifikasi komponen UI training...")
    results = {
        'success': [],
        'failed': []
    }
    
    # Step 1: Verifikasi imports
    print("\n🧩 Verifikasi imports komponen...")
    
    import_tests = [
        ('Config Tabs', 'smartcash.ui.training.components.config_tabs', 'create_config_tabs'),
        ('Metrics Accordion', 'smartcash.ui.training.components.metrics_accordion', 'create_metrics_accordion'),
        ('Control Buttons', 'smartcash.ui.training.components.control_buttons', 'create_training_control_buttons'),
        ('Fallback Component', 'smartcash.ui.training.components.fallback_component', 'create_fallback_component'),
        ('Training Form', 'smartcash.ui.training.components.training_form', 'create_training_form'),
        ('Training Layout', 'smartcash.ui.training.components.training_layout', 'create_training_layout'),
        ('Training Init', 'smartcash.ui.training.training_init', 'TrainingInitializer')
    ]
    
    for name, module_path, component_name in tqdm(import_tests, desc="Import modules"):
        try:
            module = __import__(module_path, fromlist=[component_name])
            component = getattr(module, component_name)
            results['success'].append(f"✅ {name}: {component_name} berhasil diimpor")
        except (ImportError, AttributeError) as e:
            results['failed'].append(f"❌ {name}: Gagal - {str(e)}")
    
    # Step 2: Verifikasi konteks minimum
    print("\n⚙️ Verifikasi konteks minimum untuk membuat komponen...")
    
    # Setup mock config
    test_config = {
        'model': {
            'name': 'yolov5-efficientnet-b4',
            'backbone': 'efficientnet-b4',
            'input_size': 640
        },
        'training': {
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 0.001
        }
    }
    
    context_tests = []
    
    # Step 3: Verifikasi struktur form dan layout
    if all(result.startswith('✅') for result in results['success'] if 'Training Form' in result):
        print("\n📋 Verifikasi struktur form...")
        try:
            from smartcash.ui.training.components.training_form import create_training_form
            form_components = create_training_form(test_config)
            
            expected_keys = [
                'control_buttons', 'progress_container', 'progress_tracker', 
                'status_panel', 'config_tabs', 'info_display'
            ]
            
            missing_keys = [key for key in expected_keys if key not in form_components]
            
            if not missing_keys:
                results['success'].append(f"✅ Training Form: Struktur form sesuai ekspektasi")
            else:
                results['failed'].append(f"❌ Training Form: Missing keys - {', '.join(missing_keys)}")
        except Exception as e:
            results['failed'].append(f"❌ Training Form: Gagal membuat form - {str(e)}")
    
    # Step 4: Verifikasi integrasi dengan config manager
    if all(result.startswith('✅') for result in results['success'] if 'Training Init' in result):
        print("\n🔄 Verifikasi integrasi dengan config manager...")
        try:
            from smartcash.ui.training.training_init import get_training_initializer
            
            # Menggunakan factory function untuk mendapatkan initializer
            initializer = get_training_initializer()
            
            # Tidak perlu menjalankan initialize() karena akan menampilkan UI
            results['success'].append(f"✅ Training Init: Initializer berhasil dibuat melalui factory function")
        except Exception as e:
            results['failed'].append(f"❌ Training Init: Gagal membuat initializer - {str(e)}")
            
        # Coba inisialisasi langsung dengan parameter yang benar
        try:
            from smartcash.ui.training.training_init import TrainingInitializer
            # Inisialisasi dengan parameter yang dibutuhkan
            direct_initializer = TrainingInitializer('training', 'smartcash.ui.training')
            results['success'].append(f"✅ Training Init: Initializer berhasil dibuat secara langsung")
        except Exception as e:
            results['failed'].append(f"❌ Training Init: Gagal membuat initializer secara langsung - {str(e)}")
    
    # Tampilkan hasil
    print("\n" + "="*60)
    print("📊 HASIL VERIFIKASI UI TRAINING")
    print("="*60)
    
    print("\n✅ SUKSES:")
    for result in results['success']:
        print(f"  {result}")
    
    if results['failed']:
        print("\n❌ GAGAL:")
        for result in results['failed']:
            print(f"  {result}")
    
    print("\n" + "="*60)
    print(f"Total komponen diverifikasi: {len(import_tests)}")
    print(f"Berhasil: {len(results['success'])}")
    print(f"Gagal: {len(results['failed'])}")
    print("="*60)
    
    return results


def verify_in_isolation():
    """
    Verifikasi komponen secara terisolasi
    """
    print("🧪 Memulai verifikasi komponen UI training secara terisolasi...")
    results = {
        'success': [],
        'failed': []
    }
    
    # Contoh konfigurasi untuk pengujian
    test_config = {
        'model': {
            'name': 'yolov5-efficientnet-b4',
            'backbone': 'efficientnet-b4'
        },
        'training': {
            'batch_size': 16,
            'epochs': 100
        }
    }
    
    # Verifikasi config_tabs
    print("\n🔍 Verifikasi config_tabs...")
    try:
        from smartcash.ui.training.components.config_tabs import create_config_tabs
        tabs = create_config_tabs(test_config)
        results['success'].append("✅ Config Tabs: Berhasil dibuat")
    except Exception as e:
        results['failed'].append(f"❌ Config Tabs: {str(e)}")
    
    # Verifikasi metrics_accordion
    print("🔍 Verifikasi metrics_accordion...")
    try:
        from smartcash.ui.training.components.metrics_accordion import create_metrics_accordion
        accordion = create_metrics_accordion()
        results['success'].append("✅ Metrics Accordion: Berhasil dibuat")
    except Exception as e:
        results['failed'].append(f"❌ Metrics Accordion: {str(e)}")
    
    # Verifikasi control_buttons
    print("🔍 Verifikasi control_buttons...")
    try:
        from smartcash.ui.training.components.control_buttons import create_training_control_buttons
        buttons = create_training_control_buttons()
        results['success'].append("✅ Control Buttons: Berhasil dibuat")
    except Exception as e:
        results['failed'].append(f"❌ Control Buttons: {str(e)}")
    
    # Verifikasi fallback_component
    print("🔍 Verifikasi fallback_component...")
    try:
        from smartcash.ui.training.components.fallback_component import create_fallback_component
        fallback = create_fallback_component("Test error")
        results['success'].append("✅ Fallback Component: Berhasil dibuat")
    except Exception as e:
        results['failed'].append(f"❌ Fallback Component: {str(e)}")
    
    # Tampilkan hasil
    print("\n" + "="*60)
    print("📊 HASIL VERIFIKASI KOMPONEN TERISOLASI")
    print("="*60)
    
    print("\n✅ SUKSES:")
    for result in results['success']:
        print(f"  {result}")
    
    if results['failed']:
        print("\n❌ GAGAL:")
        for result in results['failed']:
            print(f"  {result}")
    
    print("\n" + "="*60)
    print(f"Total: {len(results['success']) + len(results['failed'])}")
    print(f"Berhasil: {len(results['success'])}")
    print(f"Gagal: {len(results['failed'])}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    print("🧪 Verifikasi Komponen UI Training SmartCash 🧪")
    print("="*60)
    
    # Parse command line arguments
    isolation_mode = "--isolation" in sys.argv
    
    if isolation_mode:
        verify_in_isolation()
    else:
        verify_components()
