#!/usr/bin/env python3

# File: scripts/check_classes.py
# Author: Alfrida Sabar
# Deskripsi: Script untuk memeriksa konsistensi kelas antara konfigurasi dan implementasi

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Set
import argparse
import importlib.util
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

def load_config(config_path: str) -> Dict:
    """Load konfigurasi dari file YAML."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"{Fore.RED}❌ Gagal memuat konfigurasi: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

def extract_classes_from_config(config: Dict) -> Dict[str, Set[str]]:
    """Ekstrak nama kelas dari berbagai lokasi di konfigurasi."""
    class_locations = {
        'dataset.classes': set(),
        'data.classes': set()
    }
    
    # Check dataset.classes
    if 'dataset' in config and 'classes' in config['dataset']:
        class_locations['dataset.classes'] = set(config['dataset']['classes'])
    
    # Check data.classes
    if 'data' in config and 'classes' in config['data']:
        class_locations['data.classes'] = set(config['data']['classes'])
    
    return class_locations

def extract_classes_from_code(root_dir: str) -> Dict[str, Set[str]]:
    """Ekstrak nama kelas dari file kode penting."""
    result = {}
    
    # Paths untuk file-file penting
    files_to_check = {
        'detection_handler': os.path.join(root_dir, 'smartcash/handlers/detection_handler.py'),
        'data_handler': os.path.join(root_dir, 'smartcash/handlers/data_handler.py'),
        'config_manager': os.path.join(root_dir, 'smartcash/cli/configuration_manager.py')
    }
    
    # Periksa setiap file
    for name, path in files_to_check.items():
        if not os.path.exists(path):
            result[name] = set()
            continue
            
        # Import module dynamically 
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Ekstrak kelas berdasarkan file
            if name == 'detection_handler':
                # Extract from CURRENCY_CLASSES
                if hasattr(module, 'DetectionHandler'):
                    result[name] = set([info['name'] for info in module.DetectionHandler.CURRENCY_CLASSES.values()])
            elif name == 'data_handler':
                # Extract from MultilayerDataset.layer_config
                if hasattr(module, 'MultilayerDataset'):
                    all_classes = []
                    for layer_info in module.MultilayerDataset.layer_config.values():
                        all_classes.extend(layer_info['classes'])
                    result[name] = set(all_classes)
            elif name == 'config_manager':
                # Extract from DEFAULT_CONFIG
                if hasattr(module, 'ConfigurationManager'):
                    default_classes = module.ConfigurationManager.DEFAULT_CONFIG.get('dataset', {}).get('classes', [])
                    result[name] = set(default_classes)
                    
                    # Also check find_class_names method return value
                    if hasattr(module.ConfigurationManager, 'find_class_names'):
                        # Create temporary instance with minimal config
                        temp_config_path = Path(root_dir) / 'configs/base_config.yaml'
                        if temp_config_path.exists():
                            try:
                                cm = module.ConfigurationManager(str(temp_config_path))
                                # Call with detection_mode single and multi
                                cm.update('detection_mode', 'single')
                                single_classes = cm.find_class_names()
                                cm.update('detection_mode', 'multi')
                                multi_classes = cm.find_class_names()
                                result[f"{name}_single"] = set(single_classes)
                                result[f"{name}_multi"] = set(multi_classes)
                            except:
                                pass
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Gagal mengekstrak dari {name}: {str(e)}{Style.RESET_ALL}")
            result[name] = set()
    
    return result

def analyze_consistency(config_classes: Dict[str, Set[str]], code_classes: Dict[str, Set[str]]) -> None:
    """Analisis konsistensi antara konfigurasi dan kode."""
    print(f"\n{Fore.CYAN}📊 Analisis Konsistensi Kelas{Style.RESET_ALL}\n")
    
    # Combine all classes from config
    all_config_classes = set()
    for classes in config_classes.values():
        all_config_classes.update(classes)
        
    # List semua lokasi dengan kelas
    all_locations = list(config_classes.keys()) + list(code_classes.keys())
    
    # Create summary table
    print(f"{Fore.CYAN}📋 Ringkasan Jumlah Kelas:{Style.RESET_ALL}")
    print("-" * 50)
    print(f"{'Lokasi':<25} | {'Jumlah Kelas':<10} | {'Status':<10}")
    print("-" * 50)
    
    for loc in all_locations:
        if loc in config_classes:
            count = len(config_classes[loc])
            if count == 7:
                status = f"{Fore.GREEN}OK (Banknote){Style.RESET_ALL}"
            elif count == 17:
                status = f"{Fore.GREEN}OK (Multi){Style.RESET_ALL}"
            elif count == 0:
                status = f"{Fore.RED}Kosong{Style.RESET_ALL}"
            else:
                status = f"{Fore.YELLOW}Perlu dicek{Style.RESET_ALL}"
        else:
            count = len(code_classes[loc])
            if count == 7:
                status = f"{Fore.GREEN}OK (Banknote){Style.RESET_ALL}"
            elif count == 17:
                status = f"{Fore.GREEN}OK (Multi){Style.RESET_ALL}"
            elif count == 0:
                status = f"{Fore.RED}Kosong{Style.RESET_ALL}"
            else:
                status = f"{Fore.YELLOW}Perlu dicek{Style.RESET_ALL}"
        
        print(f"{loc:<25} | {count:<10} | {status}")
    print("-" * 50)
    
    # Check for mismatches between locations
    print(f"\n{Fore.CYAN}🔍 Perbandingan Antar Lokasi:{Style.RESET_ALL}")
    
    # Compare all pairs
    all_locations_with_classes = []
    for loc in all_locations:
        if loc in config_classes and config_classes[loc]:
            all_locations_with_classes.append((loc, config_classes[loc]))
        elif loc in code_classes and code_classes[loc]:
            all_locations_with_classes.append((loc, code_classes[loc]))
    
    for i, (loc1, classes1) in enumerate(all_locations_with_classes):
        for loc2, classes2 in all_locations_with_classes[i+1:]:
            if classes1 != classes2:
                missing_in_1 = classes2 - classes1
                missing_in_2 = classes1 - classes2
                
                print(f"\n{Fore.YELLOW}⚠️ Perbedaan antara {loc1} dan {loc2}:{Style.RESET_ALL}")
                if missing_in_1:
                    print(f"  Kelas di {loc2} yang tidak ada di {loc1}: {', '.join(sorted(missing_in_1))}")
                if missing_in_2:
                    print(f"  Kelas di {loc1} yang tidak ada di {loc2}: {', '.join(sorted(missing_in_2))}")
            else:
                print(f"{Fore.GREEN}✓ {loc1} dan {loc2} konsisten{Style.RESET_ALL}")
    
    # Print recommendations
    print(f"\n{Fore.CYAN}📝 Rekomendasi:{Style.RESET_ALL}")
    if any(len(classes) == 0 for classes in config_classes.values()) or any(len(classes) == 0 for classes in code_classes.values()):
        print(f"{Fore.YELLOW}⚠️ Beberapa lokasi tidak memiliki kelas. Pastikan semua komponen memiliki definisi kelas yang benar.{Style.RESET_ALL}")
    
    # Check if there's a mix of prefix naming conventions (e.g., both '100k' and '100')
    all_classes = set()
    for classes in config_classes.values():
        all_classes.update(classes)
    for classes in code_classes.values():
        all_classes.update(classes)
    
    has_k_suffix = any(cls.endswith('k') for cls in all_classes if cls)
    has_numeric = any(cls.isdigit() or (cls.startswith('0') and cls.isdigit()[1:]) for cls in all_classes if cls)
    
    if has_k_suffix and has_numeric:
        print(f"{Fore.RED}❌ Ada campuran konvensi penamaan (mis. '100k' dan '100'). Seragamkan penamaan kelas.{Style.RESET_ALL}")
    
    # Check if layer prefixes are consistent
    layer_prefixes = {cls.split('_')[0] for cls in all_classes if '_' in cls}
    if len(layer_prefixes) > 3:  # We expect at most l1, l2, l3
        print(f"{Fore.RED}❌ Terlalu banyak prefiks layer berbeda: {', '.join(layer_prefixes)}{Style.RESET_ALL}")
        
    # Final recommendation based on overall consistency
    all_sizes = set(len(classes) for classes in config_classes.values() if classes) | set(len(classes) for classes in code_classes.values() if classes)
    if len(all_sizes) > 2:  # More than the expected 7 and 17
        print(f"{Fore.RED}❌ Jumlah kelas tidak konsisten di seluruh sistem. Silakan seragamkan.{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✅ Jumlah kelas konsisten di seluruh sistem.{Style.RESET_ALL}")
    
    # Find the most comprehensive class list to recommend
    most_complete = max(
        all_locations_with_classes, 
        key=lambda x: len(x[1]),
        default=(None, set())
    )
    
    if most_complete[0]:
        print(f"\n{Fore.GREEN}✅ Daftar kelas paling lengkap ditemukan di {most_complete[0]} ({len(most_complete[1])} kelas){Style.RESET_ALL}")
        print("Daftar kelas:")
        for cls in sorted(most_complete[1]):
            print(f"  - '{cls}'")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cek konsistensi kelas SmartCash")
    parser.add_argument('--config', '-c', type=str, default='configs/base_config.yaml',
                        help='Path ke file konfigurasi')
    parser.add_argument('--root', '-r', type=str, default='.',
                        help='Root direktori proyek SmartCash')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract classes from config
    config_classes = extract_classes_from_config(config)
    
    # Extract classes from code
    code_classes = extract_classes_from_code(args.root)
    
    # Analyze consistency
    analyze_consistency(config_classes, code_classes)

if __name__ == "__main__":
    main()