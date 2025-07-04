#!/usr/bin/env python3
"""
File: debug_dependency_init.py
Deskripsi: Script untuk debug inisialisasi DependencyInitializer secara terisolasi
"""

import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_dependency_init')

# Tambahkan parent directory ke sys.path
sys.path.append('/Users/masdevid/Projects/smartcash')

try:
    from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer
    logger.info("✅ Berhasil mengimport DependencyInitializer")

    # Buat instance dan lakukan inisialisasi
    logger.info("🚀 Membuat instance DependencyInitializer...")
    dep_init = DependencyInitializer()
    logger.info("✅ Instance berhasil dibuat")

    logger.info("🔄 Memulai proses inisialisasi...")
    result = dep_init.initialize()
    logger.info("✅ Proses inisialisasi selesai")

    logger.info(f"📋 Hasil inisialisasi: {result}")
    if result['success']:
        logger.info("🎉 Inisialisasi berhasil!")
    else:
        logger.error(f"❌ Inisialisasi gagal: {result.get('error', 'Tidak ada detail error')}")

except Exception as e:
    logger.error(f"❌ Error saat debug inisialisasi: {str(e)}", exc_info=True)
