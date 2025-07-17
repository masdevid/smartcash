#!/usr/bin/env python3
"""Debug script to isolate the downloader config handler issue."""

import sys
import traceback

print("Testing BaseDownloaderHandler first...")
try:
    from smartcash.ui.dataset.downloader.handlers.base_downloader_handler import BaseDownloaderHandler
    print("✅ BaseDownloaderHandler import OK")
    
    # Test instantiation
    handler = BaseDownloaderHandler()
    print("✅ BaseDownloaderHandler instantiation OK")
    
except Exception as e:
    print(f"❌ BaseDownloaderHandler error: {e}")
    traceback.print_exc()

print("\nTesting DownloaderConfigHandler...")
try:
    from smartcash.ui.dataset.downloader.configs.downloader_config_handler import DownloaderConfigHandler
    print("✅ DownloaderConfigHandler import OK")
    
    # Test instantiation
    handler = DownloaderConfigHandler()
    print("✅ DownloaderConfigHandler instantiation OK")
    
except Exception as e:
    print(f"❌ DownloaderConfigHandler error: {e}")
    traceback.print_exc()

print("\nTesting with parameters...")
try:
    handler = DownloaderConfigHandler(module_name='downloader', parent_module='dataset')
    print("✅ DownloaderConfigHandler with parameters OK")
    
except Exception as e:
    print(f"❌ DownloaderConfigHandler with parameters error: {e}")
    traceback.print_exc()