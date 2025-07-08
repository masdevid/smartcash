#!/usr/bin/env python3
"""
Test script for backup/restore functionality in pretrained module.
"""

import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.ui.model.pretrained.services.pretrained_service import PretrainedService


async def test_backup_restore_functionality():
    """Test the backup and restore functionality."""
    print("🧪 Testing Backup/Restore Functionality")
    print("=" * 50)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PretrainedService()
        
        # Create mock log callback
        log_messages = []
        def log_callback(message):
            log_messages.append(message)
            print(f"LOG: {message}")
        
        # Test 1: Backup non-existent file (should succeed)
        print("\n📋 Test 1: Backup non-existent file")
        print("-" * 30)
        
        non_existent_file = Path(temp_dir) / "non_existent.pt"
        result = service._backup_existing_file(str(non_existent_file), log_callback)
        assert result == True, "Should succeed for non-existent file"
        print("✅ Test 1 passed: Non-existent file backup handled correctly")
        
        # Test 2: Backup existing file
        print("\n📋 Test 2: Backup existing file")
        print("-" * 30)
        
        # Create a test file
        test_file = Path(temp_dir) / "yolov5s.pt"
        test_content = b"fake model data for testing"
        test_file.write_bytes(test_content)
        
        result = service._backup_existing_file(str(test_file), log_callback)
        assert result == True, "Backup should succeed"
        
        # Check backup file exists
        backup_file = Path(f"{test_file}.bak")
        assert backup_file.exists(), "Backup file should exist"
        assert backup_file.read_bytes() == test_content, "Backup content should match original"
        assert not test_file.exists(), "Original file should be moved"
        
        print("✅ Test 2 passed: Existing file backed up correctly")
        
        # Test 3: Restore backup file
        print("\n📋 Test 3: Restore backup file")
        print("-" * 30)
        
        result = service._restore_backup_file(str(test_file), log_callback)
        assert result == True, "Restore should succeed"
        
        # Check original file is restored
        assert test_file.exists(), "Original file should be restored"
        assert test_file.read_bytes() == test_content, "Restored content should match original"
        assert not backup_file.exists(), "Backup file should be removed"
        
        print("✅ Test 3 passed: Backup file restored correctly")
        
        # Test 4: Cleanup backup file
        print("\n📋 Test 4: Cleanup backup file")
        print("-" * 30)
        
        # Create backup again
        service._backup_existing_file(str(test_file), log_callback)
        
        result = service._cleanup_backup_file(str(test_file), log_callback)
        assert result == True, "Cleanup should succeed"
        
        # Check backup is removed
        backup_file = Path(f"{test_file}.bak")
        assert not backup_file.exists(), "Backup file should be cleaned up"
        
        print("✅ Test 4 passed: Backup file cleaned up correctly")
        
        # Test 5: Simulate download with backup/restore on failure
        print("\n📋 Test 5: Simulate download failure with backup restore")
        print("-" * 30)
        
        # Create original file
        original_content = b"original model data"
        test_file.write_bytes(original_content)
        
        # Backup original
        service._backup_existing_file(str(test_file), log_callback)
        
        # Simulate failed download (create incomplete file)
        incomplete_content = b"incomplete download"
        test_file.write_bytes(incomplete_content)
        
        # Restore backup (simulate download failure)
        service._restore_backup_file(str(test_file), log_callback)
        
        # Check original is restored
        assert test_file.exists(), "Original file should be restored"
        assert test_file.read_bytes() == original_content, "Original content should be restored"
        
        print("✅ Test 5 passed: Download failure scenario handled correctly")
        
        # Test 6: Simulate successful download with backup cleanup
        print("\n📋 Test 6: Simulate successful download with backup cleanup")
        print("-" * 30)
        
        # Create original file
        test_file.write_bytes(original_content)
        
        # Backup original
        service._backup_existing_file(str(test_file), log_callback)
        
        # Simulate successful download
        new_content = b"new downloaded model data"
        test_file.write_bytes(new_content)
        
        # Cleanup backup (simulate successful download)
        service._cleanup_backup_file(str(test_file), log_callback)
        
        # Check new file exists and backup is gone
        assert test_file.exists(), "New file should exist"
        assert test_file.read_bytes() == new_content, "New content should be present"
        backup_file = Path(f"{test_file}.bak")
        assert not backup_file.exists(), "Backup should be cleaned up"
        
        print("✅ Test 6 passed: Successful download scenario handled correctly")
        
        print("\n" + "=" * 50)
        print("🎉 All backup/restore tests passed!")
        print("✅ File backup before download: Working")
        print("✅ Backup restoration on failure: Working")
        print("✅ Backup cleanup on success: Working")
        print("✅ Non-existent file handling: Working")
        
        # Print log summary
        print(f"\n📋 Log messages captured: {len(log_messages)}")
        backup_logs = [msg for msg in log_messages if "💾 Backed up" in msg]
        restore_logs = [msg for msg in log_messages if "🔄 Restored" in msg]
        cleanup_logs = [msg for msg in log_messages if "🧹 Cleaned up" in msg]
        
        print(f"  💾 Backup operations: {len(backup_logs)}")
        print(f"  🔄 Restore operations: {len(restore_logs)}")
        print(f"  🧹 Cleanup operations: {len(cleanup_logs)}")
        
        return True


async def test_integration_with_download():
    """Test backup functionality integration with actual download methods."""
    print("\n🧪 Testing Integration with Download Methods")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PretrainedService()
        
        log_messages = []
        def log_callback(message):
            log_messages.append(message)
            print(f"LOG: {message}")
        
        progress_updates = []
        def progress_callback(percent, message):
            progress_updates.append((percent, message))
            print(f"PROGRESS: {percent}% - {message}")
        
        # Create existing model file to test backup
        models_dir = temp_dir
        yolo_file = Path(models_dir) / "yolov5s.pt"
        original_content = b"existing yolov5s model data" * 1000  # Make it larger
        yolo_file.write_bytes(original_content)
        
        print(f"📁 Created existing model file: {yolo_file.name} ({len(original_content)} bytes)")
        
        # Mock the download process to fail
        def mock_failed_download(url, file_path, progress_cb, log_cb):
            # Simulate partial download
            Path(file_path).write_bytes(b"partial download data")
            return False  # Simulate download failure
        
        # Replace the download method temporarily
        original_method = service._download_file_with_progress
        service._download_file_with_progress = mock_failed_download
        
        try:
            # Attempt download (should fail and restore backup)
            print("\n📥 Testing failed download with backup restore...")
            result = await service.download_yolov5s(
                models_dir, 
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            # Check result
            assert result == False, "Download should have failed"
            
            # Check original file is restored
            assert yolo_file.exists(), "Original file should be restored"
            assert yolo_file.read_bytes() == original_content, "Original content should be restored"
            
            print("✅ Failed download handled correctly - original file restored")
            
        finally:
            # Restore original method
            service._download_file_with_progress = original_method
        
        # Check log messages for backup operations
        backup_mentioned = any("💾 Backed up" in msg for msg in log_messages)
        restore_mentioned = any("🔄 Restored" in msg for msg in log_messages)
        
        assert backup_mentioned, "Backup operation should be logged"
        assert restore_mentioned, "Restore operation should be logged"
        
        print("✅ Backup and restore operations properly logged")
        print(f"📊 Total log messages: {len(log_messages)}")
        print(f"📊 Total progress updates: {len(progress_updates)}")
        
        return True


async def main():
    """Run all backup functionality tests."""
    print("🚀 Starting Pretrained Module Backup Tests")
    
    try:
        # Test basic backup/restore functionality
        await test_backup_restore_functionality()
        
        # Test integration with download methods
        await test_integration_with_download()
        
        print("\n" + "=" * 50)
        print("🎉 ALL BACKUP TESTS PASSED!")
        print("✅ The backup/restore functionality is working correctly")
        print("✅ Download failures will restore original files")
        print("✅ Successful downloads will clean up backup files")
        print("✅ Integration with download methods is working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)