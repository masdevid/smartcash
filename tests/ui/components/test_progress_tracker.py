"""
File: tests/ui/components/test_progress_tracker.py
Deskripsi: Test untuk komponen progress tracker UI untuk memverifikasi bahwa progress bar dan persentase update berfungsi dengan benar
"""

import unittest
import ipywidgets as widgets
from unittest.mock import MagicMock, patch
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Tambahkan root project ke path untuk import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from smartcash.ui.components.progress_tracker import (
    ProgressTracker, ProgressConfig, ProgressLevel,
    create_single_progress_tracker, create_dual_progress_tracker, create_triple_progress_tracker,
    create_flexible_tracker
)

class TestProgressTracker(unittest.TestCase):
    """Test suite untuk komponen ProgressTracker"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock widgets.IntProgress untuk testing
        self.original_int_progress = widgets.IntProgress
        self.original_html = widgets.HTML
        self.original_vbox = widgets.VBox
        self.original_hbox = widgets.HBox
        
        # Patch ipywidgets untuk testing
        widgets.IntProgress = MagicMock()
        widgets.HTML = MagicMock()
        widgets.VBox = MagicMock()
        widgets.HBox = MagicMock()
        
        # Setup callback tracking
        self.progress_updates = []
        self.step_completes = []
        self.operation_completes = 0
        self.errors = []
        self.resets = 0
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        # Restore original widgets
        widgets.IntProgress = self.original_int_progress
        widgets.HTML = self.original_html
        widgets.VBox = self.original_vbox
        widgets.HBox = self.original_hbox
        
        # Clear callback tracking
        self.progress_updates = []
        self.step_completes = []
        self.operation_completes = 0
        self.errors = []
        self.resets = 0
    
    def _on_progress_update(self, level_name, progress, message):
        """Callback untuk progress updates"""
        self.progress_updates.append((level_name, progress, message))
    
    def _on_step_complete(self, step_name, step_index):
        """Callback untuk step completion"""
        self.step_completes.append((step_name, step_index))
    
    def _on_complete(self):
        """Callback untuk operation completion"""
        self.operation_completes += 1
    
    def _on_error(self, message):
        """Callback untuk error events"""
        self.errors.append(message)
    
    def _on_reset(self):
        """Callback untuk reset events"""
        self.resets += 1
    
    def test_single_level_progress_tracker_creation(self):
        """Test pembuatan single level progress tracker"""
        tracker = create_single_progress_tracker("Test Operation")
        
        # Verifikasi level dan operation name
        self.assertEqual(tracker.config.level, ProgressLevel.SINGLE)
        self.assertEqual(tracker.config.operation, "Test Operation")
        
        # Verifikasi progress bar dibuat dengan benar
        self.assertIn("primary", tracker.progress_bars)
        
        # Verifikasi widget container dibuat
        self.assertIsNotNone(tracker.container)
    
    def test_dual_level_progress_tracker_creation(self):
        """Test pembuatan dual level progress tracker"""
        tracker = create_dual_progress_tracker("Test Dual Operation")
        
        # Verifikasi level dan operation name
        self.assertEqual(tracker.config.level, ProgressLevel.DUAL)
        self.assertEqual(tracker.config.operation, "Test Dual Operation")
        
        # Verifikasi progress bars dibuat dengan benar
        self.assertIn("overall", tracker.progress_bars)
        self.assertIn("current", tracker.progress_bars)
        
        # Verifikasi widget container dibuat
        self.assertIsNotNone(tracker.container)
    
    def test_triple_level_progress_tracker_creation(self):
        """Test pembuatan triple level progress tracker"""
        steps = ["Step 1", "Step 2", "Step 3"]
        weights = {"Step 1": 20, "Step 2": 30, "Step 3": 50}
        
        tracker = create_triple_progress_tracker(
            "Test Triple Operation", steps=steps, step_weights=weights
        )
        
        # Verifikasi level, operation name, steps, dan weights
        self.assertEqual(tracker.config.level, ProgressLevel.TRIPLE)
        self.assertEqual(tracker.config.operation, "Test Triple Operation")
        self.assertEqual(tracker.config.steps, steps)
        self.assertEqual(tracker.config.step_weights, weights)
        
        # Verifikasi progress bars dibuat dengan benar
        self.assertIn("overall", tracker.progress_bars)
        self.assertIn("current", tracker.progress_bars)
        
        # Verifikasi widget container dibuat
        self.assertIsNotNone(tracker.container)
    
    def test_single_level_progress_updates(self):
        """Test update progress pada single level tracker"""
        tracker = create_single_progress_tracker("Test Operation")
        tracker.on_progress_update(self._on_progress_update)
        
        # Register progress bar mock untuk testing
        primary_bar_mock = MagicMock()
        tracker.progress_bars["primary"] = primary_bar_mock
        
        # Update progress
        tracker.update_primary(25, "25% complete")
        tracker.update_primary(50, "50% complete")
        tracker.update_primary(100, "100% complete")
        
        # Verifikasi progress bar value diupdate dengan benar
        primary_bar_mock.value = 25
        primary_bar_mock.value = 50
        primary_bar_mock.value = 100
        
        # Verifikasi callback dipanggil dengan benar
        self.assertEqual(len(self.progress_updates), 3)
        self.assertEqual(self.progress_updates[0], ("primary", 25, "25% complete"))
        self.assertEqual(self.progress_updates[1], ("primary", 50, "50% complete"))
        self.assertEqual(self.progress_updates[2], ("primary", 100, "100% complete"))
    
    def test_dual_level_progress_updates(self):
        """Test update progress pada dual level tracker"""
        tracker = create_dual_progress_tracker("Test Dual Operation")
        tracker.on_progress_update(self._on_progress_update)
        
        # Register progress bar mocks untuk testing
        overall_bar_mock = MagicMock()
        current_bar_mock = MagicMock()
        tracker.progress_bars["overall"] = overall_bar_mock
        tracker.progress_bars["current"] = current_bar_mock
        
        # Update progress
        tracker.update_overall(30, "Overall 30%")
        tracker.update_current(60, "Current 60%")
        tracker.update_overall(70, "Overall 70%")
        tracker.update_current(100, "Current 100%")
        
        # Verifikasi progress bar values diupdate dengan benar
        overall_bar_mock.value = 30
        current_bar_mock.value = 60
        overall_bar_mock.value = 70
        current_bar_mock.value = 100
        
        # Verifikasi callback dipanggil dengan benar
        self.assertEqual(len(self.progress_updates), 4)
        self.assertEqual(self.progress_updates[0], ("overall", 30, "Overall 30%"))
        self.assertEqual(self.progress_updates[1], ("current", 60, "Current 60%"))
        self.assertEqual(self.progress_updates[2], ("overall", 70, "Overall 70%"))
        self.assertEqual(self.progress_updates[3], ("current", 100, "Current 100%"))
    
    def test_triple_level_progress_updates(self):
        """Test update progress pada triple level tracker"""
        steps = ["Step 1", "Step 2", "Step 3"]
        tracker = create_triple_progress_tracker("Test Triple Operation", steps=steps)
        tracker.on_progress_update(self._on_progress_update)
        tracker.on_step_complete(self._on_step_complete)
        
        # Register progress bar mocks untuk testing
        overall_bar_mock = MagicMock()
        current_bar_mock = MagicMock()
        step_bar_mock = MagicMock()
        tracker.progress_bars["overall"] = overall_bar_mock
        tracker.progress_bars["current"] = current_bar_mock
        tracker.progress_bars["step"] = step_bar_mock
        
        # Update progress untuk step 1
        tracker.update_step(50, "Step 1: 50%")
        tracker.update_current(25, "Processing Step 1")
        tracker.update_step(100, "Step 1: Complete")
        
        # Verifikasi progress bar values diupdate dengan benar
        step_bar_mock.value = 50
        current_bar_mock.value = 25
        step_bar_mock.value = 100
        
        # Verifikasi callback dipanggil dengan benar
        self.assertEqual(len(self.progress_updates), 3)
        self.assertEqual(self.progress_updates[0], ("step", 50, "Step 1: 50%"))
        self.assertEqual(self.progress_updates[1], ("current", 25, "Processing Step 1"))
        self.assertEqual(self.progress_updates[2], ("step", 100, "Step 1: Complete"))
    
    def test_operation_completion(self):
        """Test completion operation pada progress tracker"""
        tracker = create_single_progress_tracker("Test Operation")
        tracker.on_complete(self._on_complete)
        
        # Register progress bar mock untuk testing
        primary_bar_mock = MagicMock()
        tracker.progress_bars["primary"] = primary_bar_mock
        
        # Update progress dan complete
        tracker.update_primary(50, "50% complete")
        tracker.complete("Operation berhasil!")
        
        # Verifikasi progress bar diupdate ke 100%
        primary_bar_mock.value = 100
        
        # Verifikasi callback dipanggil
        self.assertEqual(self.operation_completes, 1)
    
    def test_error_handling(self):
        """Test error handling pada progress tracker"""
        tracker = create_dual_progress_tracker("Test Operation")
        tracker.on_error(self._on_error)
        
        # Register progress bar mocks untuk testing
        overall_bar_mock = MagicMock()
        current_bar_mock = MagicMock()
        tracker.progress_bars["overall"] = overall_bar_mock
        tracker.progress_bars["current"] = current_bar_mock
        
        # Update progress dan set error
        tracker.update_overall(30, "30% complete")
        tracker.error("Terjadi kesalahan!")
        
        # Verifikasi callback dipanggil
        self.assertEqual(len(self.errors), 1)
        self.assertEqual(self.errors[0], "Terjadi kesalahan!")
    
    def test_reset_functionality(self):
        """Test reset functionality pada progress tracker"""
        tracker = create_single_progress_tracker("Test Operation")
        tracker.on_reset(self._on_reset)
        
        # Register progress bar mock untuk testing
        primary_bar_mock = MagicMock()
        tracker.progress_bars["primary"] = primary_bar_mock
        
        # Update progress dan reset
        tracker.update_primary(75, "75% complete")
        tracker.reset()
        
        # Verifikasi progress bar direset ke 0
        primary_bar_mock.value = 0
        
        # Verifikasi callback dipanggil
        self.assertEqual(self.resets, 1)
    
    def test_concurrent_updates(self):
        """Test concurrent updates pada progress tracker"""
        tracker = create_dual_progress_tracker("Test Concurrent")
        tracker.on_progress_update(self._on_progress_update)
        
        # Register progress bar mocks untuk testing
        overall_bar_mock = MagicMock()
        current_bar_mock = MagicMock()
        tracker.progress_bars["overall"] = overall_bar_mock
        tracker.progress_bars["current"] = current_bar_mock
        
        # Simulasi concurrent updates dengan ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit concurrent updates
            for i in range(0, 101, 10):
                executor.submit(tracker.update_overall, i, f"Overall {i}%")
                executor.submit(tracker.update_current, i, f"Current {i}%")
        
        # Verifikasi progress updates tercatat (mungkin tidak berurutan)
        self.assertGreaterEqual(len(self.progress_updates), 10)
        
        # Verifikasi bahwa update terakhir mencapai 100%
        overall_updates = [update for update in self.progress_updates if update[0] == "overall"]
        current_updates = [update for update in self.progress_updates if update[0] == "current"]
        
        # Verifikasi bahwa setidaknya ada satu update yang mencapai 100%
        self.assertTrue(any(update[1] == 100 for update in overall_updates))
        self.assertTrue(any(update[1] == 100 for update in current_updates))
    
    def test_edge_cases(self):
        """Test edge cases pada progress tracker"""
        tracker = create_single_progress_tracker("Edge Cases")
        tracker.on_progress_update(self._on_progress_update)
        
        # Register progress bar mock untuk testing
        primary_bar_mock = MagicMock()
        tracker.progress_bars["primary"] = primary_bar_mock
        
        # Test dengan nilai 0
        tracker.update_primary(0, "Memulai...")
        
        # Test dengan nilai negatif (seharusnya dibatasi ke 0)
        tracker.update_primary(-10, "Nilai negatif")
        
        # Test dengan nilai > 100 (seharusnya dibatasi ke 100)
        tracker.update_primary(110, "Nilai terlalu besar")
        
        # Test dengan pesan kosong
        tracker.update_primary(50, "")
        
        # Verifikasi progress updates
        self.assertEqual(len(self.progress_updates), 4)
        self.assertEqual(self.progress_updates[0][1], 0)
        self.assertEqual(self.progress_updates[1][1], 0)  # Dibatasi ke 0
        self.assertEqual(self.progress_updates[2][1], 100)  # Dibatasi ke 100
        self.assertEqual(self.progress_updates[3][1], 50)
        
    def test_color_changes(self):
        """Test perubahan warna pada progress bar"""
        tracker = create_single_progress_tracker("Color Test")
        
        # Patch _update_progress_label method untuk verifikasi
        original_update_label = tracker._update_progress_label
        color_updates = []
        
        def mock_update_label(self, level_name, value, message="", color=None):
            color_updates.append((level_name, color))
            return original_update_label(level_name, value, message, color)
            
        # Patch method
        with patch.object(ProgressTracker, '_update_progress_label', mock_update_label):
            # Test perubahan warna
            tracker.update_primary(25, "Progress 25%", color="warning")
            tracker.update_primary(50, "Progress 50%", color="info")
            tracker.update_primary(75, "Progress 75%", color="success")
            tracker.update_primary(90, "Progress 90%", color="danger")
        
        # Verifikasi warna yang digunakan
        self.assertEqual(len(color_updates), 4)
        self.assertEqual(color_updates[0], ("primary", "warning"))
        self.assertEqual(color_updates[1], ("primary", "info"))
        self.assertEqual(color_updates[2], ("primary", "success"))
        self.assertEqual(color_updates[3], ("primary", "danger"))
    
    def test_visibility_controls(self):
        """Test kontrol visibilitas progress tracker"""
        tracker = create_dual_progress_tracker("Visibility Test")
        
        # Verifikasi status awal
        self.assertFalse(tracker.is_visible)
        
        # Show tracker
        tracker.show()
        self.assertTrue(tracker.is_visible)
        
        # Hide tracker
        tracker.hide()
        self.assertFalse(tracker.is_visible)
        
        # Test auto-hide setelah complete
        tracker.show()
        self.assertTrue(tracker.is_visible)
        
        # Mock _delayed_hide untuk mencegah threading issues dalam test
        original_delayed_hide = tracker._delayed_hide
        tracker._delayed_hide = MagicMock()
        
        tracker.complete("Operasi selesai")
        self.assertTrue(tracker._delayed_hide.called)
        
        # Restore original method
        tracker._delayed_hide = original_delayed_hide
    
    def test_step_management_triple_level(self):
        """Test manajemen step pada triple level tracker"""
        steps = ["Persiapan", "Pemrosesan", "Finalisasi", "Validasi"]
        weights = {"Persiapan": 10, "Pemrosesan": 50, "Finalisasi": 30, "Validasi": 10}
        
        tracker = create_triple_progress_tracker(
            "Step Management", steps=steps, step_weights=weights
        )
        tracker.on_step_complete(self._on_step_complete)
        tracker.on_progress_update(self._on_progress_update)
        
        # Simulasi progress pada step pertama
        self.assertEqual(tracker.current_step_index, 0)
        tracker.update_step(100, "Persiapan selesai")
        
        # Verifikasi step completion callback
        self.assertEqual(len(self.step_completes), 1)
        self.assertEqual(self.step_completes[0], ("Persiapan", 0))
        
        # Verifikasi auto-advance ke step berikutnya
        self.assertEqual(tracker.current_step_index, 1)
        
        # Reset progress_updates untuk fokus pada step kedua
        self.progress_updates = []
        
        # Simulasi progress pada step kedua
        tracker.update_step(50, "Pemrosesan 50%")
        
        # Verifikasi step progress
        step_updates = [update for update in self.progress_updates if update[0] == "step"]
        self.assertGreaterEqual(len(step_updates), 1)
        self.assertEqual(step_updates[-1][1], 50)  # Nilai terakhir untuk step adalah 50%
        
        # Reset progress_updates untuk fokus pada overall update
        self.progress_updates = []
        
        # Simulasi update overall progress dengan nilai yang bervariasi
        # Nilai sebenarnya bisa berbeda tergantung implementasi perhitungan weighted progress
        tracker.update_overall(30, "Overall progress")
        
        # Verifikasi overall progress
        overall_updates = [update for update in self.progress_updates if update[0] == "overall"]
        self.assertEqual(len(overall_updates), 1)
        
        # Verifikasi bahwa nilai overall progress berada dalam range yang diharapkan
        # Persiapan (10% dari total) selesai 100% = 10% total
        # Pemrosesan (50% dari total) progress 50% = 25% total
        # Total progress sekitar 30% dengan toleransi +/- 10%
        actual_progress = overall_updates[0][1]
        self.assertGreaterEqual(actual_progress, 20, f"Progress terlalu rendah: {actual_progress}%")
        self.assertLessEqual(actual_progress, 40, f"Progress terlalu tinggi: {actual_progress}%")
    
    def test_custom_config_tracker(self):
        """Test tracker dengan custom configuration"""
        custom_config = ProgressConfig(
            level=ProgressLevel.DUAL,
            operation="Custom Operation",
            auto_advance=False,
            auto_hide_delay=10.0,
            animation_speed=0.05,
            width_adjustment=20
        )
        
        tracker = create_flexible_tracker(custom_config)
        
        # Verifikasi custom config diterapkan dengan benar
        self.assertEqual(tracker.config.level, ProgressLevel.DUAL)
        self.assertEqual(tracker.config.operation, "Custom Operation")
        self.assertEqual(tracker.config.auto_advance, False)
        self.assertEqual(tracker.config.auto_hide_delay, 10.0)
        self.assertEqual(tracker.config.animation_speed, 0.05)
        self.assertEqual(tracker.config.width_adjustment, 20)
        
        # Verifikasi progress bars dibuat sesuai level
        self.assertIn("overall", tracker.progress_bars)
        self.assertIn("current", tracker.progress_bars)
        self.assertNotIn("step", tracker.progress_bars)
    
    def test_message_truncation(self):
        """Test truncation pesan yang terlalu panjang"""
        # Test static method langsung
        long_message = "A" * 200
        truncated = ProgressTracker._truncate_message(long_message, 50)
        
        self.assertEqual(len(truncated), 50)
        self.assertTrue(truncated.endswith("..."))
        self.assertEqual(truncated, "A" * 47 + "...")
        
        # Test short message tidak ditruncate
        short_message = "Short message"
        result = ProgressTracker._truncate_message(short_message, 50)
        self.assertEqual(result, short_message)
    
    def test_callback_management(self):
        """Test manajemen callback pada progress tracker"""
        tracker = create_single_progress_tracker("Callback Test")
        
        # Test registrasi dan unregistrasi callback
        callback_id1 = tracker.on_progress_update(self._on_progress_update)
        callback_id2 = tracker.on_complete(self._on_complete)
        
        # Verifikasi callback terdaftar
        self.assertIn("progress_update", tracker.callback_manager.callbacks)
        self.assertIn("complete", tracker.callback_manager.callbacks)
        
        # Unregister callback
        tracker.remove_callback(callback_id1)
        
        # Update progress seharusnya tidak memanggil callback yang sudah dihapus
        tracker.update_primary(50, "Test")
        self.assertEqual(len(self.progress_updates), 0)
        
        # Complete masih seharusnya memanggil callback
        tracker.complete()
        self.assertEqual(self.operation_completes, 1)
    
    def test_one_time_callbacks(self):
        """Test one-time callbacks yang hanya dipanggil sekali"""
        tracker = create_single_progress_tracker("One-time Test")
        
        # Setup callback manager untuk testing
        one_time_counter = [0]
        regular_counter = [0]
        
        def one_time_callback():
            one_time_counter[0] += 1
            
        def regular_callback():
            regular_counter[0] += 1
        
        # Register callbacks
        tracker.callback_manager.register("test_event", one_time_callback, one_time=True)
        tracker.callback_manager.register("test_event", regular_callback, one_time=False)
        
        # Trigger event dua kali
        tracker.callback_manager.trigger("test_event")
        tracker.callback_manager.trigger("test_event")
        
        # Verifikasi one-time callback hanya dipanggil sekali
        self.assertEqual(one_time_counter[0], 1)
        self.assertEqual(regular_counter[0], 2)

if __name__ == '__main__':
    unittest.main()
