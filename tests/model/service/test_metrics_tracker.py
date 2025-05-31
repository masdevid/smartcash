"""
File: tests/model/service/test_metrics_tracker.py
Deskripsi: Unit test untuk MetricsTracker
"""

import unittest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from collections import deque
from smartcash.model.service.metrics_tracker import MetricsTracker
from smartcash.model.service.callback_interfaces import MetricsCallback

class TestMetricsTracker(unittest.TestCase):
    """Test case untuk MetricsTracker"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        self.mock_callback = MagicMock(spec=MetricsCallback)
        self.metrics_tracker = MetricsTracker(self.mock_callback)
        
    def test_initialization(self):
        """Test inisialisasi MetricsTracker"""
        # Test default values
        self.assertIsInstance(self.metrics_tracker._metrics_history, dict)
        self.assertIsInstance(self.metrics_tracker._current_metrics, dict)
        self.assertIsInstance(self.metrics_tracker._best_metrics, dict)
        self.assertIsInstance(self.metrics_tracker._metrics_improvement, dict)
        self.assertIsNotNone(self.metrics_tracker._start_time)
        self.assertEqual(self.metrics_tracker._epoch_times, [])
        self.assertIsInstance(self.metrics_tracker._batch_times, deque)
        self.assertIsNotNone(self.metrics_tracker._last_time)
        
    def test_update_metrics(self):
        """Test update metrics"""
        # Update metrics
        metrics = {"loss": 0.5, "accuracy": 0.8}
        self.metrics_tracker.update(metrics, "train")
        
        # Verify callback was called
        self.mock_callback.update_metrics.assert_called_once_with(metrics, "train")
        
        # Verify metrics were updated
        self.assertEqual(self.metrics_tracker._current_metrics["train"]["loss"], 0.5)
        self.assertEqual(self.metrics_tracker._current_metrics["train"]["accuracy"], 0.8)
        self.assertEqual(list(self.metrics_tracker._metrics_history["train"]["loss"]), [0.5])
        self.assertEqual(list(self.metrics_tracker._metrics_history["train"]["accuracy"]), [0.8])
        
    def test_update_best_metrics(self):
        """Test update best metrics"""
        # Update metrics multiple times
        self.metrics_tracker.update({"loss": 0.5}, "val")
        self.metrics_tracker.update({"loss": 0.3}, "val")
        self.metrics_tracker.update({"loss": 0.4}, "val")
        
        # Verify best metrics
        self.assertEqual(self.metrics_tracker._best_metrics["val"]["loss"], 0.3)
        
        # Verify improvement
        self.assertGreater(self.metrics_tracker._metrics_improvement["val"]["loss"], 0)
        
    def test_update_best_metrics_higher_better(self):
        """Test update best metrics for metrics where higher is better"""
        # Update metrics multiple times
        self.metrics_tracker.update({"accuracy": 0.7}, "val")
        self.metrics_tracker.update({"accuracy": 0.9}, "val")
        self.metrics_tracker.update({"accuracy": 0.8}, "val")
        
        # Verify best metrics
        self.assertEqual(self.metrics_tracker._best_metrics["val"]["accuracy"], 0.9)
        
        # Verify improvement
        self.assertGreater(self.metrics_tracker._metrics_improvement["val"]["accuracy"], 0)
        
    def test_update_learning_rate(self):
        """Test update learning rate"""
        # Update learning rate
        self.metrics_tracker.update_learning_rate(0.001)
        
        # Verify callback was called
        self.mock_callback.update_learning_rate.assert_called_once_with(0.001)
        
        # Verify learning rate was updated
        self.assertEqual(self.metrics_tracker._current_metrics["train"]["learning_rate"], 0.001)
        self.assertEqual(list(self.metrics_tracker._metrics_history["train"]["learning_rate"]), [0.001])
        
    def test_update_loss_breakdown(self):
        """Test update loss breakdown"""
        # Update loss breakdown
        loss_components = {"box": 0.2, "obj": 0.1, "cls": 0.2}
        self.metrics_tracker.update_loss_breakdown(loss_components)
        
        # Verify callback was called
        self.mock_callback.update_loss_breakdown.assert_called_once_with(loss_components)
        
        # Verify loss breakdown was updated
        self.assertEqual(self.metrics_tracker._current_metrics["train"]["loss_box"], 0.2)
        self.assertEqual(self.metrics_tracker._current_metrics["train"]["loss_obj"], 0.1)
        self.assertEqual(self.metrics_tracker._current_metrics["train"]["loss_cls"], 0.2)
        
    def test_update_prediction_samples(self):
        """Test update prediction samples"""
        # Update prediction samples
        samples = [{"image": "img1.jpg", "predictions": [1, 2, 3]}]
        self.metrics_tracker.update_prediction_samples(samples)
        
        # Verify callback was called
        self.mock_callback.update_prediction_samples.assert_called_once_with(samples)
        
    def test_get_current_metrics(self):
        """Test get_current_metrics"""
        # Update metrics
        self.metrics_tracker.update({"loss": 0.5, "accuracy": 0.8}, "train")
        self.metrics_tracker.update({"loss": 0.4}, "val")
        
        # Get current metrics
        train_metrics = self.metrics_tracker.get_current_metrics("train")
        val_metrics = self.metrics_tracker.get_current_metrics("val")
        
        # Verify metrics
        self.assertEqual(train_metrics["loss"], 0.5)
        self.assertEqual(train_metrics["accuracy"], 0.8)
        self.assertEqual(val_metrics["loss"], 0.4)
        
    def test_get_best_metrics(self):
        """Test get_best_metrics"""
        # Update metrics multiple times
        self.metrics_tracker.update({"loss": 0.5}, "val")
        self.metrics_tracker.update({"loss": 0.3}, "val")
        self.metrics_tracker.update({"loss": 0.4}, "val")
        
        # Get best metrics
        best_metrics = self.metrics_tracker.get_best_metrics("val")
        
        # Verify best metrics
        self.assertEqual(best_metrics["loss"], 0.3)
        
    def test_get_metrics_improvement(self):
        """Test get_metrics_improvement"""
        # Update metrics multiple times
        self.metrics_tracker.update({"loss": 0.5}, "val")
        self.metrics_tracker.update({"loss": 0.25}, "val")  # 50% improvement
        
        # Get metrics improvement
        improvement = self.metrics_tracker.get_metrics_improvement("val")
        
        # Verify improvement (should be close to 50%)
        self.assertAlmostEqual(improvement["loss"], 50.0, delta=1.0)
        
    def test_get_metrics_history(self):
        """Test get_metrics_history"""
        # Update metrics multiple times
        self.metrics_tracker.update({"loss": 0.5}, "train")
        self.metrics_tracker.update({"loss": 0.4}, "train")
        self.metrics_tracker.update({"loss": 0.3}, "train")
        
        # Get metrics history
        history = self.metrics_tracker.get_metrics_history("train", "loss")
        
        # Verify history
        self.assertEqual(history, [0.5, 0.4, 0.3])
        
    def test_get_metrics_history_all(self):
        """Test get_metrics_history for all metrics"""
        # Update metrics multiple times
        self.metrics_tracker.update({"loss": 0.5, "accuracy": 0.7}, "train")
        self.metrics_tracker.update({"loss": 0.4, "accuracy": 0.8}, "train")
        
        # Get all metrics history
        history = self.metrics_tracker.get_metrics_history("train")
        
        # Verify history
        self.assertEqual(history["loss"], [0.5, 0.4])
        self.assertEqual(history["accuracy"], [0.7, 0.8])
        
    def test_get_average_batch_time(self):
        """Test get_average_batch_time"""
        # Mock batch times
        self.metrics_tracker._batch_times = deque([0.1, 0.2, 0.3])
        
        # Get average batch time
        avg_time = self.metrics_tracker.get_average_batch_time()
        
        # Verify average time (menggunakan assertAlmostEqual untuk menangani presisi floating point)
        self.assertAlmostEqual(avg_time, 0.2, places=5)
        
    def test_get_average_epoch_time(self):
        """Test get_average_epoch_time"""
        # Mock epoch times
        self.metrics_tracker._epoch_times = [10, 20, 30]
        
        # Get average epoch time
        avg_time = self.metrics_tracker.get_average_epoch_time()
        
        # Verify average time
        self.assertEqual(avg_time, 20)
        
    def test_get_estimated_epoch_time(self):
        """Test get_estimated_epoch_time"""
        # Mock batch times
        self.metrics_tracker._batch_times = deque([0.1, 0.2, 0.3])
        
        # Get estimated epoch time
        estimated_time = self.metrics_tracker.get_estimated_epoch_time()
        
        # Verify estimated time (0.2 * 100 = 20) (menggunakan assertAlmostEqual untuk menangani presisi floating point)
        self.assertAlmostEqual(estimated_time, 20, places=5)
        
    def test_get_estimated_remaining_time(self):
        """Test get_estimated_remaining_time"""
        # Mock epoch times
        self.metrics_tracker._epoch_times = [10, 20, 30]
        
        # Get estimated remaining time (current=2, total=10, remaining=8)
        estimated_time = self.metrics_tracker.get_estimated_remaining_time(2, 10)
        
        # Verify estimated time (20 * 8 = 160)
        self.assertEqual(estimated_time, 160)
        
    def test_get_metrics_summary(self):
        """Test get_metrics_summary"""
        # Update metrics
        self.metrics_tracker.update({"loss": 0.5}, "train")
        self.metrics_tracker.update({"loss": 0.4}, "val")
        
        # Get metrics summary
        summary = self.metrics_tracker.get_metrics_summary()
        
        # Verify summary
        self.assertIn("current", summary)
        self.assertIn("best", summary)
        self.assertIn("improvement", summary)
        self.assertIn("timing", summary)
        self.assertEqual(summary["current"]["train"]["loss"], 0.5)
        self.assertEqual(summary["current"]["val"]["loss"], 0.4)
        
    def test_reset(self):
        """Test reset"""
        # Update metrics
        self.metrics_tracker.update({"loss": 0.5}, "train")
        
        # Reset metrics tracker
        self.metrics_tracker.reset()
        
        # Verify metrics were reset
        self.assertEqual(self.metrics_tracker._current_metrics, {})
        self.assertEqual(self.metrics_tracker._best_metrics, {})
        self.assertEqual(self.metrics_tracker._metrics_improvement, {})
        self.assertEqual(self.metrics_tracker._epoch_times, [])
        self.assertEqual(len(self.metrics_tracker._batch_times), 0)
        
    def test_is_best_metric(self):
        """Test is_best_metric"""
        # Update metrics
        self.metrics_tracker.update({"loss": 0.5}, "val")
        
        # Check if better
        is_better_loss = self.metrics_tracker.is_best_metric("loss", 0.3, "val")
        is_worse_loss = self.metrics_tracker.is_best_metric("loss", 0.7, "val")
        
        # Verify results
        self.assertTrue(is_better_loss)
        self.assertFalse(is_worse_loss)
        
        # Check for metrics where higher is better
        self.metrics_tracker.update({"accuracy": 0.7}, "val")
        is_better_acc = self.metrics_tracker.is_best_metric("accuracy", 0.9, "val")
        is_worse_acc = self.metrics_tracker.is_best_metric("accuracy", 0.5, "val")
        
        # Verify results
        self.assertTrue(is_better_acc)
        self.assertFalse(is_worse_acc)
        
    def test_dict_callback(self):
        """Test dict callback"""
        # Create dict callback
        dict_callback = {
            'metrics': MagicMock(),
            'learning_rate': MagicMock(),
            'loss_breakdown': MagicMock(),
            'prediction_samples': MagicMock(),
            'inference_time': MagicMock()
        }
        
        # Create metrics tracker with dict callback
        metrics_tracker = MetricsTracker(dict_callback)
        
        # Test update
        metrics = {"loss": 0.5}
        metrics_tracker.update(metrics, "train")
        dict_callback['metrics'].assert_called_once_with(metrics, "train")
        
        # Test update_learning_rate
        metrics_tracker.update_learning_rate(0.001)
        dict_callback['learning_rate'].assert_called_once_with(0.001)
        
        # Test update_loss_breakdown
        loss_components = {"box": 0.2, "obj": 0.1, "cls": 0.2}
        metrics_tracker.update_loss_breakdown(loss_components)
        dict_callback['loss_breakdown'].assert_called_once_with(loss_components)
        
        # Test update_prediction_samples
        samples = [{"image": "img1.jpg", "predictions": [1, 2, 3]}]
        metrics_tracker.update_prediction_samples(samples)
        dict_callback['prediction_samples'].assert_called_once_with(samples)
        
        # Test update_inference_time
        metrics_tracker.update_inference_time(0.05)
        dict_callback['inference_time'].assert_called_once_with(0.05)
        
    def test_function_callback(self):
        """Test function callback"""
        # Buat class untuk function callback
        class FunctionCallback:
            def __call__(self, **kwargs):
                # Simpan argumen yang diterima
                self.last_call = kwargs
                
        # Buat instance function callback
        function_callback = FunctionCallback()
        
        # Buat metrics tracker dengan function callback
        metrics_tracker = MetricsTracker(function_callback)
        
        # Test update
        metrics = {"loss": 0.5}
        metrics_tracker.update(metrics, "train")
        self.assertEqual(function_callback.last_call["event"], "metrics")
        self.assertEqual(function_callback.last_call["metrics"], metrics)
        self.assertEqual(function_callback.last_call["phase"], "train")
        
        # Test update_learning_rate
        metrics_tracker.update_learning_rate(0.001)
        self.assertEqual(function_callback.last_call["event"], "learning_rate")
        self.assertEqual(function_callback.last_call["lr"], 0.001)
        
        # Test update_loss_breakdown
        loss_components = {"box": 0.2, "obj": 0.1, "cls": 0.2}
        metrics_tracker.update_loss_breakdown(loss_components)
        self.assertEqual(function_callback.last_call["event"], "loss_breakdown")
        self.assertEqual(function_callback.last_call["components"], loss_components)
        
        # Test update_prediction_samples
        samples = [{"image": "img1.jpg", "predictions": [1, 2, 3]}]
        metrics_tracker.update_prediction_samples(samples)
        self.assertEqual(function_callback.last_call["event"], "prediction_samples")
        self.assertEqual(function_callback.last_call["samples"], samples)
        
        # Test update_inference_time
        metrics_tracker.update_inference_time(0.05)
        self.assertEqual(function_callback.last_call["event"], "inference_time")
        self.assertEqual(function_callback.last_call["time"], 0.05)
        
    def test_no_callback(self):
        """Test no callback"""
        # Create metrics tracker with no callback
        metrics_tracker = MetricsTracker(None)
        
        # Test all methods (should not raise exceptions)
        metrics_tracker.update({"loss": 0.5}, "train")
        metrics_tracker.update_learning_rate(0.001)
        metrics_tracker.update_loss_breakdown({"box": 0.2, "obj": 0.1, "cls": 0.2})
        metrics_tracker.update_prediction_samples([{"image": "img1.jpg", "predictions": [1, 2, 3]}])
        metrics_tracker.update_inference_time(0.05)
        
        # Verify metrics were updated correctly
        self.assertEqual(metrics_tracker._current_metrics["train"]["loss"], 0.5)
        self.assertEqual(metrics_tracker._current_metrics["train"]["learning_rate"], 0.001)
        self.assertEqual(metrics_tracker._current_metrics["train"]["loss_box"], 0.2)
        
    def test_properties(self):
        """Test properties"""
        # Update metrics
        self.metrics_tracker.update({"loss": 0.5}, "train")
        self.metrics_tracker.update({"loss": 0.4, "mAP": 0.6}, "val")
        
        # Test properties
        self.assertEqual(self.metrics_tracker.current_loss, 0.5)
        self.assertEqual(self.metrics_tracker.best_val_loss, 0.4)
        self.assertEqual(self.metrics_tracker.best_val_map, 0.6)
        self.assertIsInstance(self.metrics_tracker.total_time, float)

    def test_update_precision_recall_f1(self):
        """Test update precision, recall, dan F1-score"""
        # Update metrics dengan precision, recall, dan F1-score
        metrics = {
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81
        }
        self.metrics_tracker.update(metrics, "val")
        
        # Verify callback was called
        self.mock_callback.update_metrics.assert_called_once_with(metrics, "val")
        
        # Verify metrics were updated
        self.assertEqual(self.metrics_tracker._current_metrics["val"]["precision"], 0.85)
        self.assertEqual(self.metrics_tracker._current_metrics["val"]["recall"], 0.78)
        self.assertEqual(self.metrics_tracker._current_metrics["val"]["f1_score"], 0.81)
        
    def test_update_map(self):
        """Test update mAP"""
        # Update metrics dengan mAP
        metrics = {"mAP": 0.65, "loss": 0.4}
        self.metrics_tracker.update(metrics, "val")
        
        # Verify callback was called
        self.mock_callback.update_metrics.assert_called_once_with(metrics, "val")
        
        # Verify metrics were updated
        self.assertEqual(self.metrics_tracker._current_metrics["val"]["mAP"], 0.65)
        self.assertEqual(self.metrics_tracker._current_metrics["val"]["loss"], 0.4)
        
    def test_update_inference_time(self):
        """Test update inference time"""
        # Update inference time
        self.metrics_tracker.update_inference_time(0.05)
        
        # Verify callback was called
        self.mock_callback.update_inference_time.assert_called_once_with(0.05)
        
        # Verify inference time was updated
        self.assertEqual(self.metrics_tracker._current_metrics["inference"]["time"], 0.05)
        self.assertEqual(list(self.metrics_tracker._metrics_history["inference"]["time"]), [0.05])
        
    def test_get_average_inference_time(self):
        """Test get_average_inference_time"""
        # Mock inference times
        self.metrics_tracker._inference_times = deque([0.03, 0.05, 0.07])
        
        # Get average inference time
        avg_time = self.metrics_tracker.get_average_inference_time()
        
        # Verify average time
        self.assertAlmostEqual(avg_time, 0.05, places=5)

if __name__ == '__main__':
    unittest.main()
