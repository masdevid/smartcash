
import torch
import pytest
from smartcash.model.training.core.batch_processor import BatchProcessor

class TestBatchProcessorRecall:
    @pytest.fixture
    def batch_processor(self):
        """Fixture for BatchProcessor instance."""
        return BatchProcessor(conf_threshold=0.5, iou_threshold=0.5, debug=True)

    def test_process_batch_recall_calculation(self, batch_processor):
        """
        Test that process_batch correctly extracts target classes and calculates TP.
        This is a simplified test to ensure the fundamental components for recall are correct.
        """
        # Dummy predictions: [batch_idx, x1, y1, x2, y2, conf, class]
        # Prediction 1: class 0, high conf, matches target 0 (perfect overlap)
        # Prediction 2: class 1, high conf, matches target 1 (perfect overlap)
        # Prediction 3: class 0, low conf, should be filtered
        # Prediction 4: class 2, high conf, no matching target (FP)
        predictions = torch.tensor([
            [0, 0.1, 0.1, 0.2, 0.2, 0.9, 0],
            [0, 0.3, 0.3, 0.4, 0.4, 0.8, 1],
            [0, 0.5, 0.5, 0.6, 0.6, 0.4, 0], # Filtered by conf_threshold
            [0, 0.7, 0.7, 0.8, 0.8, 0.7, 2],
        ], dtype=torch.float32)

        # Dummy targets: [batch_idx, class, x, y, w, h]
        # Target 0: class 0 (perfect overlap with pred 1)
        # Target 1: class 1 (perfect overlap with pred 2)
        # Target 2: class 3 (no matching prediction, should be FN)
        targets = torch.tensor([
            [0, 0, 0.15, 0.15, 0.1, 0.1], # Center (0.15, 0.15), size 0.1x0.1 -> x1=0.1, y1=0.1, x2=0.2, y2=0.2
            [0, 1, 0.35, 0.35, 0.1, 0.1], # Center (0.35, 0.35), size 0.1x0.1 -> x1=0.3, y1=0.3, x2=0.4, y2=0.4
            [0, 3, 0.9, 0.9, 0.1, 0.1],
        ], dtype=torch.float32)

        tp, conf, pred_cls, target_cls = batch_processor.process_batch(predictions, targets)

        # Assert that target_cls contains all original target classes
        expected_target_classes = torch.tensor([0, 1, 3], dtype=torch.int32)
        assert torch.equal(target_cls.sort().values, expected_target_classes.sort().values), \
            f"Expected target_cls {expected_target_classes}, got {target_cls}"

        # Assert that true positives are correctly identified
        # Based on dummy data and iou_threshold=0.5, predictions 0 and 1 should be TP
        # Prediction 3 is filtered, Prediction 4 is FP
        # The `_match_predictions_to_targets` method (called internally) handles IoU matching.
        # For this simplified test, we'll assume perfect overlap for matching predictions.
        # The `tp` tensor should have a 1 for each matched prediction.
        # Given the dummy data, and assuming perfect overlap for simplicity,
        # predictions at index 0 (class 0) and 1 (class 1) should be TP.
        # The `tp` tensor is a boolean tensor of shape (num_predictions_after_filtering, 1)
        # The `_match_predictions_to_targets` function returns a boolean tensor.
        # We expect 2 true positives out of 3 predictions after confidence filtering.
        # Predictions after filtering: [0, 1, 3] (original indices)
        # TP for pred 0 (class 0) and pred 1 (class 1)
        # TP for pred 3 (class 2) should be False
        
        # Note: The actual `tp` values depend on the internal IoU matching.
        # For this test, we'll check the sum of true positives.
        # With conf_threshold=0.5 and iou_threshold=0.5, and perfect overlap for matched pairs:
        # Pred 0 (class 0, conf 0.9) matches Target 0 (class 0) -> TP
        # Pred 1 (class 1, conf 0.8) matches Target 1 (class 1) -> TP
        # Pred 2 (class 0, conf 0.4) -> filtered out
        # Pred 3 (class 2, conf 0.7) -> no matching target -> FP
        # So, we expect 2 true positives.
        assert tp.sum().item() == 2, f"Expected 2 true positives, got {tp.sum().item()}"

    def test_process_batch_empty_predictions(self, batch_processor):
        """Test handling of empty predictions."""
        predictions = torch.empty((0, 7), dtype=torch.float32)
        targets = torch.tensor([
            [0, 0, 0.1, 0.1, 0.1, 0.1],
            [0, 1, 0.3, 0.3, 0.1, 0.1],
        ], dtype=torch.float32)

        tp, conf, pred_cls, target_cls = batch_processor.process_batch(predictions, targets)

        assert tp.shape == (0, 1)
        assert conf.shape == (0,)
        assert pred_cls.shape == (0,)
        expected_target_classes = torch.tensor([0, 1], dtype=torch.int32)
        assert torch.equal(target_cls.sort().values, expected_target_classes.sort().values)

    def test_process_batch_empty_targets(self, batch_processor):
        """Test handling of empty targets."""
        predictions = torch.tensor([
            [0, 0.1, 0.1, 0.2, 0.2, 0.9, 0],
            [0, 0.3, 0.3, 0.4, 0.4, 0.8, 1],
        ], dtype=torch.float32)
        targets = torch.empty((0, 6), dtype=torch.float32)

        tp, conf, pred_cls, target_cls = batch_processor.process_batch(predictions, targets)

        assert tp.shape == (2, 1) # 2 predictions, all False (FP)
        assert conf.shape == (2,)
        assert pred_cls.shape == (2,)
        assert target_cls.shape == (0,) # No targets
        assert tp.sum().item() == 0 # No true positives
