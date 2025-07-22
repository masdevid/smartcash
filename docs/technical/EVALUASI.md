# 📊 Evaluasi Model

## 📋 Overview

Dokumen ini menjelaskan metrik dan metodologi evaluasi yang digunakan dalam SmartCash.

## 🎯 Evaluation Metrics

### 1. Detection Metrics
```python
class DetectionMetrics:
    """Calculate detection metrics."""
    
    def __init__(self):
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
        
    def calculate_metrics(self):
        """Calculate precision, recall, F1."""
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1
```

### 2. mAP Calculation
```python
def calculate_map(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5
):
    """Calculate mean Average Precision."""
    aps = []
    for c in range(num_classes):
        # Calculate AP for each class
        ap = average_precision(
            pred_boxes[c],
            true_boxes[c],
            iou_threshold
        )
        aps.append(ap)
    return np.mean(aps)
```

## 🔍 Evaluation Methods

### 1. Cross-Validation
```python
def k_fold_validation(
    model,
    dataset,
    k=5
):
    """Perform k-fold cross validation."""
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # Train and evaluate on fold
        train_data = dataset[train_idx]
        val_data = dataset[val_idx]
        
        model.train(train_data)
        score = model.evaluate(val_data)
        scores.append(score)
        
    return np.mean(scores), np.std(scores)
```

### 2. Confidence Analysis
```python
def analyze_confidence(
    predictions,
    confidence_thresholds
):
    """Analyze detection confidence."""
    results = []
    for conf in confidence_thresholds:
        # Filter predictions
        valid_preds = predictions[
            predictions[:, 4] > conf
        ]
        
        # Calculate metrics
        metrics = calculate_metrics(valid_preds)
        results.append(metrics)
        
    return results
```

## 📊 Performance Analysis

### 1. Speed Benchmarking
```python
def benchmark_speed(
    model,
    batch_sizes=[1, 4, 8, 16],
    iterations=100
):
    """Benchmark inference speed."""
    results = {}
    
    for bs in batch_sizes:
        times = []
        for _ in range(iterations):
            # Generate dummy input
            x = torch.randn(bs, 3, 640, 640)
            
            # Time inference
            start = time.time()
            with torch.no_grad():
                model(x)
            times.append(time.time() - start)
            
        results[bs] = {
            "mean": np.mean(times),
            "std": np.std(times),
            "fps": 1.0 / np.mean(times)
        }
        
    return results
```

### 2. Resource Usage
```python
def profile_resources(model):
    """Profile GPU memory and CPU usage."""
    results = {
        "gpu": {
            "allocated": torch.cuda.memory_allocated(),
            "cached": torch.cuda.memory_cached()
        },
        "cpu": {
            "memory": psutil.Process().memory_info().rss,
            "cpu_percent": psutil.Process().cpu_percent()
        }
    }
    return results
```

## 🔄 Continuous Evaluation

### 1. Regression Testing
```python
def regression_test(
    new_model,
    baseline_model,
    test_data
):
    """Compare against baseline model."""
    # Evaluate baseline
    baseline_metrics = evaluate_model(
        baseline_model,
        test_data
    )
    
    # Evaluate new model
    new_metrics = evaluate_model(
        new_model,
        test_data
    )
    
    # Compare metrics
    comparison = {}
    for metric in baseline_metrics:
        diff = new_metrics[metric] - baseline_metrics[metric]
        comparison[metric] = {
            "diff": diff,
            "percent": diff / baseline_metrics[metric] * 100
        }
        
    return comparison
```

### 2. Performance Monitoring
```python
def monitor_performance(
    model,
    data_stream,
    window_size=1000
):
    """Monitor performance over time."""
    metrics_history = []
    
    for batch in data_stream:
        # Evaluate batch
        metrics = evaluate_batch(model, batch)
        metrics_history.append(metrics)
        
        # Calculate rolling statistics
        if len(metrics_history) > window_size:
            metrics_history.pop(0)
            
        rolling_stats = calculate_statistics(
            metrics_history
        )
        
        # Check for degradation
        if detect_degradation(rolling_stats):
            alert_degradation()
            
    return metrics_history
```

## 📈 Results Visualization

### 1. Confusion Matrix
```python
def plot_confusion_matrix(
    predictions,
    ground_truth,
    class_names
):
    """Plot confusion matrix."""
    cm = confusion_matrix(ground_truth, predictions)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
```

### 2. PR Curve
```python
def plot_pr_curve(
    precisions,
    recalls,
    class_names
):
    """Plot Precision-Recall curve."""
    plt.figure(figsize=(10, 10))
    
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        plt.plot(r, p, label=class_names[i])
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
```

## 🔍 Error Analysis

### 1. False Positive Analysis
```python
def analyze_false_positives(
    predictions,
    ground_truth,
    images
):
    """Analyze false positive cases."""
    fp_cases = []
    
    for pred, gt, img in zip(predictions, ground_truth, images):
        # Find false positives
        fp_mask = (pred == 1) & (gt == 0)
        
        if fp_mask.any():
            fp_cases.append({
                "image": img,
                "prediction": pred[fp_mask],
                "confidence": confidences[fp_mask]
            })
            
    return fp_cases
```

### 2. Error Patterns
```python
def find_error_patterns(error_cases):
    """Analyze patterns in error cases."""
    patterns = {
        "lighting": [],
        "occlusion": [],
        "blur": [],
        "size": []
    }
    
    for case in error_cases:
        # Analyze image conditions
        if is_low_light(case["image"]):
            patterns["lighting"].append(case)
            
        if has_occlusion(case["boxes"]):
            patterns["occlusion"].append(case)
            
        if is_blurry(case["image"]):
            patterns["blur"].append(case)
            
        if is_small_object(case["boxes"]):
            patterns["size"].append(case)
            
    return patterns
```

## 📊 Reporting

### 1. Generate Report
```python
def generate_evaluation_report(
    model_name,
    metrics,
    plots,
    error_analysis
):
    """Generate evaluation report."""
    report = {
        "model": model_name,
        "date": datetime.now(),
        "metrics": {
            "map": metrics["map"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        },
        "performance": {
            "fps": metrics["fps"],
            "latency": metrics["latency"],
            "memory": metrics["memory"]
        },
        "analysis": {
            "error_patterns": error_analysis,
            "recommendations": generate_recommendations(
                metrics,
                error_analysis
            )
        }
    }
    return report
```

### 2. Export Results
```python
def export_results(results, format="json"):
    """Export evaluation results."""
    if format == "json":
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
            
    elif format == "csv":
        pd.DataFrame(results).to_csv(
            "results.csv",
            index=False
        )
        
    elif format == "pdf":
        generate_pdf_report(results)
```

## 🚀 Next Steps

1. [Model Optimization](OPTIMIZATION.md)
2. [Deployment](DEPLOYMENT.md)
3. [Monitoring](MONITORING.md)
