"""
WEEK 3: TRAINING LOSS MONITOR
Monitor training metrics, loss values, and provide visualizations
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from enum import Enum


class MetricType(Enum):
    """Types of metrics to track"""
    LOSS = "loss"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    ROUGE = "rouge"
    CUSTOM = "custom"


@dataclass
class TrainingMetric:
    """Single training metric data point"""
    epoch: int
    step: int
    metric_type: str
    value: float
    timestamp: str
    batch_size: int = 32
    learning_rate: float = 0.001
    additional_info: Dict = None
    
    def to_dict(self):
        return asdict(self)


class TrainingLossMonitor:
    """
    Monitor and track training metrics
    Supports multiple loss types, visualization, and reporting
    """
    
    def __init__(self, log_dir: str = "./training_logs"):
        """Initialize loss monitor"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics: List[TrainingMetric] = []
        self.checkpoints: Dict[str, Dict] = {}
        self.config: Dict = {}
        
        print(f"✓ Training logs directory: {self.log_dir}")
    
    # ==================== LOGGING ====================
    
    def log_metric(self,
                  epoch: int,
                  step: int,
                  metric_type: str,
                  value: float,
                  **kwargs) -> None:
        """
        Log a single metric
        
        Args:
            epoch: Training epoch number
            step: Training step/batch number
            metric_type: Type of metric (loss, accuracy, etc.)
            value: Metric value
            **kwargs: Additional info (batch_size, learning_rate, etc.)
        """
        metric = TrainingMetric(
            epoch=epoch,
            step=step,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now().isoformat(),
            batch_size=kwargs.get('batch_size', 32),
            learning_rate=kwargs.get('learning_rate', 0.001),
            additional_info=kwargs
        )
        
        self.metrics.append(metric)
        
        # Print progress
        print(f"[Epoch {epoch:3d}, Step {step:5d}] {metric_type}: {value:.6f} " +
              f"(LR: {metric.learning_rate:.0e}, BS: {metric.batch_size})")
    
    def log_batch(self,
                 epoch: int,
                 step: int,
                 loss: float,
                 accuracy: Optional[float] = None,
                 **kwargs) -> None:
        """
        Log metrics for a training batch
        """
        self.log_metric(epoch, step, "loss", loss, **kwargs)
        
        if accuracy is not None:
            self.log_metric(epoch, step, "accuracy", accuracy, **kwargs)
    
    def log_epoch(self,
                 epoch: int,
                 train_loss: float,
                 val_loss: float,
                 train_acc: Optional[float] = None,
                 val_acc: Optional[float] = None,
                 **kwargs) -> None:
        """
        Log end-of-epoch metrics
        """
        self.log_metric(epoch, 0, "train_loss", train_loss, **kwargs)
        self.log_metric(epoch, 0, "val_loss", val_loss, **kwargs)
        
        if train_acc is not None:
            self.log_metric(epoch, 0, "train_accuracy", train_acc, **kwargs)
        
        if val_acc is not None:
            self.log_metric(epoch, 0, "val_accuracy", val_acc, **kwargs)
    
    # ==================== CHECKPOINT MANAGEMENT ====================
    
    def save_checkpoint(self,
                       epoch: int,
                       model_state: Dict,
                       optimizer_state: Dict,
                       metrics: Dict,
                       is_best: bool = False) -> str:
        """
        Save training checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metrics": metrics
        }
        
        filename = f"checkpoint_epoch_{epoch:03d}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.checkpoints[filename] = checkpoint
        
        if is_best:
            best_path = self.log_dir / "best_checkpoint.json"
            with open(best_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"✓ Best checkpoint saved: {filename}")
        else:
            print(f"✓ Checkpoint saved: {filename}")
        
        return str(filepath)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint"""
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    # ==================== ANALYSIS ====================
    
    def get_best_epoch(self, metric_type: str = "val_loss") -> Dict:
        """Get best epoch for a metric"""
        filtered = [m for m in self.metrics if m.metric_type == metric_type]
        
        if not filtered:
            return None
        
        best_metric = min(filtered, key=lambda x: x.value) if "loss" in metric_type else max(filtered, key=lambda x: x.value)
        
        return {
            "epoch": best_metric.epoch,
            "value": best_metric.value,
            "timestamp": best_metric.timestamp
        }
    
    def get_metrics_by_epoch(self, epoch: int) -> List[TrainingMetric]:
        """Get all metrics for an epoch"""
        return [m for m in self.metrics if m.epoch == epoch]
    
    def get_metrics_by_type(self, metric_type: str) -> List[TrainingMetric]:
        """Get all metrics of a specific type"""
        return [m for m in self.metrics if m.metric_type == metric_type]
    
    def calculate_improvement(self, 
                            metric_type: str,
                            start_epoch: int,
                            end_epoch: int) -> Dict:
        """Calculate metric improvement between epochs"""
        start_metrics = [m.value for m in self.metrics 
                        if m.metric_type == metric_type and m.epoch == start_epoch]
        end_metrics = [m.value for m in self.metrics 
                      if m.metric_type == metric_type and m.epoch == end_epoch]
        
        if not start_metrics or not end_metrics:
            return None
        
        start_avg = sum(start_metrics) / len(start_metrics)
        end_avg = sum(end_metrics) / len(end_metrics)
        improvement = ((start_avg - end_avg) / start_avg) * 100
        
        return {
            "metric": metric_type,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "start_value": start_avg,
            "end_value": end_avg,
            "improvement_percent": improvement
        }
    
    # ==================== EXPORT ====================
    
    def export_to_csv(self, output_file: str = "training_metrics.csv") -> str:
        """Export metrics to CSV"""
        output_path = self.log_dir / output_file
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['epoch', 'step', 'metric_type', 'value', 'timestamp', 
                         'batch_size', 'learning_rate']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in self.metrics:
                row = {
                    'epoch': metric.epoch,
                    'step': metric.step,
                    'metric_type': metric.metric_type,
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'batch_size': metric.batch_size,
                    'learning_rate': metric.learning_rate
                }
                writer.writerow(row)
        
        print(f"✓ Exported to CSV: {output_path}")
        return str(output_path)
    
    def export_to_json(self, output_file: str = "training_metrics.json") -> str:
        """Export metrics to JSON"""
        output_path = self.log_dir / output_file
        
        data = {
            "total_metrics": len(self.metrics),
            "metrics": [m.to_dict() for m in self.metrics],
            "best_epochs": {}
        }
        
        # Find best epochs for each metric type
        metric_types = set(m.metric_type for m in self.metrics)
        for mt in metric_types:
            best = self.get_best_epoch(mt)
            if best:
                data["best_epochs"][mt] = best
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported to JSON: {output_path}")
        return str(output_path)
    
    # ==================== VISUALIZATION ====================
    
    def plot_metric(self,
                   metric_type: str,
                   save_path: Optional[str] = None,
                   title: Optional[str] = None) -> None:
        """Plot a specific metric over time"""
        metrics = self.get_metrics_by_type(metric_type)
        
        if not metrics:
            print(f"No metrics found for type: {metric_type}")
            return
        
        epochs = [m.epoch for m in metrics]
        values = [m.value for m in metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel(metric_type.replace('_', ' ').title())
        plt.title(title or f'{metric_type.replace("_", " ").title()} Over Time')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = self.log_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        
        plt.show()
    
    def plot_comparison(self,
                       metric_types: List[str],
                       save_path: Optional[str] = None) -> None:
        """Compare multiple metrics"""
        fig, axes = plt.subplots(len(metric_types), 1, figsize=(10, 4*len(metric_types)))
        
        if len(metric_types) == 1:
            axes = [axes]
        
        for ax, metric_type in zip(axes, metric_types):
            metrics = self.get_metrics_by_type(metric_type)
            if metrics:
                epochs = [m.epoch for m in metrics]
                values = [m.value for m in metrics]
                
                ax.plot(epochs, values, marker='o', linestyle='-', linewidth=2)
                ax.set_ylabel(metric_type.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
        
        plt.xlabel('Epoch')
        plt.tight_layout()
        
        if save_path:
            save_path = self.log_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved: {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """Plot training and validation curves"""
        train_loss = self.get_metrics_by_type("train_loss")
        val_loss = self.get_metrics_by_type("val_loss")
        
        if not (train_loss and val_loss):
            print("Training and validation loss metrics not found")
            return
        
        train_epochs = [m.epoch for m in train_loss]
        train_values = [m.value for m in train_loss]
        
        val_epochs = [m.epoch for m in val_loss]
        val_values = [m.value for m in val_loss]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_values, marker='o', label='Train Loss', linewidth=2)
        plt.plot(val_epochs, val_values, marker='s', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = self.log_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training curves saved: {save_path}")
        
        plt.show()
    
    # ==================== REPORTING ====================
    
    def generate_report(self) -> str:
        """Generate training report"""
        report = []
        report.append("=" * 80)
        report.append("TRAINING REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        report.append(f"Total Metrics Logged: {len(self.metrics)}")
        report.append(f"Metric Types: {set(m.metric_type for m in self.metrics)}")
        
        if self.metrics:
            min_epoch = min(m.epoch for m in self.metrics)
            max_epoch = max(m.epoch for m in self.metrics)
            report.append(f"Epoch Range: {min_epoch} - {max_epoch}")
        
        report.append("\nBest Epochs:")
        for metric_type in set(m.metric_type for m in self.metrics):
            best = self.get_best_epoch(metric_type)
            if best:
                report.append(f"  {metric_type}: Epoch {best['epoch']} (value: {best['value']:.6f})")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.log_dir / "training_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        metric_types = set(m.metric_type for m in self.metrics)
        
        summary = {
            "total_metrics": len(self.metrics),
            "metric_types": list(metric_types),
            "best_epochs": {}
        }
        
        for mt in metric_types:
            best = self.get_best_epoch(mt)
            if best:
                summary["best_epochs"][mt] = best
        
        return summary


# ==================== USAGE EXAMPLES ====================

def main():
    """Example usage"""
    
    print("="*80)
    print("WEEK 3: TRAINING LOSS MONITOR EXAMPLES")
    print("="*80)
    
    monitor = TrainingLossMonitor()
    
    # Simulate training
    print("\n[Simulating Training...]")
    for epoch in range(1, 6):
        for step in range(1, 101, 10):
            loss = 5.0 * (0.95 ** epoch) + 0.1 * (step / 100)
            accuracy = 0.5 + 0.08 * epoch
            
            monitor.log_batch(
                epoch=epoch,
                step=step,
                loss=loss,
                accuracy=accuracy,
                batch_size=32,
                learning_rate=0.001 * (0.9 ** epoch)
            )
        
        # Log epoch
        train_loss = 5.0 * (0.95 ** epoch)
        val_loss = train_loss * 1.1
        train_acc = 0.5 + 0.08 * epoch
        val_acc = train_acc - 0.02
        
        monitor.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )
        
        # Save checkpoint
        if epoch % 2 == 0:
            monitor.save_checkpoint(
                epoch=epoch,
                model_state={"weights": "..."},
                optimizer_state={"lr": 0.001},
                metrics={"loss": train_loss, "accuracy": train_acc},
                is_best=(epoch == 4)
            )
    
    print("\n" + "="*80)
    print("[Analysis]")
    print("="*80)
    
    # Get best epoch
    best_epoch = monitor.get_best_epoch("val_loss")
    print(f"\nBest Epoch (Val Loss): {best_epoch}")
    
    # Calculate improvement
    improvement = monitor.calculate_improvement("train_loss", 1, 5)
    print(f"\nTraining Loss Improvement (Epoch 1→5): {improvement['improvement_percent']:.2f}%")
    
    # Export
    print("\n[Exporting...]")
    monitor.export_to_csv()
    monitor.export_to_json()
    
    # Generate report
    print("\n[Report]")
    monitor.generate_report()
    
    # Summary
    summary = monitor.get_summary()
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
