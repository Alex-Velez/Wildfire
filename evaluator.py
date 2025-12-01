"""Evaluation metrics and visualization for Wildfire Detection Models.
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - ROC Curve and AUC
    - Precision-Recall Curve
    - Classification Report
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.class_names = ["fire", "nofire"]
        
    def predict(self, dataloader: DataLoader):
        """Generate predictions for entire dataloader.
        Returns:
            true_labels, predicted_labels, predicted_probabilities
        """
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get model outputs
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def calculate_metrics(self, true_labels, pred_labels):
        """Calculate standard classification metrics.
        """
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='binary'),
            'recall': recall_score(true_labels, pred_labels, average='binary'),
            'f1_score': f1_score(true_labels, pred_labels, average='binary'),
            'precision_macro': precision_score(true_labels, pred_labels, average='macro'),
            'recall_macro': recall_score(true_labels, pred_labels, average='macro'),
            'f1_macro': f1_score(true_labels, pred_labels, average='macro'),
        }
        return metrics
    
    def plot_confusion_matrix(self, true_labels, pred_labels, save_path=None):
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, true_labels, pred_probs, save_path=None):
        # Use probability of fire = 1
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        plt.show()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, true_labels, pred_probs, save_path=None):
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_probs[:, 1])
        avg_precision = average_precision_score(true_labels, pred_probs[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model.model_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        plt.show()
        
        return avg_precision
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        main_metrics = {k: v for k, v in metrics_dict.items() 
                       if k in ['accuracy', 'precision', 'recall', 'f1_score']}
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(main_metrics.keys(), main_metrics.values(), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.ylim([0, 1.1])
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Performance Metrics - {self.model.model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")
        plt.show()
    
    def generate_classification_report(self, true_labels, pred_labels):
        
        report = classification_report(true_labels, pred_labels, 
                                      target_names=self.class_names,
                                      digits=4)
        return report
    
    def evaluate_full(self, dataloader: DataLoader, save_dir=None):
        """Perform full evaluation with all metrics and visualizations."""
        print(f"\n{'='*60}")
        print(f"Evaluating Model: {self.model.model_name}")
        print(f"Dataset Source: {dataloader.dataset.source}")
        print(f"{'='*60}\n")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model_save_dir = os.path.join(save_dir, str(self.model))
            os.makedirs(model_save_dir, exist_ok=True)
        else:
            model_save_dir = None
        
        true_labels, pred_labels, pred_probs = self.predict(dataloader)
        
        metrics = self.calculate_metrics(true_labels, pred_labels)
        
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title():<25}: {value:.4f}")
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        report = self.generate_classification_report(true_labels, pred_labels)
        print(report)
        
        # Confusion matrix
        save_path = os.path.join(model_save_dir, "confusion_matrix.png") if model_save_dir else None
        cm = self.plot_confusion_matrix(true_labels, pred_labels, save_path)
        
        # ROC curve
        save_path = os.path.join(model_save_dir, "roc_curve.png") if model_save_dir else None
        roc_auc = self.plot_roc_curve(true_labels, pred_probs, save_path)
        metrics['roc_auc'] = roc_auc
        
        # Precision-Recall curve
        save_path = os.path.join(model_save_dir, "precision_recall_curve.png") if model_save_dir else None
        avg_precision = self.plot_precision_recall_curve(true_labels, pred_probs, save_path)
        metrics['avg_precision'] = avg_precision
        
        # Metrics comparison
        save_path = os.path.join(model_save_dir, "metrics_comparison.png") if model_save_dir else None
        self.plot_metrics_comparison(metrics, save_path)
        
        if model_save_dir:
            metrics_file = os.path.join(model_save_dir, "metrics.txt")
            with open(metrics_file, 'w') as f:
                f.write(f"Model: {self.model.model_name}\n")
                f.write(f"Dataset: {dataloader.dataset.source}\n")
                f.write("="*60 + "\n\n")
                f.write("METRICS:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n" + "="*60 + "\n")
                f.write("CLASSIFICATION REPORT:\n")
                f.write(report)
            print(f"\nMetrics saved to {metrics_file}")
        
        print(f"\n{'='*60}")
        print("Evaluation Complete!")
        print(f"{'='*60}\n")
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'pred_probs': pred_probs
        }


def compare_models(evaluators: list, dataloader: DataLoader, save_dir=None):
    """Compare FullyConnectedNetwork and Resnet34Scratch."""
    all_metrics = []
    model_names = []
    
    for evaluator in evaluators:
        true_labels, pred_labels, pred_probs = evaluator.predict(dataloader)
        metrics = evaluator.calculate_metrics(true_labels, pred_labels)
        all_metrics.append(metrics)
        model_names.append(evaluator.model.model_name)
    
    # Comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metric_names)):
        values = [m[metric] for m in all_metrics]
        bars = ax.bar(model_names, values, color=colors[idx], alpha=0.7)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Evaluator module loaded successfully!")
    print("\nTo use this module:")
    print("1. Import: from evaluator import ModelEvaluator, compare_models")
    print("2. Create evaluator: evaluator = ModelEvaluator(model, device)")
    print("3. Run evaluation: results = evaluator.evaluate_full(test_dataloader, save_dir='results')")