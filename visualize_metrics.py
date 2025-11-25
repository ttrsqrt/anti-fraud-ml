import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

METRICS_FILE = r"c:\Users\User\Documents\ttrsqr\model_metrics.json"
OUTPUT_DIR = r"c:\Users\User\Documents\ttrsqr"

def plot_metrics():
    # Load metrics
    with open(METRICS_FILE, 'r') as f:
        data = json.load(f)
    
    fold_metrics = data['fold_metrics']
    avg_metrics = data['average_metrics']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Metrics Across Folds', fontsize=16, fontweight='bold')
    
    # 1. ROC-AUC and PR-AUC by fold
    ax1 = axes[0, 0]
    folds = [m['fold'] for m in fold_metrics]
    roc_aucs = [m['roc_auc'] for m in fold_metrics]
    pr_aucs = [m['pr_auc'] for m in fold_metrics]
    
    x = np.arange(len(folds))
    width = 0.35
    
    ax1.bar(x - width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8, color='#2E86AB')
    ax1.bar(x + width/2, pr_aucs, width, label='PR-AUC', alpha=0.8, color='#A23B72')
    
    ax1.axhline(y=avg_metrics['roc_auc'], color='#2E86AB', linestyle='--', linewidth=2, label=f"Avg ROC-AUC: {avg_metrics['roc_auc']:.4f}")
    ax1.axhline(y=avg_metrics['pr_auc'], color='#A23B72', linestyle='--', linewidth=2, label=f"Avg PR-AUC: {avg_metrics['pr_auc']:.4f}")
    
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('AUC Scores by Fold', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Precision, Recall, F1, F2
    ax2 = axes[0, 1]
    metrics_data = {
        'Precision': [m['precision'] for m in fold_metrics],
        'Recall': [m['recall'] for m in fold_metrics],
        'F1-Score': [m['f1_score'] for m in fold_metrics],
        'F2-Score': [m['f2_score'] for m in fold_metrics]
    }
    
    for metric_name, values in metrics_data.items():
        ax2.plot(folds, values, marker='o', linewidth=2, label=metric_name)
    
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Classification Metrics by Fold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Average Confusion Matrix
    ax3 = axes[1, 0]
    
    # Sum all confusion matrices
    total_cm = np.zeros((2, 2))
    for m in fold_metrics:
        total_cm += np.array(m['confusion_matrix'])
    
    sns.heatmap(total_cm.astype(int), annot=True, fmt='d', cmap='Blues', 
                ax=ax3, cbar_kws={'label': 'Count'})
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('Actual', fontsize=12)
    ax3.set_title('Total Confusion Matrix (All Folds)', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(['Legitimate', 'Fraud'])
    ax3.set_yticklabels(['Legitimate', 'Fraud'])
    
    # 4. Average Metrics Bar Chart
    ax4 = axes[1, 1]
    
    avg_metric_names = list(avg_metrics.keys())
    avg_metric_values = list(avg_metrics.values())
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(avg_metric_names)))
    bars = ax4.barh(avg_metric_names, avg_metric_values, color=colors)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2, 
                f'{avg_metric_values[i]:.4f}', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Score', fontsize=12)
    ax4.set_title('Average Metrics Across All Folds', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 1.0)
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"{OUTPUT_DIR}/model_metrics_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("METRICS SUMMARY TABLE")
    print("="*80)
    
    df_metrics = pd.DataFrame(fold_metrics)
    df_metrics = df_metrics.drop(columns=['confusion_matrix'])
    print("\nPer-Fold Metrics:")
    print(df_metrics.to_string(index=False))
    
    print("\n" + "-"*80)
    print("Average Metrics:")
    for metric, value in avg_metrics.items():
        print(f"  {metric.upper():15s}: {value:.4f}")
    print("="*80)

if __name__ == "__main__":
    plot_metrics()
