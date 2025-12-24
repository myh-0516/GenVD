#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from sklearn.metrics import precision_recall_fscore_support

CLASSES = ["CWE-119", "CWE-125", "CWE-189", "CWE-190", "CWE-20", 
           "CWE-200", "CWE-264", "CWE-362", "CWE-399", "CWE-416"]

def parse_metrics_file(metrics_file):
    with open(metrics_file, 'r') as f:
        content = f.read()
    
    acc = float(content.split('Accuracy: ')[1].split('\n')[0])
    f1 = float(content.split('Macro F1: ')[1].split('\n')[0])
    mcc = float(content.split('MCC): ')[1].split('\n')[0])
    
    cm_section = content.split('=== Confusion Matrix ===')[1]
    cm_lines = cm_section.split('[[')[1].split(']]')[0]
    
    rows = []
    for line in cm_lines.split(']'):
        line = line.strip().replace('[', '')
        if line:
            row = [int(x.strip()) for x in line.split() if x.strip().isdigit()]
            if row:
                rows.append(row)
    
    cm = np.array(rows)
    return cm, acc, f1, mcc


def calculate_per_class_metrics(cm):
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return precision, recall, f1


def plot_side_by_side_cm(baseline_data, proposed_data, model_name, output_path):
    cm_base, acc_base, f1_base, mcc_base = baseline_data
    cm_prop, acc_prop, f1_prop, mcc_prop = proposed_data
    
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    
    cm_base_norm = cm_base.astype('float') / cm_base.sum(axis=1)[:, np.newaxis]
    annotations_base = []
    for i in range(cm_base.shape[0]):
        row = []
        for j in range(cm_base.shape[1]):
            pct = cm_base_norm[i, j] * 100
            cnt = cm_base[i, j]
            row.append(f'{pct:.1f}%\n({cnt})')
        annotations_base.append(row)
    
    sns.heatmap(cm_base_norm, annot=annotations_base, fmt='', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0],
                cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)
    axes[0].set_xlabel('Predicted CWE', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True CWE', fontsize=13, fontweight='bold')
    axes[0].set_title(f'{model_name} - Classification',
                     fontsize=14, pad=15, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0].tick_params(axis='y', rotation=0, labelsize=10)
    
    cm_prop_norm = cm_prop.astype('float') / cm_prop.sum(axis=1)[:, np.newaxis]
    annotations_prop = []
    for i in range(cm_prop.shape[0]):
        row = []
        for j in range(cm_prop.shape[1]):
            pct = cm_prop_norm[i, j] * 100
            cnt = cm_prop[i, j]
            row.append(f'{pct:.1f}%\n({cnt})')
        annotations_prop.append(row)
    
    sns.heatmap(cm_prop_norm, annot=annotations_prop, fmt='', cmap='Greens',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1],
                cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted CWE', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True CWE', fontsize=13, fontweight='bold')
    axes[1].set_title(f'{model_name} - Generation',
                     fontsize=14, pad=15, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    axes[1].tick_params(axis='y', rotation=0, labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_f1_comparison(baseline_data, proposed_data, model_name, output_path):
    cm_base, acc_base, f1_base, mcc_base = baseline_data
    cm_prop, acc_prop, f1_prop, mcc_prop = proposed_data
    
    _, _, f1_base_per_class = calculate_per_class_metrics(cm_base)
    _, _, f1_prop_per_class = calculate_per_class_metrics(cm_prop)
    
    f1_diff = f1_prop_per_class - f1_base_per_class
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(CLASSES))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, f1_base_per_class, width, label='Traditional (Fine-tuning)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_prop_per_class, width, label='Generative (Prompt-based)',
                   color='seagreen', alpha=0.8)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, color='darkgreen', fontweight='bold')
    
    for i, diff in enumerate(f1_diff):
        if diff > 0:
            ax.annotate('', xy=(i + width/2, f1_prop_per_class[i]),
                       xytext=(i - width/2, f1_base_per_class[i]),
                       arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.6))
    
    ax.set_xlabel('CWE Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Per-Class F1 Score Comparison\n' +
                 f'Average Improvement: {np.mean(f1_diff):+.4f} (better in {(f1_diff > 0).sum()}/{len(CLASSES)} classes)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_improvement_heatmap(baseline_data, proposed_data, model_name, output_path):
    cm_base, acc_base, f1_base, mcc_base = baseline_data
    cm_prop, acc_prop, f1_prop, mcc_prop = proposed_data
    
    cm_base_norm = cm_base.astype('float') / cm_base.sum(axis=1)[:, np.newaxis]
    cm_prop_norm = cm_prop.astype('float') / cm_prop.sum(axis=1)[:, np.newaxis]
    
    cm_diff = cm_prop_norm - cm_base_norm
    
    annotations = []
    for i in range(cm_diff.shape[0]):
        row = []
        for j in range(cm_diff.shape[1]):
            diff_pct = cm_diff[i, j] * 100
            if abs(diff_pct) < 0.5:
                row.append('')
            else:
                row.append(f'{diff_pct:+.1f}%')
        annotations.append(row)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cm_diff, annot=annotations, fmt='', cmap='RdYlGn', center=0,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Improvement (Green = Better)'})
    
    ax.set_xlabel('Predicted CWE', fontsize=12, fontweight='bold')
    ax.set_ylabel('True CWE', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Improvement Analysis (Generative vs Traditional)\n' +
                 f'Diagonal values show improvement in correctly classifying each CWE',
                 fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)
    
    for i in range(len(CLASSES)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_models_comparison(results_dict, output_path):
    models = list(results_dict.keys())
    metrics_names = ['Accuracy', 'Macro F1', 'MCC']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        
        baseline_vals = []
        proposed_vals = []
        
        for model in models:
            base_data, prop_data = results_dict[model]
            cm_base, acc_base, f1_base, mcc_base = base_data
            cm_prop, acc_prop, f1_prop, mcc_prop = prop_data
            
            if metric_name == 'Accuracy':
                baseline_vals.append(acc_base * 100)
                proposed_vals.append(acc_prop * 100)
            elif metric_name == 'Macro F1':
                baseline_vals.append(f1_base * 100)
                proposed_vals.append(f1_prop * 100)
            else:  # MCC
                baseline_vals.append(mcc_base * 100)
                proposed_vals.append(mcc_prop * 100)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Traditional',
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, proposed_vals, width, label='Generative',
                      color='seagreen', alpha=0.8)
        
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10,
                   color='darkgreen', fontweight='bold')
        
        ax.set_ylabel(f'{metric_name} (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Multi-Model Performance Comparison: Traditional vs Generative Methods',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create comparison visualizations for RQ4')
    parser.add_argument('--results_dir', type=str, default='results/RQ4',
                       help='Root directory containing results')
    parser.add_argument('--output_dir', type=str, default='results/RQ4/plots',
                       help='Output directory for comparison plots')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    models = ['codebert', 'graphcodebert']
    results_dict = {}
    
    for model in models:
        baseline_file = os.path.join(results_dir, model, 'classify/top10cwe/metrics.txt')
        proposed_file = os.path.join(results_dir, model, 'generate/top10cwe/metrics.txt')
        
        if os.path.exists(baseline_file) and os.path.exists(proposed_file):
            try:
                baseline_data = parse_metrics_file(baseline_file)
                proposed_data = parse_metrics_file(proposed_file)
                results_dict[model] = (baseline_data, proposed_data)
                
                model_display = model.replace('codebert', 'CodeBERT').replace('graphcodebert', 'GraphCodeBERT')
                
                plot_side_by_side_cm(baseline_data, proposed_data, model_display,
                                    os.path.join(output_dir, f'{model}_comparison.png'))
            except Exception as e:
                pass
    
    print(f"\nConfusion matrix comparisons generated in: {output_dir}")

    for model, (baseline_data, proposed_data) in results_dict.items():
        _, acc_base, f1_base, mcc_base = baseline_data
        _, acc_prop, f1_prop, mcc_prop = proposed_data
        f1_improvement = ((f1_prop - f1_base) / f1_base) * 100
        print(f"  {model.upper():15s}: F1 {f1_base:.4f} → {f1_prop:.4f} ({f1_improvement:+.2f}%)")


if __name__ == '__main__':
    main()

