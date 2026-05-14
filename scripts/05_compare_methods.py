
import sys
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")


def load_loss_data(loss_file_path, model_name, gradient_accumulation_steps=16):

    df = pd.read_csv(loss_file_path)
    print(f"\nLoading {model_name} loss from {loss_file_path}")
    print(f"  Available columns: {df.columns.tolist()}")
    
    loss_col = None
    for col in ['total_loss', 'loss', 'train_loss', 'Loss', 'loss_skl']:
        if col in df.columns:
            loss_col = col
            break
    
    if loss_col is None:
        print(f"  Warning: No loss column found, using first numeric column")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            loss_col = numeric_cols[0]
        else:
            raise ValueError(f"No numeric columns found in {loss_file_path}")
    
    if 'total_loss' in df.columns and model_name == "DistiLLM-2":
        df['loss_scaled'] = df['total_loss'] * gradient_accumulation_steps
        loss_col = 'loss_scaled'
        print(f"  Scaling DistiLLM-2 loss by factor {gradient_accumulation_steps} for comparison")
    
    step_col = None
    
    if 'batch_step' in df.columns and 'epoch' in df.columns and model_name == "SoftKD":
        steps_per_epoch = df[df['epoch'] == 0]['batch_step'].max() + 1
        df['global_step'] = df['epoch'] * steps_per_epoch + df['batch_step']
        step_col = 'global_step'
        print(f"  Created global_step for {model_name} (steps_per_epoch={steps_per_epoch})")
    elif 'step' in df.columns:
        step_col = 'step'
    elif 'batch_step' in df.columns:
        step_col = 'batch_step'
    else:
        step_col = 'index'
        df[step_col] = range(len(df))
    
    print(f"  Using loss column: '{loss_col}', step column: '{step_col}'")
    print(f"  Loss range: {df[loss_col].min():.4f} - {df[loss_col].max():.4f}")
    print(f"  Step range: {df[step_col].min()} - {df[step_col].max()}")
    print(f"  Number of records: {len(df)}")
    
    return df, loss_col, step_col


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GRADIENT_ACCUMULATION_STEPS = 16
    
    softkd_model_dir = os.path.join(BASE_DIR, "outputs/softkd_model_5k_v2")
    distillm2_model_dir = os.path.join(BASE_DIR, "outputs/distillm2_model_5k_v3")
    eval_dir = os.path.join(BASE_DIR, "outputs/evaluation")
    
    os.makedirs(eval_dir, exist_ok=True)
    
    softkd_eval_file = os.path.join(eval_dir, "softkd_model_5k_v2_summary.csv")
    distillm2_eval_file = os.path.join(eval_dir, "distillm2_model_5k_v3_summary.csv")
    
    missing_files = []
    if not os.path.exists(softkd_eval_file):
        missing_files.append(softkd_eval_file)
    if not os.path.exists(distillm2_eval_file):
        missing_files.append(distillm2_eval_file)
    
    if missing_files:
        print(f"Error: Missing evaluation files:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    print("\nGenerating plots...")
    
    softkd_loss_df, softkd_loss_col, softkd_step_col = load_loss_data(
        os.path.join(softkd_model_dir, "training_loss.csv"), 
        "SoftKD", 
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    distillm2_loss_df, distillm2_loss_col, distillm2_step_col = load_loss_data(
        os.path.join(distillm2_model_dir, "training_loss.csv"), 
        "DistiLLM-2", 
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    
    softkd_metrics = pd.read_csv(softkd_eval_file)
    distillm2_metrics = pd.read_csv(distillm2_eval_file)
    

    print("\nPlot 1: SoftKD Training Loss ---")
    
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    
    window_softkd = max(5, min(50, len(softkd_loss_df) // 10))
    softkd_loss_smooth = softkd_loss_df[softkd_loss_col].rolling(window_softkd, min_periods=1).mean()
    
    ax1.plot(softkd_loss_df[softkd_step_col], softkd_loss_smooth, 
             linewidth=1.5, color='steelblue')
    
    step_every = max(1, len(softkd_loss_df) // 100)
    ax1.scatter(softkd_loss_df[softkd_step_col][::step_every], 
                softkd_loss_df[softkd_loss_col][::step_every],
                s=10, alpha=0.3, color='steelblue', label='Original data (sampled)')
    
    ax1.set_xlabel('Global Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('SoftKD: Training Loss Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    final_softkd_loss = softkd_loss_df[softkd_loss_col].iloc[-1]
    ax1.annotate(f'Final Loss: {final_softkd_loss:.4f}', 
                 xy=(softkd_loss_df[softkd_step_col].iloc[-1], final_softkd_loss),
                 xytext=(-100, -50), textcoords='offset points', 
                 fontsize=10, color='steelblue', fontweight='normal')
    
    plt.tight_layout()
    softkd_loss_file = os.path.join(eval_dir, "loss_curve_softkd_5k_v2.png")
    plt.savefig(softkd_loss_file, dpi=150)
    plt.close()
    print(f"  SoftKD loss curve saved to {softkd_loss_file}")
    
    print("\nPlot 2: DistiLLM-2 Training Loss ---")
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    
    window_distillm2 = max(3, min(20, len(distillm2_loss_df) // 5))
    distillm2_loss_smooth = distillm2_loss_df[distillm2_loss_col].rolling(window_distillm2, min_periods=1).mean()
    
    ax2.plot(distillm2_loss_df[distillm2_step_col], distillm2_loss_smooth, 
             linewidth=1.5, color='darkorange')

    ax2.set_xlabel('Global Training Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('DistiLLM-2: Training Loss Curve', fontsize=14)
    ax2.grid(True, alpha=0.3)

    final_distillm2_loss = distillm2_loss_df[distillm2_loss_col].iloc[-1]
    ax2.annotate(f'Final Loss: {final_distillm2_loss:.4f}', 
                 xy=(distillm2_loss_df[distillm2_step_col].iloc[-1], final_distillm2_loss),
                 xytext=(-100, -50), textcoords='offset points', 
                 fontsize=10, color='darkorange', fontweight='normal')

    plt.tight_layout()
    distillm2_loss_file = os.path.join(eval_dir, "loss_curve_distillm2_5k_v3.png")
    plt.savefig(distillm2_loss_file, dpi=150)
    plt.close()
    print(f"  DistiLLM-2 loss curve saved to {distillm2_loss_file}")

    print("\nPlot 3: Metrics Comparison ---")
    
    metrics = ['rouge1_mean', 'rouge2_mean', 'rougeL_mean', 'bert_f1_mean']
    labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']
    
    softkd_vals = [softkd_metrics[m].values[0] for m in metrics]
    distillm2_vals = [distillm2_metrics[m].values[0] for m in metrics]
    
    print(f"  SoftKD metrics: {softkd_vals}")
    print(f"  DistiLLM-2 metrics: {distillm2_vals}")
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars1 = ax3.bar(x - width/2, softkd_vals, width, label='SoftKD', color='steelblue')
    bars2 = ax3.bar(x + width/2, distillm2_vals, width, label='DistiLLM-2', color='darkorange')
    
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Comparison of Distillation Methods', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11)
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, softkd_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, distillm2_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    max_val = max(max(softkd_vals), max(distillm2_vals))
    ax3.set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    metrics_file = os.path.join(eval_dir, "metrics_comparison_5k_v3.png")
    plt.savefig(metrics_file, dpi=150)
    plt.close()
    print(f"  Metrics comparison saved to {metrics_file}")
    
    print("\n" + "="*60)
    print("Metrics")
    print("="*60)
    print(f"\n{'Metric':<15} {'SoftKD':<12} {'DistiLLM-2':<12}")
    print("-" * 40)
    for label, s_val, d_val in zip(labels, softkd_vals, distillm2_vals):
        print(f"{label:<15} {s_val:<12.4f} {d_val:<12.4f}")
    
    baseline_summary = os.path.join(eval_dir, 'baseline_summary.csv')
    if os.path.exists(baseline_summary):
        baseline_df = pd.read_csv(baseline_summary)
        print(f"\n{'='*60}")
        print("Improvement over baseline")
        print("="*60)
        print(f"\n{'Metric':<15} {'Baseline':<12} {'SoftKD Δ':<12} {'DistiLLM-2 Δ':<12}")
        print("-" * 55)
        for metric, label in zip(metrics, labels):
            baseline_val = baseline_df[metric].values[0]
            softkd_change = ((softkd_vals[metrics.index(metric)] - baseline_val) / baseline_val) * 100
            distillm2_change = ((distillm2_vals[metrics.index(metric)] - baseline_val) / baseline_val) * 100
            print(f"{label:<15} {baseline_val:<12.4f} {softkd_change:+.1f}%{'':<7} {distillm2_change:+.1f}%")
    
    if 'valid_samples' in distillm2_loss_df.columns:
        avg_valid_samples = distillm2_loss_df['valid_samples'].mean()
        print(f"\n{'='*60}")
        print("DistiLLM-2 Training Statistics")
        print("="*60)
        print(f"  Average valid samples per batch: {avg_valid_samples:.1f}")
        print(f"  Total training steps recorded: {len(distillm2_loss_df)}")
        print(f"  (Loss logged every {GRADIENT_ACCUMULATION_STEPS} steps due to gradient accumulation)")
    

    print(f"  SoftKD loss curve: {softkd_loss_file}")
    print(f"  DistiLLM-2 loss curve: {distillm2_loss_file}")
    print(f"  Metrics comparison: {metrics_file}")
    print(f"  All files saved to {eval_dir}")


if __name__ == "__main__":
    main()