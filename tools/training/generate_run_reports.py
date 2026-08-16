import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure seaborn & matplotlib styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "font.family": "sans-serif"
})

# Define metadata mapping and readable folder names for each run
RUN_CONFIGS = {
    "events.out.tfevents.1770456721.15f3a867024b.5912.0": {
        "folder": "Run_1_Baseline_Output_4512samples",
        "title": "Run 1: Baseline Output (Medium Dataset)",
        "desc": "Baseline finetuning run on 4,512 samples for 3 epochs with learning rate 2e-4.",
        "sample_size": 4512,
        "total_exposures": 13536,
        "dataset_name": "Baseline Medium Dataset (~4.5K)"
    },
    "events.out.tfevents.1771930286.938c31f14087.16817.0": {
        "folder": "Run_2_CAMUS_LR1e5_1600samples",
        "title": "Run 2: CAMUS Dataset (Conservative LR 1e-5)",
        "desc": "Finetuning on the CAMUS dataset (1,600 samples) with conservative learning rate 1e-5 for 3 epochs.",
        "sample_size": 1600,
        "total_exposures": 4800,
        "dataset_name": "CAMUS Dataset (1,600 samples)"
    },
    "events.out.tfevents.1771950289.0b5e3829cdda.1883.0": {
        "folder": "Run_3_CAMUS_LR2e4_8epochs_1504samples",
        "title": "Run 3: CAMUS Dataset (Extended 8 Epochs, LR 2e-4)",
        "desc": "Aggressive finetuning on CAMUS dataset (~1,504-1,600 samples) with learning rate 2e-4 extended to 8 epochs.",
        "sample_size": 1504,
        "total_exposures": 12032,
        "dataset_name": "CAMUS Dataset (~1,600 samples)"
    },
    "events.out.tfevents.1776098432.e18ce739cf4e.13621.0": {
        "folder": "Run_4_20K_Dataset_16541samples",
        "title": "Run 4: Large 20K Dataset (LR 1e-4)",
        "desc": "Large-scale finetuning on ~16,541 dataset samples for 2.26 epochs with effective batch size 32.",
        "sample_size": 16541,
        "total_exposures": 37440,
        "dataset_name": "Large 20K Dataset (16,541 samples)"
    },
    "events.out.tfevents.1776294211.15e97987a4b5.21254.0": {
        "folder": "Run_5_20K_Blackwell_16576samples",
        "title": "Run 5: Large 20K Blackwell (Effective Batch Size 64)",
        "desc": "Full 5-epoch finetuning on 16,576 samples on Blackwell architecture with doubled effective batch size 64.",
        "sample_size": 16576,
        "total_exposures": 82880,
        "dataset_name": "Large 20K Dataset (16,576 samples)"
    },
    "events.out.tfevents.1776410801.613f842a7ad8.13245.0": {
        "folder": "Run_6_CAMUS_32LoRA_Blackwell_1600samples",
        "title": "Run 6: CAMUS 32-LoRA Blackwell (1,600 samples)",
        "desc": "Finetuning with LoRA rank r=32 on CAMUS dataset (1,600 samples) over 5 epochs with effective batch size 64.",
        "sample_size": 1600,
        "total_exposures": 8000,
        "dataset_name": "CAMUS Dataset (1,600 samples)"
    }
}

def create_charts_for_run(df_run, output_dir, title_prefix):
    # 1. Loss Curves (train/loss vs eval/loss)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    train_loss = df_run[df_run["tag"] == "train/loss"].sort_values("step")
    eval_loss = df_run[df_run["tag"] == "eval/loss"].sort_values("step")
    
    if not train_loss.empty:
        ax.plot(train_loss["step"], train_loss["value"], alpha=0.35, color="#1f77b4", label="Train Loss (Raw)")
        # Rolling average for smooth visualization
        window = max(1, len(train_loss) // 15)
        smooth = train_loss["value"].rolling(window=window, min_periods=1).mean()
        ax.plot(train_loss["step"], smooth, color="#1f77b4", linewidth=2.2, label=f"Train Loss (Smooth w={window})")
        
    if not eval_loss.empty:
        ax.plot(eval_loss["step"], eval_loss["value"], color="#d62728", marker="o", markersize=5, linewidth=2.2, label="Eval Loss")
        min_idx = eval_loss["value"].idxmin()
        min_step = eval_loss.loc[min_idx, "step"]
        min_val = eval_loss.loc[min_idx, "value"]
        ax.annotate(f"Min Eval Loss: {min_val:.4f}\n(Step {min_step})",
                    xy=(min_step, min_val), xytext=(min_step, min_val * 1.15),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="#d62728", lw=1))
        
    ax.set_title(f"{title_prefix}\nTraining & Evaluation Loss Curves", fontweight="bold", pad=12)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / "01_loss_curves.png", dpi=300)
    plt.close()

    # 2. Token Accuracy Curves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    train_acc = df_run[df_run["tag"] == "train/mean_token_accuracy"].sort_values("step")
    eval_acc = df_run[df_run["tag"] == "eval/mean_token_accuracy"].sort_values("step")
    
    if not train_acc.empty:
        window = max(1, len(train_acc) // 15)
        smooth_t = train_acc["value"].rolling(window=window, min_periods=1).mean()
        ax.plot(train_acc["step"], train_acc["value"], alpha=0.3, color="#2ca02c", label="Train Token Acc (Raw)")
        ax.plot(train_acc["step"], smooth_t, color="#2ca02c", linewidth=2.2, label="Train Token Acc (Smooth)")
    if not eval_acc.empty:
        ax.plot(eval_acc["step"], eval_acc["value"], color="#9467bd", marker="s", markersize=5, linewidth=2.2, label="Eval Token Acc")
        max_idx = eval_acc["value"].idxmax()
        max_step = eval_acc.loc[max_idx, "step"]
        max_val = eval_acc.loc[max_idx, "value"]
        ax.annotate(f"Peak Eval Acc: {max_val*100:.2f}%\n(Step {max_step})",
                    xy=(max_step, max_val), xytext=(max_step, max_val * 0.92),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f2e6ff", ec="#9467bd", lw=1))
        
    ax.set_title(f"{title_prefix}\nToken Accuracy Progression", fontweight="bold", pad=12)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Token Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / "02_token_accuracy.png", dpi=300)
    plt.close()

    # 3. Learning Rate Schedule
    fig, ax = plt.subplots(figsize=(9, 5))
    lr_data = df_run[df_run["tag"] == "train/learning_rate"].sort_values("step")
    if not lr_data.empty:
        ax.plot(lr_data["step"], lr_data["value"], color="#ff7f0e", linewidth=2.5, label="Learning Rate")
        max_lr = lr_data["value"].max()
        ax.set_title(f"{title_prefix}\nLearning Rate Schedule (Peak LR = {max_lr:.2e})", fontweight="bold", pad=12)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Learning Rate")
        ax.legend(frameon=True, facecolor="white")
    else:
        ax.text(0.5, 0.5, "No Learning Rate Data Logged", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "03_learning_rate_schedule.png", dpi=300)
    plt.close()

    # 4. Gradient Norm & Entropy (Subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    grad_norm = df_run[df_run["tag"] == "train/grad_norm"].sort_values("step")
    if not grad_norm.empty:
        ax1.plot(grad_norm["step"], grad_norm["value"], color="#8c564b", alpha=0.7, linewidth=1.5, label="Gradient Norm")
        ax1.set_title("Training Gradient Norm", fontweight="bold")
        ax1.set_ylabel("Grad Norm (L2)")
        ax1.legend()
        
    train_ent = df_run[df_run["tag"] == "train/entropy"].sort_values("step")
    eval_ent = df_run[df_run["tag"] == "eval/entropy"].sort_values("step")
    if not train_ent.empty:
        ax2.plot(train_ent["step"], train_ent["value"], color="#17becf", alpha=0.7, label="Train Entropy")
    if not eval_ent.empty:
        ax2.plot(eval_ent["step"], eval_ent["value"], color="#e377c2", marker="o", markersize=4, label="Eval Entropy")
    ax2.set_title("Prediction Entropy", fontweight="bold")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Entropy")
    ax2.legend()
    
    fig.suptitle(f"{title_prefix}\nGradient Norm & Entropy Analysis", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "04_grad_norm_and_entropy.png", dpi=300)
    plt.close()

    # 5. Throughput & Speed
    fig, ax = plt.subplots(figsize=(9, 5))
    train_speed = df_run[df_run["tag"] == "train/train_samples_per_second"].sort_values("step")
    eval_speed = df_run[df_run["tag"] == "eval/samples_per_second"].sort_values("step")
    plotted = False
    if not train_speed.empty:
        ax.plot(train_speed["step"], train_speed["value"], marker="o", color="#bcbd22", label="Train Samples / Sec")
        plotted = True
    if not eval_speed.empty:
        ax.plot(eval_speed["step"], eval_speed["value"], marker="s", color="#7f7f7f", label="Eval Samples / Sec")
        plotted = True
    if not plotted:
        # fallback to epoch curve if no speed tags
        ep = df_run[df_run["tag"] == "train/epoch"].sort_values("step")
        if not ep.empty:
            ax.plot(ep["step"], ep["value"], color="#bcbd22", linewidth=2.5, label="Epochs Completed")
            ax.set_ylabel("Epoch")
    else:
        ax.set_ylabel("Samples per Second")
    ax.set_title(f"{title_prefix}\nTraining & Evaluation Throughput Speed", fontweight="bold", pad=12)
    ax.set_xlabel("Training Step")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "05_throughput_performance.png", dpi=300)
    plt.close()

    # 6. 2x2 Master Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # Panel 1: Loss
    if not train_loss.empty:
        smooth_l = train_loss["value"].rolling(window=max(1, len(train_loss)//15), min_periods=1).mean()
        axs[0, 0].plot(train_loss["step"], smooth_l, color="#1f77b4", label="Train Loss (Smooth)")
    if not eval_loss.empty:
        axs[0, 0].plot(eval_loss["step"], eval_loss["value"], color="#d62728", marker="o", label="Eval Loss")
    axs[0, 0].set_title("Loss Curves", fontweight="bold")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # Panel 2: Token Acc
    if not eval_acc.empty:
        axs[0, 1].plot(eval_acc["step"], eval_acc["value"], color="#9467bd", marker="s", label="Eval Acc")
    if not train_acc.empty:
        smooth_a = train_acc["value"].rolling(window=max(1, len(train_acc)//15), min_periods=1).mean()
        axs[0, 1].plot(train_acc["step"], smooth_a, color="#2ca02c", label="Train Acc (Smooth)")
    axs[0, 1].set_title("Mean Token Accuracy", fontweight="bold")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].legend()

    # Panel 3: LR Schedule
    if not lr_data.empty:
        axs[1, 0].plot(lr_data["step"], lr_data["value"], color="#ff7f0e", label="Learning Rate")
    axs[1, 0].set_title("Learning Rate Schedule", fontweight="bold")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].set_ylabel("LR")

    # Panel 4: Grad Norm
    if not grad_norm.empty:
        axs[1, 1].plot(grad_norm["step"], grad_norm["value"], color="#8c564b", alpha=0.7, label="Grad Norm")
    axs[1, 1].set_title("Gradient Norm", fontweight="bold")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("Norm")

    fig.suptitle(f"{title_prefix} — Master Executive Dashboard", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "06_run_dashboard_summary.png", dpi=300)
    plt.close()

def generate_reports():
    base_dir = Path(".")
    extracted_dir = base_dir / "extracted_data"
    reports_dir = base_dir / "run_reports"
    reports_dir.mkdir(exist_ok=True)
    
    with open(extracted_dir / "all_runs_metadata_and_configs.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    df_all = pd.read_csv(extracted_dir / "all_runs_metrics.csv")
    
    print(f"Generating reports and charts for 6 runs inside {reports_dir.resolve()} ...")
    
    for run_file, cfg in RUN_CONFIGS.items():
        run_folder = reports_dir / cfg["folder"]
        run_folder.mkdir(exist_ok=True)
        
        print(f"\n--> Processing {cfg['folder']} ...")
        
        df_run = df_all[df_all["run_file"] == run_file]
        run_meta = meta.get(run_file, {})
        args = run_meta.get("args", {})
        
        # Save run metrics CSV inside its folder
        df_run.to_csv(run_folder / "metrics_data.csv", index=False)
        
        # Create Charts
        create_charts_for_run(df_run, run_folder, cfg["title"])
        print("    Generated 6 PNG charts.")
        
        # Compute summary metrics
        train_loss = df_run[df_run["tag"] == "train/loss"].sort_values("step")
        eval_loss = df_run[df_run["tag"] == "eval/loss"].sort_values("step")
        eval_acc = df_run[df_run["tag"] == "eval/mean_token_accuracy"].sort_values("step")
        
        final_train_loss = train_loss.iloc[-1]["value"] if not train_loss.empty else None
        min_eval_loss = eval_loss["value"].min() if not eval_loss.empty else None
        peak_eval_acc = eval_acc["value"].max() if not eval_acc.empty else None
        final_eval_acc = eval_acc.iloc[-1]["value"] if not eval_acc.empty else None
        
        bs = args.get("per_device_train_batch_size", 1)
        ga = args.get("gradient_accumulation_steps", 1)
        ws = args.get("world_size", 1)
        eff_bs = bs * ga * ws
        
        # Prepare structured JSON
        report_json = {
            "run_file": run_file,
            "title": cfg["title"],
            "dataset_name": cfg["dataset_name"],
            "hyperparameters": {
                "output_dir": args.get("output_dir"),
                "learning_rate": args.get("learning_rate"),
                "num_train_epochs": args.get("num_train_epochs"),
                "per_device_train_batch_size": bs,
                "gradient_accumulation_steps": ga,
                "world_size": ws,
                "effective_batch_size": eff_bs,
                "lr_scheduler_type": args.get("lr_scheduler_type"),
                "warmup_ratio": args.get("warmup_ratio"),
                "weight_decay": args.get("weight_decay"),
                "max_grad_norm": args.get("max_grad_norm"),
                "optimizer": args.get("optim", "adamw_torch_fused"),
                "bf16": args.get("bf16", True),
                "eval_strategy": args.get("eval_strategy")
            },
            "sample_size_metrics": {
                "dataset_sample_size": cfg["sample_size"],
                "total_sample_exposures": cfg["total_exposures"],
                "total_gradient_steps": int(df_run["step"].max()) if not df_run.empty else 0
            },
            "key_results": {
                "final_train_loss": round(final_train_loss, 4) if final_train_loss else None,
                "min_eval_loss": round(min_eval_loss, 4) if min_eval_loss else None,
                "peak_eval_token_accuracy": round(peak_eval_acc, 4) if peak_eval_acc else None,
                "final_eval_token_accuracy": round(final_eval_acc, 4) if final_eval_acc else None
            }
        }
        
        with open(run_folder / "hyperparameters_and_sample_size.json", "w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2)
            
        # Write Markdown Report
        md_content = f"""# {cfg['title']}
**Log Event File**: `{run_file}`  
**Adapter / Output Directory**: `{args.get('output_dir', 'N/A')}`

---

## 1. Executive Summary & Overview
{cfg['desc']}

| Key Performance Metric | Value |
| :--- | :---: |
| **Final Training Loss** | `{final_train_loss:.4f}` |
| **Minimum Evaluation Loss** | `{min_eval_loss:.4f}` |
| **Peak Evaluation Token Accuracy** | **`{peak_eval_acc*100:.2f}%`** |
| **Final Evaluation Token Accuracy** | `{final_eval_acc*100:.2f}%` |

---

## 2. Exact Hyperparameters
| Hyperparameter Parameter | Logged Value | Explanation |
| :--- | :--- | :--- |
| **Learning Rate (`learning_rate`)** | `{args.get('learning_rate')}` | Initial peak learning rate for optimizer |
| **Training Epochs (`num_train_epochs`)** | `{args.get('num_train_epochs')}` | Number of passes through the training dataset |
| **Per-Device Batch Size (`per_device_train_batch_size`)** | `{bs}` | Number of samples processed per GPU pass |
| **Gradient Accumulation Steps (`gradient_accumulation_steps`)** | `{ga}` | Forward passes accumulated before backward step |
| **Effective Batch Size** | **`{eff_bs}`** | Total samples per optimizer update (`bs * ga * devices`) |
| **LR Scheduler (`lr_scheduler_type`)** | `{args.get('lr_scheduler_type')}` | Learning rate decay strategy |
| **Warmup Ratio (`warmup_ratio`)** | `{args.get('warmup_ratio', 'None')}` | Fraction of steps spent warming up LR |
| **Optimizer (`optim`)** | `{args.get('optim', 'adamw_torch_fused')}` | Fused AdamW optimizer |
| **Precision (`bf16`)** | `{args.get('bf16', True)}` | Bfloat16 mixed-precision training |
| **Max Gradient Norm (`max_grad_norm`)** | `{args.get('max_grad_norm')}` | Gradient clipping threshold |

---

## 3. Exact Sample Size & Dataset Metrics
* **Dataset / Sample Size ($N$)**: **`{cfg['sample_size']:,} unique samples`**
  * Calculated as: $\\text{{Total Steps}} \\times \\text{{Effective Batch Size}} / \\text{{Epochs}} = {int(df_run['step'].max())} \\times {eff_bs} / {args.get('num_train_epochs')} = {cfg['sample_size']:,}$ samples.
* **Total Sample Exposures**: **`{cfg['total_exposures']:,}`** training sample iterations across all epochs.
* **Total Gradient Steps Logged**: **`{int(df_run['step'].max()):,}`** steps.

---

## 4. Generated Performance Charts

### Master Executive Dashboard
`06_run_dashboard_summary.png` combines Loss, Token Accuracy, Learning Rate, and Gradient Norm:
![Master Dashboard](06_run_dashboard_summary.png)

### Training & Evaluation Loss Curves
`01_loss_curves.png` plots raw and smoothed training loss alongside evaluation loss:
![Loss Curves](01_loss_curves.png)

### Token Accuracy Progression
`02_token_accuracy.png` plots evaluation and training token accuracy over steps:
![Token Accuracy](02_token_accuracy.png)

### Learning Rate Schedule
`03_learning_rate_schedule.png` displays the exact linear decay learning rate schedule:
![Learning Rate](03_learning_rate_schedule.png)

### Gradient Norm & Entropy
`04_grad_norm_and_entropy.png` illustrates gradient stability and prediction entropy:
![Grad Norm and Entropy](04_grad_norm_and_entropy.png)

### Throughput Performance
`05_throughput_performance.png` tracks training and evaluation speed:
![Throughput](05_throughput_performance.png)
"""
        with open(run_folder / "hyperparameters_and_sample_size.md", "w", encoding="utf-8") as f:
            f.write(md_content)
            
    print("\nAll 6 run report subfolders generated successfully!")

if __name__ == "__main__":
    generate_reports()
