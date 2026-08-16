import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util import tensor_util

def parse_tensor_value(tensor_proto):
    try:
        val = tensor_util.make_ndarray(tensor_proto)
        if hasattr(val, "tolist"):
            val_list = val.tolist()
            if isinstance(val_list, bytes):
                return val_list.decode("utf-8", errors="replace")
            elif isinstance(val_list, list) and len(val_list) > 0 and isinstance(val_list[0], bytes):
                return [x.decode("utf-8", errors="replace") for x in val_list]
            elif val.ndim == 0:
                return float(val)
            elif val.size == 1:
                return float(val.item())
            return val_list
        return str(val)
    except Exception as e:
        return str(tensor_proto)

def extract_all():
    log_dir = Path(".")
    event_files = sorted(list(log_dir.glob("events.out.tfevents.*")))
    
    output_dir = log_dir / "extracted_data"
    output_dir.mkdir(exist_ok=True)
    
    all_metrics = []
    run_summaries = []
    all_runs_metadata = {}
    
    print(f"Found {len(event_files)} TensorBoard event files.\n")
    print("=" * 80)
    
    for idx, ef in enumerate(event_files):
        filename = ef.name
        print(f"[{idx+1}/{len(event_files)}] Processing: {filename}")
        
        # Load ALL events with size_guidance=0 (no downsampling)
        ea = EventAccumulator(
            str(ef),
            size_guidance={
                "scalars": 0,
                "tensors": 0,
                "histograms": 0,
                "images": 0,
                "audio": 0,
                "distributions": 0,
            }
        )
        ea.Reload()
        tags = ea.Tags()
        
        run_records = []
        run_meta = {
            "file": filename,
            "args": {},
            "model_config": {}
        }
        
        # Extract Scalar tags
        for tag in tags.get("scalars", []):
            for e in ea.Scalars(tag):
                row = {
                    "run_file": filename,
                    "tag": tag,
                    "step": e.step,
                    "wall_time": e.wall_time,
                    "value": float(e.value),
                    "type": "scalar"
                }
                all_metrics.append(row)
                run_records.append(row)
                
        # Extract Tensor tags (includes args/text_summary, model_config/text_summary, or scalar tensors)
        for tag in tags.get("tensors", []):
            for e in ea.Tensors(tag):
                parsed_val = parse_tensor_value(e.tensor_proto)
                
                # Check if it's a JSON string / metadata text
                if tag in ("args/text_summary", "model_config/text_summary") or isinstance(parsed_val, (str, list)):
                    text_str = parsed_val[0] if isinstance(parsed_val, list) else parsed_val
                    if isinstance(text_str, str):
                        try:
                            json_obj = json.loads(text_str)
                            if "args" in tag:
                                run_meta["args"] = json_obj
                            elif "model_config" in tag:
                                run_meta["model_config"] = json_obj
                            else:
                                run_meta[tag] = json_obj
                        except Exception:
                            run_meta[tag] = text_str
                else:
                    # Numeric tensor scalar
                    row = {
                        "run_file": filename,
                        "tag": tag,
                        "step": e.step,
                        "wall_time": e.wall_time,
                        "value": parsed_val,
                        "type": "tensor"
                    }
                    all_metrics.append(row)
                    run_records.append(row)
                    
        # Extract Texts if any
        for tag in tags.get("texts", []):
            for e in ea.Texts(tag):
                run_meta[tag] = e.text
                
        # Save individual run CSV
        if run_records:
            df_run = pd.DataFrame(run_records)
            run_csv_path = output_dir / f"{filename}.csv"
            df_run.to_csv(run_csv_path, index=False)
            print(f"  -> Extracted {len(run_records)} metric steps to {run_csv_path.name}")
            
        # Save individual run metadata JSON
        run_json_path = output_dir / f"{filename}_metadata.json"
        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, ensure_ascii=False)
            
        all_runs_metadata[filename] = run_meta
        
        # Build summary entry for overview table
        args = run_meta.get("args", {})
        summary_entry = {
            "run_file": filename,
            "output_dir": args.get("output_dir", "N/A"),
            "learning_rate": args.get("learning_rate", "N/A"),
            "num_train_epochs": args.get("num_train_epochs", "N/A"),
            "batch_size": args.get("per_device_train_batch_size", "N/A"),
            "lr_scheduler": args.get("lr_scheduler_type", "N/A"),
            "total_metric_records": len(run_records)
        }
        
        # Compute min/max/last of key metrics for this run
        df_r = pd.DataFrame(run_records)
        if not df_r.empty:
            for m_tag in ["train/loss", "eval/loss", "eval/mean_token_accuracy"]:
                sub = df_r[df_r["tag"] == m_tag]
                if not sub.empty:
                    last_val = sub.sort_values("step").iloc[-1]["value"]
                    min_val = sub["value"].min()
                    summary_entry[f"{m_tag}_last"] = round(last_val, 4)
                    summary_entry[f"{m_tag}_min"] = round(min_val, 4)
        run_summaries.append(summary_entry)
        
    print("=" * 80)
    
    # Save combined metrics CSV
    df_all = pd.DataFrame(all_metrics)
    combined_csv = output_dir / "all_runs_metrics.csv"
    df_all.to_csv(combined_csv, index=False)
    print(f"Saved ALL metrics ({len(df_all)} records) to: {combined_csv}")
    
    # Save combined metadata JSON
    combined_json = output_dir / "all_runs_metadata_and_configs.json"
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(all_runs_metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved ALL configs & hyperparameters to: {combined_json}")
    
    # Save overview table CSV & JSON
    df_summary = pd.DataFrame(run_summaries)
    summary_csv = output_dir / "finetuning_runs_overview.csv"
    df_summary.to_csv(summary_csv, index=False)
    
    summary_json = output_dir / "finetuning_runs_overview.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(run_summaries, f, indent=2)
    print(f"Saved finetuning runs overview to: {summary_csv}")
    
    print("\n--- FINETUNING RUNS OVERVIEW ---")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    extract_all()
