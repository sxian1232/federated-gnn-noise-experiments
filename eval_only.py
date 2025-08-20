import argparse, os, json
import numpy as np
import torch
from model import Model
from main import data_loader, val_model, my_load_model, display_result, dev, graph_args, batch_size_val

def eval_one(model_path: str, val_data: str):
    model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
    model.to(dev)
    my_load_model(model, model_path)
    loader = data_loader(val_data, pra_batch_size=batch_size_val,
                        pra_shuffle=False, pra_drop_last=False, train_val_test='val')
    sums, nums = val_model(model, loader)
    ws = display_result([sums, nums], pra_pref=f"EvalOnly {os.path.basename(model_path)}")
    return {
        "ws_per_horizon": [float(x) for x in ws],
        "ws_sum": float(np.sum(ws))  # <- 更稳
    }

def main():
    ap = argparse.ArgumentParser(description="Evaluate existing .pt models on the common validation set.")
    ap.add_argument("--models", nargs="+", required=True, help="List of model .pt paths")
    ap.add_argument("--val_data", required=True, help="Validation dataset (processed_data/test_data.pkl)")
    ap.add_argument("--out_json", required=True, help="Where to save the eval results JSON")
    args = ap.parse_args()

    results = {}
    for p in args.models:
        try:
            results[p] = eval_one(p, args.val_data)
        except Exception as e:
            results[p] = {"error": str(e)}  # <- 单个失败也不中断

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] eval results -> {args.out_json}")

if __name__ == "__main__":
    main()
