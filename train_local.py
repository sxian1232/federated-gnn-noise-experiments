import argparse
import os
import json
import time
import torch
import numpy as np
from model import Model
from main import run_trainval

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_init_weights_if_any(model, init_model_path):
    if not init_model_path:
        return
    if not os.path.exists(init_model_path):
        raise FileNotFoundError(init_model_path)
    ckpt = torch.load(init_model_path, map_location="cpu")
    state = ckpt["xin_graph_seq2seq_model"] if isinstance(ckpt, dict) and "xin_graph_seq2seq_model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data',  type=str, required=True)
    parser.add_argument('--out_model',  type=str, required=True)
    parser.add_argument('--out_edge',   type=str, required=True)
    parser.add_argument('--init_model', type=str, default=None)
    # NEW: optional metrics json
    parser.add_argument('--metrics_out', type=str, default=None)
    args = parser.parse_args()

    # 保持原有行为：只确保模型和edge保存路径存在
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_edge),  exist_ok=True)

    device = get_device()
    graph_args = {'max_hop': 2, 'num_node': 120}
    model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True).to(device)

    # 可选 warm start
    load_init_weights_if_any(model, args.init_model)

    # 训练+验证（与原 run_trainval 签名一致）
    ret = run_trainval(model, pra_traindata_path=args.train_data, pra_testdata_path=args.test_data)

    # 保存 edge importance
    with torch.no_grad():
        edge_importance = [param.detach().cpu().numpy() for param in model.edge_importance]
        np.save(args.out_edge, edge_importance)
        print(f"Saved edge importance to {args.out_edge}")

    # 保存模型（CPU安全）
    to_save = {"xin_graph_seq2seq_model": {k: v.cpu() for k, v in model.state_dict().items()}}
    torch.save(to_save, args.out_model)
    print(f"Saved model to {args.out_model}")

    # NEW: 保存metrics（可选；目录仅在需要时创建）
    if args.metrics_out:
        d = os.path.dirname(args.metrics_out)
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device.type,
            "train_data": args.train_data,
            "test_data": args.test_data,
            "init_model": args.init_model,
            "out_model": args.out_model,
            "out_edge": args.out_edge,
        }
        try:
            payload["run_trainval_return"] = ret
        except Exception as e:
            payload["run_trainval_return"] = f"<unserializable: {e}>"
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {args.metrics_out}")

if __name__ == '__main__':
    main()
