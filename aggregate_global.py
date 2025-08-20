# aggregate_global.py
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

# -----------------------------
# I/O utils
# -----------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_model_state(p):
    ckpt = torch.load(p, map_location="cpu")
    if isinstance(ckpt, dict) and "xin_graph_seq2seq_model" in ckpt:
        return ckpt["xin_graph_seq2seq_model"]
    elif isinstance(ckpt, dict):
        # 直接是 state_dict
        return ckpt
    else:
        raise ValueError(f"Unexpected model file format: {p}")

def save_model_state(state, out_path):
    ensure_dir(Path(out_path).parent)
    torch.save({"xin_graph_seq2seq_model": state}, out_path)

def load_edge_vector(npy_path):
    """
    读取 edge_importance.npy 并拉平成 1D 向量（会自动平均 layer/channel）。
    兼容形状：
      [L,C,V,V] / [L,V,V] / [C,V,V] / [V,V] / [V*V]
    """
    arr = np.load(npy_path, allow_pickle=True)

    # 兼容 list/object 包装
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.array(list(arr), dtype=object)
        if arr.dtype == object and len(arr) == 1:
            arr = arr[0]

    # 这里改成 np.asarray(...)（去掉 copy=False）
    a = np.asarray(arr, dtype=np.float64)

    # 已是扁平向量，直接返回
    if a.ndim == 1:
        return a.copy()

    if a.ndim == 4:
        a = a.mean(axis=(0, 1))
    elif a.ndim == 3:
        a = a.mean(axis=0)
    elif a.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected edge importance shape: {a.shape} in {npy_path}")

    return a.ravel()

# -----------------------------
# Blocked MMD (内存友好)
# -----------------------------
def _rbf_block(a, b, gamma):
    # a:[p,1], b:[q,1] -> RBF kernel block [p,q]
    return np.exp(-gamma * (a - b.T) ** 2)

def compute_mmd_rbf(x, y, gamma=0.01, block=65536):
    """
    分块估计 MMD^2：
      MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2 E[k(X,Y)]
    x, y: 1D numpy arrays
    """
    x = x.reshape(-1, 1).astype(np.float64, copy=False)
    y = y.reshape(-1, 1).astype(np.float64, copy=False)
    nx, ny = x.shape[0], y.shape[0]

    # E[k(X,X')]
    sx = 0.0; cntx = 0
    for i in range(0, nx, block):
        xi = x[i:i+block]
        for j in range(0, nx, block):
            xj = x[j:j+block]
            k = _rbf_block(xi, xj, gamma)
            sx += k.sum()
            cntx += k.size
    exx = sx / cntx

    # E[k(Y,Y')]
    sy = 0.0; cnty = 0
    for i in range(0, ny, block):
        yi = y[i:i+block]
        for j in range(0, ny, block):
            yj = y[j:j+block]
            k = _rbf_block(yi, yj, gamma)
            sy += k.sum()
            cnty += k.size
    eyy = sy / cnty

    # E[k(X,Y)]
    sxy = 0.0; cntxy = 0
    for i in range(0, nx, block):
        xi = x[i:i+block]
        for j in range(0, ny, block):
            yj = y[j:j+block]
            k = _rbf_block(xi, yj, gamma)
            sxy += k.sum()
            cntxy += k.size
    exy = sxy / cntxy

    return float(exx + eyy - 2.0 * exy)

def mmd_weights(user_edge_vecs, ref_vec, gamma=0.01, eps=1e-12, block=65536, subsample=None, seed=0):
    """
    计算每个用户 edge 与 ref 的 MMD 权重:
      w_i ∝ 1 / (MMD_i + eps)

    subsample: 可选的下采样间隔（比如 10 表示每 10 个取 1 个），速度更快。
    """
    rng = np.random.default_rng(seed)

    def maybe_subsample(v):
        if subsample is None or subsample <= 1:
            return v
        # 统一用等间隔采样，避免随机性影响复现实验
        return v[::int(subsample)]

    r = maybe_subsample(ref_vec)
    mmds = []
    for v in user_edge_vecs:
        vv = maybe_subsample(v)
        m = compute_mmd_rbf(vv, r, gamma=gamma, block=block)
        mmds.append(m)

    inv = np.array([1.0 / (m + eps) for m in mmds], dtype=np.float64)
    w = inv / inv.sum()
    return mmds, w

# -----------------------------
# 聚合
# -----------------------------
def aggregate_states(states, weights):
    """对若干个 state_dict 做加权平均（只聚合共有且形状相同的键）"""
    keys = set(states[0].keys())
    for st in states[1:]:
        keys &= set(st.keys())

    out = {}
    for k in keys:
        vs = [st[k] for st in states]
        # 确保 dtype/shape 一致
        shape0 = vs[0].shape
        if not all(v.shape == shape0 for v in vs):
            continue
        if all(torch.is_floating_point(v) for v in vs):
            acc = None
            for v, w in zip(vs, weights):
                vv = v.float() * float(w)
                acc = vv if acc is None else (acc + vv)
            out[k] = acc.to(vs[0].dtype)
        else:
            # 非浮点参数（如整数 buffer）直接取第一个
            out[k] = vs[0]
    return out

def weighted_edge_average(edge_paths, weights, save_path=None):
    """用同一组权重对多个 edge_importance.npy 做加权平均并保存（可选）"""
    vecs = [load_edge_vector(p) for p in edge_paths]
    # 先统一长度
    L = min(v.size for v in vecs)
    vecs = [v[:L] for v in vecs]
    acc = np.zeros(L, dtype=np.float64)
    for v, w in zip(vecs, weights):
        acc += w * v
    if save_path:
        ensure_dir(Path(save_path).parent)
        np.save(save_path, acc)
    return acc

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, help="联邦轮次号（用于命名输出）")
    ap.add_argument("--ref_edge", type=str, required=True, help="参考（baseline）edge_importance.npy 路径")
    ap.add_argument("--user_models", nargs="+", required=True, help="用户本地模型 .pt 列表")
    ap.add_argument("--user_edges",  nargs="+", required=True, help="用户 edge_importance.npy 列表（与 user_models 一一对应）")
    # MMD 参数
    ap.add_argument("--mmd_gamma", type=float, default=0.01, help="RBF 核 gamma")
    ap.add_argument("--mmd_block", type=int, default=65536, help="MMD 分块大小")
    ap.add_argument("--mmd_eps",   type=float, default=1e-12, help="MMD 逆权重平滑项")
    ap.add_argument("--mmd_subsample", type=int, default=1, help="向量下采样间隔（>1 则加速）")
    # 输出
    ap.add_argument("--out_dir", type=str, default=None, help="输出目录（默认：trained/round{r}/global）")
    args = ap.parse_args()

    if len(args.user_models) != len(args.user_edges):
        raise ValueError("user_models 与 user_edges 数量不一致！")

    # 输出目录
    out_dir = args.out_dir or f"trained/round{args.round}/global"
    ensure_dir(out_dir)

    # ---- 1) 读取边重要性并计算权重（MMD 对 baseline 越小权重越大）
    print("[Info] loading ref edge:", args.ref_edge)
    ref_vec = load_edge_vector(args.ref_edge)
    user_vecs = [load_edge_vector(p) for p in args.user_edges]
    # 对齐长度（保守做法：取最短）
    minL = min([ref_vec.size] + [v.size for v in user_vecs])
    ref_vec = ref_vec[:minL]
    user_vecs = [v[:minL] for v in user_vecs]

    print(f"[Info] ref_len={ref_vec.size}, user_lens={[v.size for v in user_vecs]}")
    mmds, weights = mmd_weights(
        user_vecs,
        ref_vec,
        gamma=args.mmd_gamma,
        eps=args.mmd_eps,
        block=args.mmd_block,
        subsample=None if args.mmd_subsample <= 1 else args.mmd_subsample,
    )
    print("[Info] MMD:", [float(x) for x in mmds])
    print("[Info] Weights (sum=1):", [float(x) for x in weights])

    # ---- 2) 加载用户模型并做加权聚合
    print("[Info] loading user models...")
    states = [load_model_state(p) for p in args.user_models]
    agg_state = aggregate_states(states, weights)

    # ---- 3) 保存聚合后的模型
    out_model = os.path.join(out_dir, f"r{args.round}_aggregated_global_model.pt")
    save_model_state(agg_state, out_model)
    print("[Info] saved model ->", out_model)

    # ---- 4) 同步保存加权后的 edge importance（向量形式）
    out_edge = os.path.join(out_dir, f"r{args.round}_aggregated_edge_importance.npy")
    _ = weighted_edge_average(args.user_edges, weights, save_path=out_edge)
    print("[Info] saved edge ->", out_edge)

    # ---- 5) 保存指标
    metrics = {
        "round": args.round,
        "ref_edge": args.ref_edge,
        "user_models": args.user_models,
        "user_edges": args.user_edges,
        "mmd_gamma": args.mmd_gamma,
        "mmd_block": args.mmd_block,
        "mmd_eps": args.mmd_eps,
        "mmd_subsample": args.mmd_subsample,
        "mmds": [float(x) for x in mmds],
        "weights": [float(x) for x in weights],
        "out_model": out_model,
        "out_edge": out_edge,
    }
    out_metrics = os.path.join(out_dir, f"r{args.round}_aggregated_global_metrics.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[Info] saved metrics ->", out_metrics)


if __name__ == "__main__":
    main()
