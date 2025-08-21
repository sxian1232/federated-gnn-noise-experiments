# 向 processed_data 的训练数据 (train_data_*.pkl) 中的 x/y 位置通道注入高斯噪声
# 仅对 mask=1 的位置注入；不改动邻接矩阵和 mean_xy

import argparse
import os
import pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Inject Gaussian noise into x/y channels of processed train data.")
    ap.add_argument("--in", dest="inp", required=True, help="输入 pkl（形如 processed_data/train_data_user1.pkl）")
    ap.add_argument("--out", dest="out", required=True, help="输出 pkl 路径")
    ap.add_argument("--sigma", type=float, required=True, help="高斯噪声标准差（单位与位置坐标一致，e.g. 米）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（可复现实验）")
    # 可选：需要时可改通道索引；默认 3/4 是 x/y，10 是 mask（与你的工程一致）
    ap.add_argument("--x_idx", type=int, default=3, help="x 通道在 C 维的索引")
    ap.add_argument("--y_idx", type=int, default=4, help="y 通道在 C 维的索引")
    ap.add_argument("--mask_idx", type=int, default=10, help="mask 通道在 C 维的索引（1 表示该节点该时刻存在）")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 读取
    with open(args.inp, "rb") as f:
        all_data, all_adj, all_mean_xy = pickle.load(f)
        # all_data: (N, C=11, T, V)
        # all_adj:  (N, V, V)
        # all_mean_xy: (N, 2)

    # 基本检查
    if all_data.ndim != 4:
        raise ValueError(f"Unexpected all_data shape: {all_data.shape}")
    N, C, T, V = all_data.shape
    if not (0 <= args.x_idx < C and 0 <= args.y_idx < C and 0 <= args.mask_idx < C):
        raise ValueError("Channel index out of range.")

    # 仅对 mask==1 的位置注入噪声
    # mask 形状: (N, T, V) 从 (N, C, T, V) 取出第 mask_idx 通道
    mask = all_data[:, args.mask_idx, :, :]  # float in {0,1}
    # 生成噪声
    noise_x = rng.normal(loc=0.0, scale=args.sigma, size=(N, T, V)).astype(all_data.dtype, copy=False)
    noise_y = rng.normal(loc=0.0, scale=args.sigma, size=(N, T, V)).astype(all_data.dtype, copy=False)

    # 只在 mask==1 的位置加噪
    noise_x *= (mask > 0.5)
    noise_y *= (mask > 0.5)

    # 写回 x/y 通道
    all_data[:, args.x_idx, :, :] += noise_x
    all_data[:, args.y_idx, :, :] += noise_y

    # 保存
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump([all_data, all_adj, all_mean_xy], f)

    # 简要信息
    print(f"[DONE] noise injected: sigma={args.sigma}, seed={args.seed}")
    print(f"       in : {args.inp}")
    print(f"       out: {args.out}")
    print(f"       shape all_data: {all_data.shape}, all_adj: {all_adj.shape}, all_mean_xy: {all_mean_xy.shape}")


if __name__ == "__main__":
    main()
