# merge_users.py
import argparse
import os
import pickle
import numpy as np

def load_triplet(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)) and len(obj) == 3:
        data, adj, mean = obj
        return np.asarray(data), np.asarray(adj), np.asarray(mean)
    elif isinstance(obj, (list, tuple)) and len(obj) == 15:
        # 兼容错误写法： [d1,a1,m1,d2,a2,m2,...]，自动修正为拼接后的三元组
        datas, adjs, means = [], [], []
        for i in range(0, 15, 3):
            datas.append(np.asarray(obj[i]))
            adjs.append(np.asarray(obj[i+1]))
            means.append(np.asarray(obj[i+2]))
        return np.concatenate(datas, axis=0), np.concatenate(adjs, axis=0), np.concatenate(means, axis=0)
    else:
        raise ValueError(f"Unexpected content in {pkl_path}: type={type(obj)}, len={len(obj) if isinstance(obj,(list,tuple)) else 'NA'}")

def main():
    ap = argparse.ArgumentParser(description="Merge multiple user train_data .pkl files into one 3-tuple [data, adj, mean_xy].")
    ap.add_argument("--inputs", nargs="+", required=True, help="List of user train_data_user*.pkl")
    ap.add_argument("--out", required=True, help="Output path for merged train_all.pkl")
    args = ap.parse_args()

    data_list, adj_list, mean_list = [], [], []
    total = 0
    for p in args.inputs:
        d, a, m = load_triplet(p)
        # 基本一致性检查：形状最后两维应一致 (T,V) / (V,V)
        if adj_list:
            # 检查 V
            v_prev = adj_list[0].shape[-1]
            if a.shape[-1] != v_prev or a.shape[-2] != v_prev:
                raise ValueError(f"Incompatible V between files: {p}")
        data_list.append(d)
        adj_list.append(a)
        mean_list.append(m)
        total += d.shape[0]
        print(f"[LOAD] {p} -> data {d.shape}, adj {a.shape}, mean {m.shape}")

    data_all = np.concatenate(data_list, axis=0)
    adj_all  = np.concatenate(adj_list,  axis=0)
    mean_all = np.concatenate(mean_list, axis=0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump([data_all, adj_all, mean_all], f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[SAVE] {args.out}")
    print(f"       data {data_all.shape}, adj {adj_all.shape}, mean {mean_all.shape}")

if __name__ == "__main__":
    main()