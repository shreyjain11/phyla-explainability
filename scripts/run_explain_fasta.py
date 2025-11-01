import os, sys, json, csv, argparse, torch, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Phyla.phyla.model.model import Phyla
from integrations.phyla_mambalrp_adapter import PhylaMambaLRPAnalyzer


def read_fasta(path):
    names, seqs = [], []
    with open(path) as f:
        name, seq = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    names.append(name)
                    seqs.append("".join(seq))
                name = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if name is not None:
            names.append(name)
            seqs.append("".join(seq))
    return seqs, names


def save_matrix_csv(mat, names, out_csv, fmt="%.12g"):
    m = np.asarray(mat)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + list(names))
        for i, n in enumerate(names):
            w.writerow([n] + [fmt % m[i, j] for j in range(len(names))])


def save_relevances_csv(rel, names, out_csv, fmt="%.12g"):
    rel = np.asarray(rel)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["sequence_name"] + [f"pos_{i}" for i in range(rel.shape[1])]
        w.writerow(header)
        for i, n in enumerate(names):
            w.writerow([n] + [fmt % rel[i, j] for j in range(rel.shape[1])])


def plot_relevance_heatmap(rel, names, out_png, title):
    rel = np.asarray(rel)
    plt.figure(figsize=(max(8, rel.shape[1] / 4), max(3, len(names) * 0.4)))
    plt.imshow(rel, aspect="auto", interpolation="nearest", cmap="magma")
    plt.colorbar(label="relevance")
    plt.title(title)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xticks(range(rel.shape[1]), range(rel.shape[1]), fontsize=6, rotation=90)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="phyla-alpha", choices=["phyla-alpha", "phyla-beta"]) 
    p.add_argument("--device", default="cuda")
    p.add_argument("--ig_steps", type=int, default=256)
    p.add_argument("--nt_samples", type=int, default=8)
    p.add_argument("--nt_stdevs", type=float, default=0.02)
    p.add_argument("--nt_type", type=str, default="smoothgrad_sq")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    model = Phyla(name=args.model, device=device).eval()
    try:
        model.load()
        print(f"Loaded {args.model} pretrained weights.")
    except Exception as e:
        print("Proceeding without pretrained weights:", e)

    sequences, names = read_fasta(args.fasta)
    print(f"Read {len(sequences)} sequences from {args.fasta}")

    import integrations.phyla_mambalrp_adapter as adapter_mod
    adapter_mod.IG_STEPS = args.ig_steps
    adapter_mod.NT_SAMPLES = args.nt_samples
    adapter_mod.NT_STDEVS = args.nt_stdevs
    adapter_mod.NT_TYPE = args.nt_type

    analyzer = PhylaMambaLRPAnalyzer(model, device=device)
    res = analyzer.analyze_sequences(sequences, names)

    D = res.pairwise_distances.detach().cpu().numpy()
    save_matrix_csv(D, res.sequence_names, os.path.join(args.outdir, "pairwise_distances.csv"))

    R = res.relevances.detach().cpu().numpy()
    save_relevances_csv(R, res.sequence_names, os.path.join(args.outdir, "relevances_raw.csv"))

    Rsum = R.sum(axis=1, keepdims=True) + 1e-20
    Rn = R / Rsum
    save_relevances_csv(Rn, res.sequence_names, os.path.join(args.outdir, "relevances_norm.csv"))

    plot_relevance_heatmap(R,  res.sequence_names, os.path.join(args.outdir, "relevances_raw.png"),  "Raw relevance")
    plot_relevance_heatmap(Rn, res.sequence_names, os.path.join(args.outdir, "relevances_norm.png"), "Normalized relevance (sum=1)")

    try:
        tree = model.reconstruct_tree(res.sequence_embeddings, res.sequence_names)
        with open(os.path.join(args.outdir, "tree.newick"), "w") as f:
            tree.write(f)
        print(f"Saved Newick tree to {os.path.join(args.outdir, 'tree.newick')}")
    except Exception as e:
        print("Tree reconstruction failed:", e)

    meta = {
        "objective_value": res.objective_value,
        "sequence_names": res.sequence_names,
        "ig_steps": args.ig_steps,
        "nt_samples": args.nt_samples,
        "nt_stdevs": args.nt_stdevs,
        "nt_type": args.nt_type,
        "device": device,
        "model": args.model,
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done. Wrote raw and normalized relevances.")


if __name__ == "__main__":
    main()





