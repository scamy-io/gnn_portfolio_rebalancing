import argparse
import json

import pandas as pd
import torch

from src.config import RESULTS_DIR, TICKERS
from src.model.train import PRESETS, nw_t_stat, train_model


def run_single(name, seed, epochs, batch_size, lr, patience, device):
    tag = f"{name}_seed{seed}"
    model, channels, best_ic, best_ep, test = train_model(
        name, seed, epochs, batch_size, lr, patience, device)

    pd.DataFrame({"date": test["dates"], "ic": test["ic"]}).to_csv(
        RESULTS_DIR / f"ic_series_{tag}.csv", index=False)

    rows = [{"date": d, "ticker": t, "score": test["scores"][b, i],
             "y": test["y"][b, i]}
            for b, d in enumerate(test["dates"])
            for i, t in enumerate(TICKERS) if test["ymask"][b, i]]
    pd.DataFrame(rows).to_parquet(RESULTS_DIR / f"preds_{tag}.parquet", index=False)

    metrics = {"model": name, "seed": seed,
               "params": sum(p.numel() for p in model.parameters()),
               "features": PRESETS[name][0],
               "relations": list(PRESETS[name][1]),
               "best_epoch": best_ep, "val_ic": float(best_ic),
               "test_ic_mean": float(test["ic"].mean()),
               "test_ic_nw_t": nw_t_stat(test["ic"]),
               "test_days": int(len(test["ic"]))}
    (RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ABCD")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    device = torch.device(
        "cuda" if a.device == "auto" and torch.cuda.is_available()
        else ("cpu" if a.device == "auto" else a.device))

    all_m = []
    for name in a.models:
        for seed in a.seeds:
            tag = f"{name}_seed{seed}"
            m_path = RESULTS_DIR / f"metrics_{tag}.json"
            i_path = RESULTS_DIR / f"ic_series_{tag}.csv"
            if a.skip_existing and m_path.exists() and i_path.exists():
                print(f"[skip] {tag} (use without --skip-existing to force)")
                all_m.append(json.loads(m_path.read_text()))
                continue
            m = run_single(name, seed, a.epochs, a.batch_size, a.lr,
                           a.patience, device)
            all_m.append(m)
            print(f"[done] {tag}: val {m['val_ic']:+.4f}  "
                  f"test {m['test_ic_mean']:+.4f}  (NW t {m['test_ic_nw_t']:+.2f})")

    df = pd.DataFrame(all_m)
    summary = (df.groupby("model")
                 .agg(val_mean=("val_ic", "mean"), val_sd=("val_ic", "std"),
                      test_mean=("test_ic_mean", "mean"),
                      test_sd=("test_ic_mean", "std"), n=("seed", "size")))
    print("\n--- matrix summary (frozen HPs, confirmatory) ---")
    print(summary.round(4).to_string())
    df.to_csv(RESULTS_DIR / "matrix_runs.csv", index=False)
    print(f"saved -> {RESULTS_DIR / 'matrix_runs.csv'}")


if __name__ == "__main__":
    main()
