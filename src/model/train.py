import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import HORIZON, LOOKBACK, RESULTS_DIR, TICKERS, VAL_END
from src.model.dataset import build_datasets, collate_samples
from src.model.thgnn import THGNN

PRESETS = {"A": ("price", ("pos", "neg")),
           "B": ("full", ("pos", "neg")),
           "C": ("full", ("pos", "neg", "sent")),
           "D": ("full", ("sent",))}


def spearman(a, b) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def daily_rankic(scores, y, ymask) -> np.ndarray:
    ics = []
    for b in range(scores.shape[0]):
        m = ymask[b]
        if m.sum() >= 3:
            ics.append(spearman(scores[b][m], y[b][m]))
    return np.asarray(ics, dtype=float)


def nw_t_stat(ic, lag=HORIZON) -> float:
    ic = np.asarray(ic, dtype=float)
    n = len(ic)
    if n < 3:
        return 0.0
    e = ic - ic.mean()
    g0 = (e * e).mean()
    if g0 < 1e-12:
        return 0.0
    var = g0
    for l in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).mean()
    return float(ic.mean() / np.sqrt(max(var, 1e-12) / n))


@torch.no_grad()
def evaluate(model, ds, relations, device, batch_size=64, return_embeddings=False):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate_samples(b, relations))
    S, Y, M, Z, dates = [], [], [], [], []
    for batch in dl:
        edges = {r: (batch[f"ei_{r}"].to(device), batch[f"ew_{r}"].to(device))
                 for r in relations}
        scores, z = model(batch["x"].to(device), edges)
        S.append(scores.cpu().numpy())
        Z.append(z.cpu().numpy())
        Y.append(batch["y"].numpy())
        M.append(batch["ymask"].numpy())
        dates += batch["dates"]
    out = {"scores": np.concatenate(S), "y": np.concatenate(Y),
           "ymask": np.concatenate(M), "dates": dates,
           "ic": daily_rankic(np.concatenate(S), np.concatenate(Y),
                              np.concatenate(M))}
    if return_embeddings:
        out["Z"] = np.concatenate(Z)
    return out


def train_model(name, seed, epochs, batch_size, lr, patience, device):
    feature_set, relations = PRESETS[name]
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds, channels = build_datasets(feature_set, relations)
    model = THGNN(d_feat=len(channels), T=LOOKBACK,
                  relations=relations).to(device)
    dl = DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                    collate_fn=lambda b: collate_samples(b, relations))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_ic, best_state, best_ep, bad = -np.inf, None, -1, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = nb = 0
        for batch in dl:
            edges = {r: (batch[f"ei_{r}"].to(device), batch[f"ew_{r}"].to(device))
                     for r in relations}
            scores, _ = model(batch["x"].to(device), edges)
            lm = batch["lmask"].to(device)
            loss = F.binary_cross_entropy_with_logits(
                scores[lm], batch["label"].to(device)[lm])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        vic = evaluate(model, ds["val"], relations, device)["ic"].mean()
        if vic > best_ic + 1e-4:
            best_ic, best_ep, bad = vic, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0:
            print(f"ep {ep:3d}  train_bce {tot / max(nb, 1):.4f}  "
                  f"val_IC {vic:+.4f}  best {best_ic:+.4f}")
        if bad >= patience:
            print(f"early stop at ep {ep} (best ep {best_ep})")
            break

    model.load_state_dict(best_state)
    test = evaluate(model, ds["test"], relations, device, return_embeddings=True)
    print(f"\nTEST  mean RankIC {test['ic'].mean():+.4f}  "
          f"NW t({HORIZON}) {nw_t_stat(test['ic']):+.2f}  "
          f"days {len(test['ic'])}")
    return model, channels, best_ic, best_ep, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="C", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if a.device == "auto" else a.device))

    model, channels, vic, best_ep, test = train_model(
        a.model, a.seed, a.epochs, a.batch_size, a.lr, a.patience, device)

    tag = f"{a.model}_seed{a.seed}"
    rows = [{"date": d, "ticker": t, "score": test["scores"][b, i],
             "y": test["y"][b, i]}
            for b, d in enumerate(test["dates"])
            for i, t in enumerate(TICKERS) if test["ymask"][b, i]]
    pd.DataFrame(rows).to_parquet(RESULTS_DIR / f"preds_{tag}.parquet", index=False)
    np.savez_compressed(RESULTS_DIR / f"embeddings_{tag}.npz",
                        dates=np.array([np.datetime64(d) for d in test["dates"]]),
                        tickers=np.array(TICKERS),
                        Z=test["Z"].astype(np.float32))
    metrics = {"model": a.model, "seed": a.seed,
               "params": sum(p.numel() for p in model.parameters()),
               "features": PRESETS[a.model][0],
               "relations": list(PRESETS[a.model][1]),
               "best_epoch": best_ep, "val_ic": float(vic),
               "test_ic_mean": float(test["ic"].mean()),
               "test_ic_nw_t": nw_t_stat(test["ic"]),
               "test_days": int(len(test["ic"]))}
    (RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), RESULTS_DIR / f"ckpt_{tag}.pt")
    print(f"saved preds / embeddings / metrics / ckpt for {tag}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
