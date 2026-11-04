"""Memory-bounded variant of btcmovement2/scripts/train_production.py (the original repo is untouched):
same architecture and hyper-parameters, but every split is thinned by whole windows (keep 1/N) so the
refit fits in 15 GB.  Walk-forward: --end 2026-07-31 -> models_<head>_valid<TAG>.pkl in /home/user/data/models."""
import os, sys, pickle, argparse, numpy as np, pandas as pd
os.environ.setdefault("BTC_DATA_DIR", "/home/user/data"); os.environ["BTC_REGIME"] = "1"
sys.path.insert(0, "/home/user/btcmovement2"); sys.path.insert(0, "/home/user/btcmovement2/scripts")
from train_eval import load_head, MODEL_DIR, STACK_DIR, HEAD_ORDER  # noqa: E402
from btcmodel.models import HeadModel, _fit, VOL_FEATS, DIR_FEATS  # noqa: E402
from btcmodel.events import touch_rows, TouchModel, fp_rows, FirstPassageModel  # noqa: E402
from btcmodel.regime import REGIME_FEATS  # noqa: E402
from btcmodel.dataset import SPLITS  # noqa: E402
from datetime import date
ES_DAYS = 30
ap = argparse.ArgumentParser(); ap.add_argument("--end", default="2026-07-31"); ap.add_argument("--tag", default="_wf0731")
ap.add_argument("--thin", type=int, default=3); ap.add_argument("--old-thin", type=int, default=4); ap.add_argument("heads", nargs="*", default=HEAD_ORDER)
a = ap.parse_args(); tag = a.tag
for head in a.heads:
    splits = [s for s in ("train", "valid", "test") if SPLITS[s][0] <= date.fromisoformat(a.end)]
    parts = []
    for sp in splits:                                   # thin each split right after loading (memory)
        x = load_head(sp, head); w = x["T0"].values // 300; old = x["day"].values < 365
        keep = np.where(old, (w % a.old_thin) == 0, (w % a.thin) == 0)
        parts.append(x[keep].reset_index(drop=True)); del x
    d = pd.concat(parts, ignore_index=True); del parts
    end_day = int((np.datetime64(a.end) - np.datetime64("2025-01-01")) / np.timedelta64(1, "D"))
    d = d[d["day"].values <= end_day].reset_index(drop=True)
    last = d["day"].max(); es = d["day"].values > last - ES_DAYS; gap = (d["day"].values > last - ES_DAYS - 1) & ~es
    tr, va = d[~es & ~gap].reset_index(drop=True), d[es].reset_index(drop=True); del d
    print(head, "rows train", len(tr), "valid", len(va), flush=True)
    extra = []
    for sh in [h for h in HEAD_ORDER if HEAD_ORDER.index(h) < HEAD_ORDER.index(head)]:
        for x, key in ((tr, "tr"), (va, "va")):
            s = pd.read_parquet(os.path.join(STACK_DIR, f"prod{tag}_{sh}_{key}.parquet"))
            x[f"mu_{sh}_px"] = pd.merge(x[["T0", "e"]], s, on=["T0", "e"], how="left")["drift_px"].values
            x[f"mu_{sh}_px"] = (x[f"mu_{sh}_px"] / x["naive_scale"]).astype(np.float32)
        extra.append(f"mu_{sh}_px")
    hm = HeadModel(head, extra_dir_feats=extra, regime_feats=REGIME_FEATS).fit(tr, va)
    for x, key in ((tr, "tr"), (va, "va")):
        pd.DataFrame({"T0": x["T0"].values, "e": x["e"].values, "drift_px": x["mu"].values * x["sig"].values}).to_parquet(os.path.join(STACK_DIR, f"prod{tag}_{head}_{key}.parquet"))
    feats = sorted(set(VOL_FEATS + DIR_FEATS + ["lH", "z_naive", "lproj_300", "lsint"] + REGIME_FEATS))
    ytr, yva = (tr[f"{head}__y"].values >= 0).astype(int), (va[f"{head}__y"].values >= 0).astype(int)
    dm = _fit(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000), tr[feats], ytr, va[feats], yva, es=100)
    sub = lambda x, f: x.iloc[::f].reset_index(drop=True)
    tm = TouchModel().fit(touch_rows(sub(tr, 4), head, hm), touch_rows(sub(va, 2), head, hm))
    fm = FirstPassageModel().fit(fp_rows(sub(tr, 4), head, hm), fp_rows(sub(va, 2), head, hm))
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"models_{head}_valid{tag}.pkl"), "wb") as fh:
        pickle.dump({"head": hm, "touch": tm, "fp": fm, "direct": dm}, fh)
    print("saved", head, flush=True); del tr, va
