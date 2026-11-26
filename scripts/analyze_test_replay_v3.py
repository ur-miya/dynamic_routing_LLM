from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "artifacts/test_replay/frozen_v3"
SEEDS, DRAWS = (42, 43, 44), 1000
DIRECT = ["direct_LR_BE", "direct_LR_BEU", "direct_Ridge_BE", "direct_Ridge_BEU"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def band(draws, est):
    se = draws.std(axis=0, ddof=1)
    se = np.where(se > 0, se, np.nan)
    t = np.nanmax(np.abs(draws - est) / se, axis=1)
    return se, float(np.quantile(t, .95))


def main():
    run = Path((T / "latest_replay.txt").read_text().strip())
    man = json.loads((run / "manifest.json").read_text())
    for n, d in man["outputs_sha256"].items():
        if sha(run / n) != d:
            raise ValueError(f"Replay output changed: {n}")
    pre = json.loads((T / "preflight_20261124T200030_635202Z/manifest.json").read_text())
    el = pd.read_parquet(pre["eligible_output"]["path"])
    dec = pd.read_parquet(run / "test_decisions.parquet")
    summ = pd.read_csv(run / "policy_summary.csv")
    pols = [p for p in summ["policy_id"] if p not in ("always_student", "always_teacher")]
    wide = dec.pivot(index="pair_id", columns="policy_id", values="teacher").loc[el["pair_id"], pols].astype(float)
    t = wide.to_numpy()
    qS, qT = el["q_S"].to_numpy(float), el["q_T"].to_numpy(float)
    q = np.where(t > 0, qT[:, None], qS[:, None])
    P = len(pols)
    meth = [p.rsplit("__", 1)[0] for p in pols]
    bud = [float(p.rsplit("__", 1)[1]) for p in pols]
    targets = [i for i, p in enumerate(pols) if meth[i] in ("cascade_A", "hybrid_A")]
    curves = {m: [i for i in range(P) if meth[i] == m] for m in DIRECT}

    def stats(C, A, Qs, n, SS, ST):
        N = C @ n
        f = (C @ A) / N[:, None]; Qm = (C @ Qs) / N[:, None]
        s = (C @ SS) / N; tt = (C @ ST) / N
        gain = Qm - (s[:, None] + f * (tt - s)[:, None])
        diffs = np.empty((len(C), len(targets) * 4))
        for d in range(len(C)):
            k = 0
            for i in targets:
                for m in DIRECT:
                    ix = curves[m]
                    x = np.r_[0., f[d, ix], 1.]; y = np.r_[s[d], Qm[d, ix], tt[d]]
                    o = np.argsort(x)
                    diffs[d, k] = Qm[d, i] - np.interp(f[d, i], x[o], y[o]); k += 1
        return gain, diffs

    cmp_names = [(pols[i], m) for i in targets for m in DIRECT]
    rows_g, rows_c = [], []
    for ctype in ("prompt_id", "context_hash", "message_tree_id"):
        codes, _ = pd.factorize(el[ctype]); K = codes.max() + 1
        agg = lambda M: np.vstack([np.bincount(codes, weights=M[:, j], minlength=K) for j in range(M.shape[1])]).T
        A, Qs = agg(t), agg(q)
        n = np.bincount(codes, minlength=K).astype(float)
        SS, ST = np.bincount(codes, weights=qS, minlength=K), np.bincount(codes, weights=qT, minlength=K)
        g0, d0 = stats(np.ones((1, K)), A, Qs, n, SS, ST)
        G, D, per = [], [], {}
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            C = np.vstack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(DRAWS)]).astype(float)
            g, d = stats(C, A, Qs, n, SS, ST)
            G.append(g); D.append(d)
            per[sd] = (band(g, g0[0]), band(d, d0[0]), g, d)
        Gp, Dp = np.vstack(G), np.vstack(D)
        seg, cg = band(Gp, g0[0]); sed, cd = band(Dp, d0[0])
        for j, p in enumerate(pols):
            ex = [bool(abs(g0[0, j]) > per[s][0][1] * per[s][0][0][j]) for s in SEEDS]
            rows_g.append({"cluster": ctype, "policy_id": p, "gain_over_line": g0[0, j],
                "ci95_lo": np.quantile(Gp[:, j], .025), "ci95_hi": np.quantile(Gp[:, j], .975),
                "band_lo": g0[0, j] - cg * seg[j], "band_hi": g0[0, j] + cg * seg[j],
                "band_excludes_zero_by_seed": "".join("Y" if e else "n" for e in ex)})
        for j, (p, m) in enumerate(cmp_names):
            ex = [bool(abs(d0[0, j]) > per[s][1][1] * per[s][1][0][j]) for s in SEEDS]
            rows_c.append({"cluster": ctype, "policy_id": p, "vs_direct_curve": m, "diff": d0[0, j],
                "ci95_lo": np.quantile(Dp[:, j], .025), "ci95_hi": np.quantile(Dp[:, j], .975),
                "band_lo": d0[0, j] - cd * sed[j], "band_hi": d0[0, j] + cd * sed[j],
                "band_excludes_zero_by_seed": "".join("Y" if e else "n" for e in ex)})
    out = T / datetime.now(timezone.utc).strftime("analysis_%Y%m%dT%H%M%S_%fZ")
    out.mkdir()
    g = pd.DataFrame(rows_g); c = pd.DataFrame(rows_c)
    g.to_csv(out / "gain_over_line.csv", index=False); c.to_csv(out / "paired_vs_direct.csv", index=False)
    json.dump({"status": "complete", "replay_run": str(run), "seeds": SEEDS, "draws_per_seed": DRAWS,
               "comparisons": len(cmp_names), "winner_selected": False, "thresholds_changed": False,
               "interpolation_bias": "linear interpolation of concave direct curves favors cascade/hybrid",
               "outputs_sha256": {n: sha(out / n) for n in ("gain_over_line.csv", "paired_vs_direct.csv")}},
              open(out / "manifest.json", "w"), indent=2)
    f = lambda x: f"{x:+.4f}"
    cc = c.assign(ci=lambda d: d.ci95_lo.map(f) + ".." + d.ci95_hi.map(f), band=lambda d: d.band_lo.map(f) + ".." + d.band_hi.map(f), diff=lambda d: d["diff"].map(f))
    txt = f"RUN: {out}\nSTATUS: complete\nWINNER_SELECTED: False\nTHRESHOLDS_CHANGED: False\n\n"
    txt += "=== SIMULTANEOUS BAND EXCLUDES ZERO (paired vs direct), by cluster, count of 28 per seed-pattern YYY ===\n"
    txt += c.groupby("cluster")["band_excludes_zero_by_seed"].apply(lambda s: int((s == "YYY").sum())).to_string() + "\n\n"
    txt += "=== PAIRED vs DIRECT, cluster=prompt_id ===\n" + cc[cc.cluster == "prompt_id"][["policy_id", "vs_direct_curve", "diff", "ci", "band", "band_excludes_zero_by_seed"]].to_string(index=False) + "\n\n"
    gg = g.assign(gain=lambda d: d.gain_over_line.map(f), band=lambda d: d.band_lo.map(f) + ".." + d.band_hi.map(f))
    txt += "=== GAIN OVER LINE, cluster=prompt_id ===\n" + gg[gg.cluster == "prompt_id"][["policy_id", "gain", "band", "band_excludes_zero_by_seed"]].to_string(index=False) + "\n"
    (out / "SEND_BACK.txt").write_text(txt); (T / "latest_analysis.txt").write_text(str(out) + "\n")
    print(txt)

if __name__ == "__main__":
    main()
