"""
Aggregate all run metrics and analysis CSVs into machine-readable tables and the required figures.

    python -m cort_mvp.export_report
Outputs: results/metrics/action_metrics.csv, summary_by_experiment.csv, *.md tables in reports/tables/,
         figures in results/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cort_mvp.common import WS  # noqa: E402
from cort_mvp.analysis.common import RUN_ROOT, run_info  # noqa: E402

MET = WS / "results" / "metrics"
FIG = WS / "results" / "figures"
TAB = WS / "reports" / "tables"
PRIMARY = ["m0", "m1", "m2", "m3", "m4"]
COLORS = {"m0": "k", "m1": "tab:gray", "m2": "tab:blue", "m3": "tab:red", "m4": "tab:green"}
LABELS = {"m0": "M0 ridge (frozen DINO)", "m1": "M1 direct regression", "m2": "M2 endpoint ODE", "m3": "M3 FM + endpoint", "m4": "M4 FM + CF invariance"}


def exp_key(info: dict) -> str:
    """Group label for an experiment configuration (model x projector x backbone x fraction x n_steps x pairing)."""
    k = info["exp"] if info["kind"] != "m0" else info["run"]
    extras = []
    if info.get("projector", "linear") not in ("linear", "none"):
        extras.append(info["projector"])
    if info.get("backbone", "dinov3_vits16") != "dinov3_vits16" and info["kind"] != "m0":
        extras.append(info["backbone"])
    if info.get("train_fraction", 1.0) < 1.0:
        extras.append(f"frac{info['train_fraction']}")
    if info["kind"] in ("m2", "m3", "m4") and info.get("n_steps", 8) != 8:
        extras.append(f"nsteps{info['n_steps']}")
    return k + ("[" + ",".join(extras) + "]" if extras else "")


def collect_action_metrics() -> pd.DataFrame:
    rows = []
    for rd in sorted(RUN_ROOT.iterdir()):
        if not (rd / "eval.json").exists():
            continue
        e = json.load(open(rd / "eval.json")); info = run_info(rd.name); key = exp_key(info)
        rj = json.load(open(rd / "run.json"))
        base = dict(run=rd.name, exp=key, kind=info["kind"], seed=info.get("seed", 0), nonfinite_steps=rj.get("nonfinite_loss_steps", 0),
                    best_step=rj.get("best_step", -1), params=rj.get("params", {}).get("trainable_total", 0))
        for s, m in e["splits"].items():
            rows.append({**base, "split": s, **{k: v for k, v in m.items() if k in ("mse", "rmse", "angular_err_deg", "r2", "nmse")}})
        rows.append({**base, "split": "cf_consistency_all", "mse": e["cf_action_consistency"]["all_pairs"]})
        rows.append({**base, "split": "cf_consistency_seen", "mse": e["cf_action_consistency"]["seen_seen"]})
        rows.append({**base, "split": "cf_consistency_unseen", "mse": e["cf_action_consistency"]["unseen_unseen"]})
        for n, d in e.get("step_sweep", {}).items():
            for s, v in d.items():
                rows.append({**base, "split": f"{s}@steps{n}", "mse": v})
    df = pd.DataFrame(rows)
    df.to_csv(MET / "action_metrics.csv", index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["exp", "kind", "split"])
    out = g.agg(n=("mse", "size"), mse_mean=("mse", "mean"), mse_std=("mse", "std"), ang_mean=("angular_err_deg", "mean"),
                ang_std=("angular_err_deg", "std"), r2_mean=("r2", "mean"), r2_std=("r2", "std")).reset_index()
    out.to_csv(MET / "summary_by_experiment.csv", index=False)
    return out


def md_table(df: pd.DataFrame, cols, floatfmt=".4f") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if isinstance(v, float) and np.isnan(v) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_main(summary: pd.DataFrame, exps: list[str], name: str):
    rows = []
    for e in exps:
        s = summary[summary.exp == e]
        if s.empty:
            continue
        def ms(split, col="mse"):
            r = s[s.split == split]
            if r.empty:
                return ""
            return f"{r[f'{col}_mean'].values[0]:.4f} ± {r[f'{col}_std'].values[0]:.4f}" if r.n.values[0] > 1 else f"{r[f'{col}_mean'].values[0]:.4f}"
        rows.append(dict(experiment=e, n_seeds=int(s.n.max()), id_test=ms("id_test"), ood_bg=ms("ood_bg"), geo_ood=ms("geo_ood"),
                         id_angular_deg=ms("id_test", "ang"), ood_bg_angular_deg=ms("ood_bg", "ang"),
                         cf_consistency=ms("cf_consistency_all"), cf_consistency_unseen=ms("cf_consistency_unseen")))
    t = pd.DataFrame(rows)
    (TAB / f"{name}.md").write_text(md_table(t, list(t.columns)) if not t.empty else "(no runs)")
    return t


def fig_action_errors(summary: pd.DataFrame):
    exps = [e for e in PRIMARY if (summary.exp.str.startswith(e) & (summary.exp.str.len() == len(e))).any() or e == "m0"]
    splits = ["id_test", "ood_bg", "geo_ood", "cf_consistency_all"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, sp in zip(axes, splits):
        xs, ys, es, cs = [], [], [], []
        for e in PRIMARY:
            key = "m0_ridge_dinov3_vits16_cls" if e == "m0" else e
            r = summary[(summary.exp == key) & (summary.split == sp)]
            if r.empty:
                continue
            xs.append(LABELS[e].split(" ")[0]); ys.append(r.mse_mean.values[0]); es.append(0 if np.isnan(r.mse_std.values[0]) else r.mse_std.values[0]); cs.append(COLORS[e])
        ax.bar(xs, ys, yerr=es, color=cs, capsize=4)
        ax.set_title(sp); ax.set_ylabel("action MSE" if "cf" not in sp else "mean pairwise ||Δa||² across cf variants")
    plt.suptitle("Action prediction error (mean ± std over seeds; raw action units)")
    plt.tight_layout(); plt.savefig(FIG / "fig2_action_errors.png", dpi=110); plt.close()


def _stage_x(stage: str):
    if stage.startswith("t="):
        return float(stage[2:])
    return None


def _plot_vs_time(df, ycol, title, ylabel, fname, split=None, target=None, probe=None, logy=False, exps=None):
    d = df.copy()
    if split is not None: d = d[d.split == split]
    if target is not None: d = d[d.target == target]
    if probe is not None: d = d[d.probe == probe]
    d["exp"] = [exp_key(run_info(r)) for r in d.run]
    exps = exps or [e for e in ["m2", "m3", "m4"] if e in set(d.exp)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axes[0]
    for e in exps:
        dd = d[d.exp == e]; dd = dd[dd.stage.str.startswith("t=")]
        if dd.empty: continue
        g = dd.groupby("stage")[ycol].agg(["mean", "std"]).reset_index(); g["t"] = g.stage.map(_stage_x); g = g.sort_values("t")
        ax.errorbar(g.t, g["mean"], yerr=g["std"].fillna(0), label=LABELS.get(e, e), color=COLORS.get(e, None), marker="o", ms=3, capsize=2)
    for e, ls in (("m0_ridge_dinov3_vits16_cls", ":"),):
        dd = d[d.exp == e]
        if not dd.empty:
            ax.axhline(dd[ycol].mean(), color="k", ls=ls, label="M0 frozen DINO (h)")
    ax.set_xlabel("transport time t (numerical ODE, 10 Euler steps)"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)
    if logy: ax.set_yscale("log")
    ax2 = axes[1]
    dd = d[d.exp == "m1"]
    if not dd.empty:
        order = ["z0"] + sorted([s for s in dd.stage.unique() if s.startswith("h")], key=lambda s: int(s[1:])) + ["z1"]
        g = dd.groupby("stage")[ycol].agg(["mean", "std"]).reindex(order)
        ax2.errorbar(range(len(order)), g["mean"], yerr=g["std"].fillna(0), color=COLORS["m1"], marker="s", capsize=2)
        ax2.set_xticks(range(len(order))); ax2.set_xticklabels(order)
    ax2.set_title("M1 stages (layer index ≠ FM time)"); ax2.set_ylabel(ylabel)
    if logy: ax2.set_yscale("log")
    plt.tight_layout(); plt.savefig(FIG / fname, dpi=110); plt.close()


def fig_probes(probes: pd.DataFrame):
    for split in ("id_test", "ood_bg", "geo_ood"):
        _plot_vs_time(probes, "r2", f"Ridge probe R² for rel_goal — {split}", "test R²", f"fig3_probe_r2_relgoal_{split}.png",
                      split=split, target="rel_goal", probe="ridge")
    _plot_vs_time(probes, "r2", "Ridge probe R² for absolute state (ee, goal) — id_test", "test R²", "fig3b_probe_r2_stateabs_id.png",
                  split="id_test", target="state_abs", probe="ridge")
    _plot_vs_time(probes, "r2", "MLP(64) probe R² for rel_goal — ood_bg", "test R²", "fig3c_probe_mlp_relgoal_ood.png",
                  split="ood_bg", target="rel_goal", probe="mlp64")


def fig_invariance(inv: pd.DataFrame):
    _plot_vs_time(inv, "z_R_all_rand", "Nuisance ratio R = within-state / between-state (z-scored)", "R (log)", "fig4_nuisance_ratio.png", logy=True)
    _plot_vs_time(inv, "raw_R_all_rand", "Nuisance ratio R (raw coordinates)", "R (log)", "fig4b_nuisance_ratio_raw.png", logy=True)
    _plot_vs_time(inv, "z_within_all", "Within-state variation (counterfactual pairs, z-scored)", "mean ||Δz||²", "fig5a_within.png", logy=True)
    _plot_vs_time(inv, "z_between_rand", "Between-state variation (same nuisance, z-scored)", "mean ||Δz||²", "fig5b_between.png", logy=True)
    _plot_vs_time(inv, "z_R_unseen_rand", "Nuisance ratio, unseen-family pairs only (z-scored)", "R (log)", "fig4c_nuisance_ratio_unseen.png", logy=True)
    _plot_vs_time(inv, "train_erank", "Effective rank of train representations", "entropy effective rank", "fig7_effective_rank.png")


def fig_subspaces(sub: pd.DataFrame):
    _plot_vs_time(sub, "frac_eta_in_C", "Fraction of nuisance-difference energy inside the control subspace C", "fraction", "fig7b_frac_eta_in_C.png")
    _plot_vs_time(sub, "frac_x_in_C", "Fraction of state-difference energy inside the control subspace C", "fraction", "fig7c_frac_x_in_C.png")
    _plot_vs_time(sub, "selectivity_C", "Selectivity = frac_x_in_C / frac_eta_in_C", "selectivity (log)", "fig7d_selectivity.png", logy=True)
    _plot_vs_time(sub, "overlap_eta_pc4_C", "Principal-angle overlap (mean cos²) between top-4 nuisance PCs and C", "mean cos²", "fig7e_overlap_pc4.png")
    _plot_vs_time(sub, "erank_eta", "Effective rank of nuisance differences δη", "erank", "fig7f_erank_eta.png")


KEY_STAGES = ["h", "z0", "h2", "z1", "t=0.0", "t=0.5", "t=0.9", "t=1.0"]


def _stage_table(df: pd.DataFrame, cols: list[str], name: str, exps=None, filt=None, stages=KEY_STAGES, floatfmt=".3f"):
    """Mean ± std over seeds of `cols` at a few key stages, for the primary experiments (plus any `exps`)."""
    d = df.copy(); d["exp"] = [exp_key(run_info(r)) for r in d.run]
    if filt is not None:
        d = filt(d)
    exps = exps or (["m0_ridge_dinov3_vits16_cls"] + [e for e in PRIMARY[1:] if e in set(d.exp)])
    d = d[d.exp.isin(exps) & d.stage.isin(stages)]
    rows = []
    for e in exps:
        for st in stages:
            dd = d[(d.exp == e) & (d.stage == st)]
            if dd.empty: continue
            row = {"experiment": e, "stage": st, "n": len(dd)}
            for c in cols:
                m, s = dd[c].mean(), dd[c].std()
                row[c] = f"{m:{floatfmt}} ± {s:{floatfmt}}" if len(dd) > 1 else f"{m:{floatfmt}}"
            rows.append(row)
    t = pd.DataFrame(rows)
    (TAB / f"{name}.md").write_text(md_table(t, list(t.columns)) if not t.empty else "(no rows)")
    return t


def tables_mechanism():
    p = MET / "probes.csv"
    if p.exists():
        pr = pd.read_csv(p)
        for split in ("id_test", "ood_bg", "geo_ood"):
            _stage_table(pr, ["r2", "nmse"], f"table_probe_ridge_relgoal_{split}", filt=lambda d, s=split: d[(d.split == s) & (d.probe == "ridge") & (d.target == "rel_goal")])
        _stage_table(pr, ["r2"], "table_probe_mlp_relgoal_ood_bg", filt=lambda d: d[(d.split == "ood_bg") & (d.probe == "mlp64") & (d.target == "rel_goal")])
        _stage_table(pr, ["r2"], "table_probe_ridge_stateabs_id_test", filt=lambda d: d[(d.split == "id_test") & (d.probe == "ridge") & (d.target == "state_abs")])
    p = MET / "invariance.csv"
    if p.exists():
        _stage_table(pd.read_csv(p), ["z_R_all_rand", "raw_R_all_rand", "z_R_unseen_rand", "z_within_all", "z_between_rand", "train_erank", "train_total_var"], "table_invariance", floatfmt=".3g")
    p = MET / "subspaces.csv"
    if p.exists():
        _stage_table(pd.read_csv(p), ["frac_x_in_C", "frac_eta_in_C", "selectivity_C", "overlap_eta_pc4_C", "erank_eta", "erank_x", "frac_x_in_W", "frac_eta_in_W"], "table_subspaces", floatfmt=".3g")
    for frac, suf in ((None, ""), ("0.1", "_frac0.1")):
        p = MET / f"sensitivity{suf}.csv"
        if p.exists():
            _stage_table(pd.read_csv(p), ["Jx_norm", "Jeta_norm", "ratio_eta_over_x", "erank_Jx", "erank_Jeta"], f"table_sensitivity{suf}", floatfmt=".3g")


def table_step_sweep(df: pd.DataFrame):
    d = df[df.split.str.contains("@steps")].copy()
    if d.empty: return
    d["eval_steps"] = d.split.str.extract(r"@steps(\d+)").astype(int); d["split"] = d.split.str.replace(r"@steps\d+", "", regex=True)
    g = d.groupby(["exp", "eval_steps", "split"]).mse.agg(["mean", "std", "size"]).reset_index()
    t = g.pivot_table(index=["exp", "eval_steps"], columns="split", values=["mean", "std"]).reset_index()
    rows = []
    for _, r in t.iterrows():
        row = {"experiment": r["exp"].values[0] if hasattr(r["exp"], "values") else r["exp"], "eval_steps": int(r["eval_steps"].values[0] if hasattr(r["eval_steps"], "values") else r["eval_steps"])}
        for s in ("id_test", "ood_bg", "geo_ood"):
            m, sd = r[("mean", s)], r[("std", s)]
            row[s] = f"{m:.4f} ± {sd:.4f}" if not np.isnan(sd) else f"{m:.4f}"
        rows.append(row)
    (TAB / "table_step_sweep.md").write_text(md_table(pd.DataFrame(rows), ["experiment", "eval_steps", "id_test", "ood_bg", "geo_ood"]))


def fig_sensitivity(sens: pd.DataFrame, suffix: str = ""):
    tag = f" (step = {sens['frac'].iloc[0]:.0%} of factor range)" if "frac" in sens else ""
    _plot_vs_time(sens, "ratio_eta_over_x", "Finite-difference sensitivity ratio ||J_eta|| / ||J_x||" + tag, "ratio (log)", f"fig6b_sensitivity_ratio{suffix}.png", logy=True)
    _plot_vs_time(sens, "Jx_norm", "||J_x|| (state sensitivity, per unit normalized range)" + tag, "norm (log)", f"fig6c_Jx{suffix}.png", logy=True)
    _plot_vs_time(sens, "Jeta_norm", "||J_eta|| (continuous nuisance sensitivity)" + tag, "norm (log)", f"fig6d_Jeta{suffix}.png", logy=True)


def fig_pca_and_trajectories(run_m3: str, run_m2: str | None):
    from cort_mvp.analysis.common import load_stages
    meta = np.load(WS / "data_cache/reaching_v1/cf_test_v0/meta.npz")
    rel = meta["rel_goal"]; ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    fams = [np.load(WS / f"data_cache/reaching_v1/cf_test_v{v}/meta.npz")["bg_family"] for v in range(4)]
    for run, tag in ((run_m3, "m3"), (run_m2, "m2")):
        if run is None: continue
        st = [load_stages(run, f"cf_test_v{v}") for v in range(4)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for j, t in enumerate(("t=0.0", "t=0.5", "t=1.0")):
            Z = np.concatenate([s[t] for s in st]); Zc = Z - Z.mean(0)
            U, S, Vt = np.linalg.svd(Zc, full_matrices=False); P = Zc @ Vt[:2].T
            n = len(rel)
            axes[0, j].scatter(P[:n, 0], P[:n, 1], c=ang, cmap="hsv", s=4); axes[0, j].set_title(f"{tag} {t}: colored by rel-goal angle (state)")
            axes[1, j].scatter(P[:, 0], P[:, 1], c=np.concatenate(fams), cmap="tab10", s=4, vmin=0, vmax=9); axes[1, j].set_title(f"{tag} {t}: colored by background family (nuisance)")
        plt.suptitle(f"PCA of cf_test representations ({run}) — qualitative only"); plt.tight_layout(); plt.savefig(FIG / f"fig6_pca_{tag}.png", dpi=100); plt.close()
        # trajectories for 6 matched counterfactual states
        d = np.load(WS / "results/representations" / run / "cf_test_v0.npz")
        if "traj" not in d.files: continue
        trajs = [np.load(WS / "results/representations" / run / f"cf_test_v{v}.npz")["traj"] for v in range(4)]
        allpts = np.concatenate([tr.reshape(-1, tr.shape[-1]) for tr in trajs]); mu = allpts.mean(0)
        _, _, Vt = np.linalg.svd(allpts - mu, full_matrices=False)
        fig, ax = plt.subplots(figsize=(8, 7))
        cm = plt.get_cmap("tab10")
        for k, i in enumerate(range(0, 60, 10)):
            for v in range(4):
                P = (trajs[v][:, i] - mu) @ Vt[:2].T
                ax.plot(P[:, 0], P[:, 1], color=cm(k), alpha=0.8, lw=1, marker="." if v == 0 else None, ms=4)
                ax.scatter(P[0, 0], P[0, 1], color=cm(k), marker="o", s=25, edgecolor="k", zorder=3)
                ax.scatter(P[-1, 0], P[-1, 1], color=cm(k), marker="*", s=70, edgecolor="k", zorder=3)
        ax.set_title(f"{tag}: learned ODE trajectories for 6 states × 4 nuisance variants (PCA-2 of all points)\n○ = z(0), ★ = z(1); same color = same physical state")
        plt.tight_layout(); plt.savefig(FIG / f"fig8_trajectories_{tag}.png", dpi=110); plt.close()


def fig_step_sweep(df: pd.DataFrame):
    d = df[df.split.str.contains("@steps")].copy()
    if d.empty: return
    d["eval_split"] = d.split.str.split("@").str[0]; d["steps"] = d.split.str.extract(r"steps(\d+)").astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, sp in zip(axes, ("id_test", "ood_bg", "geo_ood")):
        extra = {"ablB_endpoint_emb": ("B: endpoint + ||z1 - Wa||² (no FM)", "tab:purple"), "ablB_fm_only": ("B: FM only (no endpoint loss)", "tab:orange"),
                 "ablF_ot_pairing": ("F: FM + endpoint, OT re-pairing", "tab:brown")}
        for e in ("m2", "m3", "m4", *extra):
            dd = d[(d.exp == e) & (d.eval_split == sp)].groupby("steps").mse.agg(["mean", "std"]).reset_index()
            if dd.empty: continue
            lab, col = (LABELS[e], COLORS[e]) if e in LABELS else extra[e]
            ax.errorbar(dd.steps, dd["mean"], yerr=dd["std"].fillna(0), marker="o", label=lab, color=col, capsize=2, ls="-" if e in LABELS else "--")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("evaluation Euler steps (trained with 8)"); ax.set_ylabel("action MSE (log)"); ax.set_title(sp); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "fig9_ode_step_sweep.png", dpi=110); plt.close()


def fig_grad_cosine():
    runs = sorted(p for p in RUN_ROOT.glob("m3_linear_dinov3_vits16_s*") if (p / "curves.csv").exists() and "nsteps" not in p.name)
    if not runs: return
    fig, ax = plt.subplots(figsize=(7, 4))
    for rd in runs:
        c = pd.read_csv(rd / "curves.csv")
        if "grad_cos_fm_action" not in c: continue
        cc = c.dropna(subset=["grad_cos_fm_action"])
        ax.plot(cc.step, cc.grad_cos_fm_action, marker=".", label=rd.name)
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("step"); ax.set_ylabel("cos(∇L_fm, ∇L_action)"); ax.set_title("M3: gradient alignment between FM and endpoint objectives"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "fig10_grad_cosine.png", dpi=110); plt.close()


def main():
    MET.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    df = collect_action_metrics(); summary = summarize(df)
    all_exps = sorted(summary.exp.unique())
    table_main(summary, ["m0_ridge_dinov3_vits16_cls", "m0_ridge_dinov3_vits16_patch_mean", "m0_ridge_dinov2_vits14_cls", "m1", "m2", "m3", "m4"], "table_main_action")
    table_main(summary, [e for e in all_exps if e not in PRIMARY and not e.startswith("m0")], "table_ablations_action")
    fig_action_errors(summary); fig_step_sweep(df); fig_grad_cosine(); table_step_sweep(df); tables_mechanism()
    for name, fn in (("probes", fig_probes), ("invariance", fig_invariance), ("subspaces", fig_subspaces), ("sensitivity", fig_sensitivity)):
        p = MET / f"{name}.csv"
        if p.exists():
            fn(pd.read_csv(p))
    p = MET / "sensitivity_frac0.1.csv"
    if p.exists():
        fig_sensitivity(pd.read_csv(p), suffix="_frac0.1")
    m3 = sorted(p.name for p in RUN_ROOT.glob("m3_linear_dinov3_vits16_s0")); m2 = sorted(p.name for p in RUN_ROOT.glob("m2_linear_dinov3_vits16_s0"))
    if m3 and (WS / "results/representations" / m3[0]).exists():
        fig_pca_and_trajectories(m3[0], m2[0] if m2 else None)
    print("tables:", sorted(p.name for p in TAB.iterdir())); print("figures:", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
