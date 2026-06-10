from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catch_robot import CatchRunner, RunConfig, ThrowConfig


class ProgressCSV:
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        self._fp = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=fields)
        self._writer.writeheader()
        self._fp.flush()

    def write(self, row: dict):
        self._writer.writerow(row)
        self._fp.flush()  # critical: tail -f sees rows now

    def close(self):
        self._fp.close()


def banner(text: str):
    bar = "=" * 70
    print(f"\n{bar}\n  {text}\n{bar}", flush=True)


def fmt_dur(s: float) -> str:
    if s < 60: return f"{s:.0f} s"
    if s < 3600: return f"{s/60:.1f} min"
    return f"{s/3600:.2f} h"


# Single-throw runner (shared by all experiments)
def one_throw(*, vx: float, vy: float, vz: float,
              predictor: str = "analytical",
              use_gt: bool = True,
              vision_noise: float = 0.01,
              duration: float = 1.5,
              learned_ckpt: str | None = None,
              spin=(0., 0., 0.)) -> dict:
    cfg = RunConfig(
        duration_s=duration,
        predictor=predictor,
        learned_ckpt=learned_ckpt,
        use_gt_perception=use_gt,
        vision_noise_m=vision_noise,
        throw=ThrowConfig(init_pos=[0.0, 1.5, 1.7],
                          init_vel=[float(vx), float(vy), float(vz)],
                          spin=list(spin)),
    )
    runner = CatchRunner(cfg)
    res = runner.run()
    runner.close()
    return {
        "vx": vx, "vy": vy, "vz": vz,
        "predictor": predictor,
        "use_gt": use_gt,
        "vision_noise_m": vision_noise,
        "caught": int(res.caught),
        "min_dist_m": res.min_dist_m,
        "min_dist_t": res.min_dist_t,
        "t_first_detection": res.t_first_detection,
        "t_first_intercept": res.t_first_intercept,
        "n_detections": res.n_detections,
        "n_vision_frames": res.n_vision_frames,
        "wall_time_s": res.wall_time_s,
        "final_ball_z": res.final_ball_pos[2],
    }


# Experiment A — velocity sweep
def exp_A(out_dir: Path, full: bool, use_gt: bool = True) -> dict:
    banner("Experiment A — throw-velocity sweep")
    if full:
        vxs = [-0.2, 0.0, 0.2]
        vys = [-2.0, -2.3, -2.5, -2.7, -3.0]
        vzs = [0.5, 1.0, 1.5, 2.0]
    else:
        vxs = [0.0]
        vys = [-2.0, -2.5, -3.0]
        vzs = [0.8, 1.5, 2.0]
    combos = list(product(vxs, vys, vzs))
    print(f"  {len(combos)} throws, predictor=analytical, " f"perception={'GT+noise' if use_gt else 'full vision'}")

    pcsv = ProgressCSV(out_dir / "progress_A.csv",
                       ["i", "vx", "vy", "vz", "caught", "min_dist_cm",
                        "t_first_intercept_ms", "wall_time_s"])
    t0 = time.perf_counter()
    rows = []
    for i, (vx, vy, vz) in enumerate(combos):
        r = one_throw(vx=vx, vy=vy, vz=vz, use_gt=use_gt)
        rows.append(r)
        pcsv.write({
            "i": i + 1,
            "vx": f"{vx:+.2f}", "vy": f"{vy:+.2f}", "vz": f"{vz:+.2f}",
            "caught": r["caught"],
            "min_dist_cm": f"{r['min_dist_m']*100:.2f}",
            "t_first_intercept_ms": (f"{r['t_first_intercept']*1000:.0f}"
                                     if r["t_first_intercept"] else ""),
            "wall_time_s": f"{r['wall_time_s']:.1f}",
        })
        if (i + 1) % 5 == 0 or i + 1 == len(combos):
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            n_caught = sum(r["caught"] for r in rows)
            print(f"  [{i+1:3d}/{len(combos)}]  "
                  f"caught {n_caught}/{i+1} ({100*n_caught/(i+1):.0f}%)  "
                  f"elapsed {fmt_dur(elapsed)}  ETA {fmt_dur(eta)}",
                  flush=True)
    pcsv.close()

    n_caught = sum(r["caught"] for r in rows)
    summary = {
        "vxs": vxs, "vys": vys, "vzs": vzs,
        "n_runs": len(rows),
        "n_caught": n_caught,
        "catch_rate": n_caught / len(rows),
        "mean_min_dist_cm": float(np.mean([r["min_dist_m"] for r in rows]) * 100),
        "median_min_dist_cm": float(np.median([r["min_dist_m"] for r in rows]) * 100),
        "mean_first_intercept_ms": float(np.mean(
            [r["t_first_intercept"] * 1000 for r in rows
             if r["t_first_intercept"] is not None])),
        "rows": rows,
    }
    plot_velocity_heatmap(rows, vxs, vys, vzs, out_dir / "catch_rate_by_velocity.png")
    print(f"  → catch rate {n_caught}/{len(rows)} = "
          f"{summary['catch_rate']*100:.0f}%   "
          f"mean min-dist {summary['mean_min_dist_cm']:.1f} cm")
    return summary


def plot_velocity_heatmap(rows, vxs, vys, vzs, out_path):
    rate = np.zeros((len(vys), len(vzs)))
    count = np.zeros_like(rate)
    for r in rows:
        i = vys.index(r["vy"]); j = vzs.index(r["vz"])
        rate[i, j] += float(r["caught"]); count[i, j] += 1
    rate = rate / np.maximum(count, 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rate, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(vzs))); ax.set_xticklabels([f"{v:+.1f}" for v in vzs])
    ax.set_yticks(range(len(vys))); ax.set_yticklabels([f"{v:+.1f}" for v in vys])
    ax.set_xlabel("initial v_z  (m/s)")
    ax.set_ylabel("initial v_y  (m/s, toward robot)")
    ax.set_title(f"Catch rate by initial velocity (averaged over v_x={vxs})")
    for i in range(len(vys)):
        for j in range(len(vzs)):
            ax.text(j, i, f"{rate[i,j]*100:.0f}%", ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, label="catch rate")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# Experiment B — vision noise robustness
def exp_B(out_dir: Path, full: bool) -> dict:
    banner("Experiment B — vision-noise robustness")
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
    n_each = 12 if full else 5
    total = len(noise_levels) * n_each
    print(f"  {total} throws ({n_each} per noise level)")

    pcsv = ProgressCSV(out_dir / "progress_B.csv",
                       ["i", "sigma_mm", "vx", "vy", "vz",
                        "caught", "min_dist_cm"])
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    rows = []
    base_vel = (0.0, -2.5, 1.5)
    i = 0
    for sigma in noise_levels:
        for _ in range(n_each):
            jitter = rng.normal(scale=[0.05, 0.05, 0.05])
            v = np.array(base_vel) + jitter
            r = one_throw(vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
                          vision_noise=float(sigma))
            r["sigma_m"] = float(sigma)
            rows.append(r)
            i += 1
            pcsv.write({
                "i": i, "sigma_mm": f"{sigma*1000:.1f}",
                "vx": f"{v[0]:+.2f}", "vy": f"{v[1]:+.2f}",
                "vz": f"{v[2]:+.2f}",
                "caught": r["caught"],
                "min_dist_cm": f"{r['min_dist_m']*100:.2f}",
            })
        n_caught_lvl = sum(1 for r in rows
                           if abs(r["sigma_m"] - sigma) < 1e-9 and r["caught"])
        print(f"  σ = {sigma*1000:5.1f} mm   caught {n_caught_lvl}/{n_each}  "
              f"elapsed {fmt_dur(time.perf_counter()-t0)}", flush=True)
    pcsv.close()

    per_level = {}
    for sigma in noise_levels:
        these = [r for r in rows if abs(r["sigma_m"] - sigma) < 1e-9]
        per_level[float(sigma)] = {
            "catch_rate": float(np.mean([r["caught"] for r in these])),
            "median_dist_cm": float(np.median([r["min_dist_m"] * 100 for r in these])),
            "mean_dist_cm": float(np.mean([r["min_dist_m"] * 100 for r in these])),
        }
    plot_error_vs_noise(rows, out_dir / "error_vs_noise.png")
    return {"noise_levels_m": noise_levels, "n_per_level": n_each,
            "per_level": per_level, "rows": rows}


def plot_error_vs_noise(rows, out_path):
    sigmas = sorted(set(r["sigma_m"] for r in rows))
    data = [[r["min_dist_m"] * 100 for r in rows if r["sigma_m"] == s]
            for s in sigmas]
    catch_rates = [100 * np.mean([r["caught"] for r in rows if r["sigma_m"] == s])
                   for s in sigmas]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    bp = ax1.boxplot(data, positions=range(len(sigmas)), widths=0.6,
                     patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("C0"); patch.set_alpha(0.6)
    ax1.set_xticks(range(len(sigmas)))
    ax1.set_xticklabels([f"{s*1000:.0f}" for s in sigmas])
    ax1.set_xlabel("vision noise σ (mm)")
    ax1.set_ylabel("min ball–EE distance (cm)", color="C0")
    ax1.axhline(10.0, ls="--", color="C0", alpha=0.4, label="catch threshold")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(range(len(sigmas)), catch_rates, "C3o-", lw=2)
    ax2.set_ylabel("catch rate (%)", color="C3"); ax2.set_ylim(0, 105)
    ax1.set_title("Robustness to vision noise")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# Experiment C — intercept-decision timeline
def exp_C(out_dir: Path, vel=(0.0, -2.5, 1.5)) -> dict:
    banner("Experiment C — intercept-decision timeline (single throw)")
    cfg = RunConfig(duration_s=1.2, use_gt_perception=True, vision_noise_m=0.01, throw=ThrowConfig(init_pos=[0.0, 1.5, 1.7], init_vel=list(vel)))
    runner = CatchRunner(cfg)
    res = runner.run()
    print(f"  caught={bool(res.caught)}  "
          f"min_dist={res.min_dist_m*100:.1f} cm  "
          f"#intercept_commits={len(runner.log_intercept)}")

    t = np.array(runner.log_t); gt = np.array(runner.log_gt_pos)
    ee = np.array(runner.log_ee_pos); d = np.array(runner.log_dist) * 100
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axs[0].plot(t, gt[:, 0], "k-",  label="ball x")
    axs[0].plot(t, gt[:, 1], "k--", label="ball y")
    axs[0].plot(t, gt[:, 2], "k:",  label="ball z")
    axs[0].plot(t, ee[:, 0], "C0-",  alpha=0.7, label="EE x")
    axs[0].plot(t, ee[:, 1], "C0--", alpha=0.7, label="EE y")
    axs[0].plot(t, ee[:, 2], "C0:",  alpha=0.7, label="EE z")
    if runner.log_intercept:
        tc = np.array([row[1] for row in runner.log_intercept])
        pc = np.array([row[2] for row in runner.log_intercept])
        axs[0].scatter(tc, pc[:, 0], marker="x", c="C3", s=24, label="catch pt x")
        axs[0].scatter(tc, pc[:, 1], marker="x", c="C2", s=24, label="catch pt y")
        axs[0].scatter(tc, pc[:, 2], marker="x", c="C1", s=24, label="catch pt z")
    axs[0].set_ylabel("position (m)")
    axs[0].set_title("Ball / EE / committed catch-point trajectories")
    axs[0].legend(ncol=3, fontsize=8); axs[0].grid(alpha=0.3)
    axs[1].plot(t, d, "C0-", lw=1.5)
    axs[1].axhline(10.0, ls="--", color="r", alpha=0.5, label="catch threshold")
    axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("ball–EE distance (cm)")
    axs[1].grid(alpha=0.3); axs[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "intercept_timeline.png", dpi=120); plt.close(fig)
    runner.close()
    return {
        "throw_velocity": list(vel),
        "caught": bool(res.caught),
        "min_dist_cm": res.min_dist_m * 100,
        "n_intercept_commits": len(runner.log_intercept),
    }


# Experiment D — predictor ablation
def exp_D(out_dir: Path, ckpt: str | None, full: bool) -> dict:
    banner("Experiment D — predictor ablation (analytical vs learned LSTM)")
    ckpt_path = ROOT / ckpt if ckpt else None
    if ckpt is None or not ckpt_path.exists():
        print(f"  WARNING: no checkpoint at '{ckpt}'.")
        print(f"           LearnedPredictor will silently fall back to "
              f"analytical, so the comparison is meaningless.")
        print(f"           Run `python scripts/collect_data.py && "
              f"python scripts/train_predictor.py` first.")
        return {"skipped": True, "reason": "no_checkpoint"}

    if full:
        combos = list(product([0.0], [-2.3, -2.5, -2.7], [1.0, 1.5, 2.0]))
    else:
        combos = list(product([0.0], [-2.5, -3.0], [1.5, 2.0]))
    print(f"  {len(combos)} throws × 2 predictors = {len(combos)*2} runs")

    pcsv = ProgressCSV(out_dir / "progress_D.csv",
                       ["i", "predictor", "vy", "vz",
                        "caught", "min_dist_cm"])
    out = {"analytical": [], "learned": []}
    i = 0
    for vx, vy, vz in combos:
        for pred in ("analytical", "learned"):
            r = one_throw(vx=vx, vy=vy, vz=vz, predictor=pred, learned_ckpt=ckpt)
            i += 1
            out[pred].append(r)
            pcsv.write({"i": i, "predictor": pred,
                        "vy": f"{vy:+.2f}", "vz": f"{vz:+.2f}",
                        "caught": r["caught"],
                        "min_dist_cm": f"{r['min_dist_m']*100:.2f}"})
        n_a = sum(rr["caught"] for rr in out["analytical"])
        n_l = sum(rr["caught"] for rr in out["learned"])
        print(f"  [{(i+1)//2:3d}/{len(combos)}]  "
              f"analytical {n_a}/{len(out['analytical'])}   "
              f"learned {n_l}/{len(out['learned'])}", flush=True)
    pcsv.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    a_dist = [r["min_dist_m"] * 100 for r in out["analytical"]]
    l_dist = [r["min_dist_m"] * 100 for r in out["learned"]]
    bp = ax.boxplot([a_dist, l_dist],
                    tick_labels=["analytical (RK4+drag)", "learned (LSTM)"],
                    patch_artist=True, widths=0.6)
    for p, c in zip(bp["boxes"], ["C0", "C3"]):
        p.set_facecolor(c); p.set_alpha(0.6)
    ax.set_ylabel("min ball–EE distance (cm)")
    ax.axhline(10.0, ls="--", color="r", alpha=0.4, label="catch threshold")
    rates = [sum(r["caught"] for r in out["analytical"]) / len(out["analytical"]), sum(r["caught"] for r in out["learned"])    / len(out["learned"])]
    ax.set_title(f"Predictor ablation — analytical "
                 f"{rates[0]*100:.0f}% vs learned {rates[1]*100:.0f}%")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "predictor_ablation.png", dpi=120); plt.close(fig)
    return {"rows_analytical": out["analytical"],
            "rows_learned":    out["learned"],
            "catch_rate_analytical": rates[0],
            "catch_rate_learned":    rates[1]}


# Experiment E — real-vision spot check
def exp_E(out_dir: Path) -> dict:
    banner("Experiment E — real-vision spot check (full stereo render)")
    print(f"  WARNING: uses the actual stereo render pipeline.")
    print(f"           Each throw is ~5 s on a GPU, ~60-90 s on a CPU-only host.")
    combos = list(product([0.0], [-2.0, -2.5, -3.0], [1.5]))
    print(f"  {len(combos)} throws")

    pcsv = ProgressCSV(out_dir / "progress_E.csv",
                       ["i", "vy", "vz", "caught", "min_dist_cm",
                        "n_detections", "wall_time_s"])
    rows = []
    t0 = time.perf_counter()
    for i, (vx, vy, vz) in enumerate(combos):
        r = one_throw(vx=vx, vy=vy, vz=vz, use_gt=False, duration=1.2)
        rows.append(r)
        pcsv.write({"i": i + 1, "vy": f"{vy:+.2f}", "vz": f"{vz:+.2f}",
                    "caught": r["caught"],
                    "min_dist_cm": f"{r['min_dist_m']*100:.2f}",
                    "n_detections": r["n_detections"],
                    "wall_time_s": f"{r['wall_time_s']:.1f}"})
        print(f"  [{i+1}/{len(combos)}]  vy={vy:+.1f} vz={vz:+.1f}  "
              f"caught={r['caught']}  dist={r['min_dist_m']*100:.1f} cm  "
              f"dets={r['n_detections']}  "
              f"elapsed={fmt_dur(time.perf_counter()-t0)}", flush=True)
    pcsv.close()
    n_caught = sum(r["caught"] for r in rows)
    return {"n_runs": len(rows), "n_caught": n_caught, "catch_rate": n_caught / len(rows), "rows": rows}


# Experiment F — failure-mode breakdown
def exp_F(out_dir: Path, full: bool) -> dict:
    banner("Experiment F — failure-mode breakdown")
    if full:
        combos = list(product([0.0], [-2.0, -2.3, -2.5, -2.7, -3.0], [0.5, 1.0, 1.5, 2.0]))
    else:
        combos = list(product([0.0], [-2.0, -2.5, -3.0], [0.8, 1.5, 2.0]))
    print(f"  {len(combos)} throws, classifying every failure")

    pcsv = ProgressCSV(out_dir / "progress_F.csv",
                       ["i", "vy", "vz", "caught", "min_dist_cm",
                        "t_first_intercept_ms", "mode"])
    rows = []
    for i, (vx, vy, vz) in enumerate(combos):
        cfg = RunConfig(duration_s=1.5, use_gt_perception=True, throw=ThrowConfig(init_pos=[0.0, 1.5, 1.7], init_vel=[vx, vy, vz]))
        runner = CatchRunner(cfg)
        res = runner.run()

        if res.caught:
            mode = "caught"
        elif res.t_first_intercept is None:
            mode = "no_intercept_committed"
        elif (res.t_first_intercept is not None and res.min_dist_t < res.t_first_intercept + 0.1):
            mode = "intercept_too_late"
        elif res.min_dist_m > 0.20:
            mode = "ee_diverged"
        else:
            mode = "near_miss"

        rows.append({
            "vx": vx, "vy": vy, "vz": vz,
            "caught": res.caught,
            "min_dist_cm": res.min_dist_m * 100,
            "min_dist_t": res.min_dist_t,
            "t_first_intercept": res.t_first_intercept,
            "mode": mode,
        })
        pcsv.write({"i": i + 1, "vy": f"{vy:+.2f}", "vz": f"{vz:+.2f}",
                    "caught": int(res.caught),
                    "min_dist_cm": f"{res.min_dist_m*100:.2f}",
                    "t_first_intercept_ms": (f"{res.t_first_intercept*1000:.0f}"
                                             if res.t_first_intercept else ""),
                    "mode": mode})
        runner.close()
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(combos)}]", flush=True)
    pcsv.close()

    counts = defaultdict(int)
    for r in rows: counts[r["mode"]] += 1
    print("  failure modes:")
    for m, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {m:30s} {c:3d}  ({100*c/len(rows):.0f}%)")

    order = ["caught", "near_miss", "ee_diverged", "intercept_too_late", "no_intercept_committed"]
    palette = {"caught": "#2ecc71", "near_miss": "#f1c40f",
               "ee_diverged": "#e67e22",
               "intercept_too_late": "#e74c3c",
               "no_intercept_committed": "#95a5a6"}
    bars = [counts.get(m, 0) for m in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(order, bars, color=[palette[m] for m in order])
    for i, v in enumerate(bars):
        if v > 0:
            ax.text(i, v + 0.3, f"{v}\n({100*v/len(rows):.0f}%)",
                    ha="center", fontsize=10)
    ax.set_ylabel("count")
    ax.set_title(f"Failure-mode breakdown across {len(rows)} throws")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_modes.png", dpi=120); plt.close(fig)
    return {"counts": dict(counts), "n_runs": len(rows), "rows": rows}


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Catch robot — experiment suite")
    p.add_argument("--exp", default="default",
                   help="Which experiment(s) to run: A | B | C | D | E | F | "
                        "default (= A+B+C) | all (= A+B+C+D+F).")
    p.add_argument("--full", action="store_true",
                   help="Use the larger sweep grids (~3 min instead of ~30 s).")
    p.add_argument("--out-dir", default="data/experiments",
                   help="Where to write timestamped result folders.")
    p.add_argument("--ckpt", default="data/lstm_predictor.pt",
                   help="LSTM checkpoint for experiment D.")
    return p.parse_args()


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / args.out_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing experiments to: {out_dir}")
    print(f"  Live progress  : `tail -f {out_dir}/progress_*.csv`")
    print(f"  Plots & summary: appear here once each experiment finishes", flush=True)

    exps = args.exp.upper()
    if exps == "DEFAULT": exps = "ABC"
    elif exps == "ALL":   exps = "ABCDF"

    summary = {"stamp": stamp, "args": vars(args), "results": {}}
    t_total = time.perf_counter()

    if "A" in exps: summary["results"]["A"] = exp_A(out_dir, args.full)
    if "B" in exps: summary["results"]["B"] = exp_B(out_dir, args.full)
    if "C" in exps: summary["results"]["C"] = exp_C(out_dir)
    if "D" in exps: summary["results"]["D"] = exp_D(out_dir, args.ckpt, args.full)
    if "E" in exps: summary["results"]["E"] = exp_E(out_dir)
    if "F" in exps: summary["results"]["F"] = exp_F(out_dir, args.full)

    summary["total_wall_time_s"] = time.perf_counter() - t_total
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    banner(f"DONE  (total {fmt_dur(summary['total_wall_time_s'])})")
    print(f"  Results : {out_dir}")
    print(f"  Summary : {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
