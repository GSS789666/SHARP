"""Aggregate every run under ``results/main/`` and produce paper-grade figures.

The earlier version of this script reported the final-iter metrics of each
training run. That hides the well-known SafeRL phenomenon where the policy
oscillates along the safety-efficiency Pareto front: the LAST iter can be far
from a representative operating point. We therefore extract two
complementary aggregates per run:

* **Pareto-best iter** — the training iteration that minimises the per-task
  collision rate (``cost / max(done, 1)``) among iters with ``done >=
  MIN_DONE``. This matches the standard SafeRL reporting convention (Achiam
  2017, Stooke 2020).
* **Last-30 mean** — the rolling mean of (cost, done) over the last 30
  iterations. Smoother but pessimistic when collapse strikes late.

Outputs (in ``results/main/_aggregate/``):

* ``summary.csv`` — one row per (scenario, algo, seed) with all aggregates.
* ``pareto.png`` — Pareto front, log-x collision axis.
* ``per_task_collision.png`` — bar chart of cost-per-task by algo/scenario.
* ``training_curves.png`` — cost trajectory for every learned run.
* ``significance.txt`` — Welch t-tests of SHARP vs each baseline on cost.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from glob import glob
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from scipy import stats


MAIN: str = os.path.join(ROOT, "results", "main")
OUT: str = os.path.join(MAIN, "_aggregate")
os.makedirs(OUT, exist_ok=True)

# Minimum number of tasks an episode must complete for the iter to count as
# a viable Pareto candidate; below this the policy has collapsed to wait.
MIN_DONE: int = 20

# Window size for the rolling-mean aggregate.
LAST_N: int = 30

# Publication-grade DPI.
DPI: int = 320

# SCI top-journal colour palette (Okabe-Ito colour-blind safe, mapped to
# the four algorithms compared in the paper).
COLORS: dict[str, str] = {
    "sharp": "#0072B2",          # Okabe-Ito blue
    "lagrangian_ppo": "#D55E00", # vermillion
    "caht": "#009E73",           # bluish green
    "milp": "#CC79A7",           # reddish purple
}
MARKERS: dict[str, str] = {
    "sharp": "o",
    "lagrangian_ppo": "s",
    "caht": "^",
    "milp": "D",
}
# Display names — algorithm acronyms are upper-cased per house style.
DISPLAY: dict[str, str] = {
    "sharp": "SHARP",
    "lagrangian_ppo": "Lagrangian-PPO",
    "caht": "CAHT",
    "milp": "MILP",
}


def _set_pub_style() -> None:
    """Apply publication-style matplotlib rcParams."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#222222",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#888888",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "lines.linewidth": 1.8,
        "savefig.dpi": DPI,
        "figure.dpi": DPI,
    })


# =====================================================================
# Data classes
# =====================================================================


@dataclass(frozen=True)
class RunMetrics:
    """All aggregate metrics for one (scenario, algo, seed) cell."""

    scenario: str
    algo: str
    seed: int
    # final-iter metrics (raw, kept for diagnostics)
    cost_final: float
    done_final: float
    # Pareto-best iter (minimises per-task collisions)
    cost_best: float
    done_best: float
    cost_per_task_best: float
    best_iter: int
    # last-LAST_N mean
    cost_last_mean: float
    cost_last_std: float
    done_last_mean: float
    done_last_std: float
    # global summary
    min_cost: float
    max_done: float
    lambda_final: float


@dataclass(frozen=True)
class BaselineMetrics:
    """Metrics for a non-learned baseline (one episode rollout per seed)."""

    scenario: str
    algo: str
    seed: int
    cost: float
    done: float
    cost_per_task: float


# =====================================================================
# Loading
# =====================================================================


def _pareto_best_iter(log: list[dict]) -> int:
    """Return the iter index that minimises per-task collision rate.

    Among iters with done >= MIN_DONE, pick the one with smallest cost/done.
    If no iter qualifies, fall back to the iter that maximises done.
    """
    viable: list[tuple[float, int]] = [
        (float(r["ep_cost_mean"]) / max(float(r["ep_done_mean"]), 1.0), i)
        for i, r in enumerate(log)
        if r.get("ep_done_mean", 0) >= MIN_DONE
    ]
    if viable:
        return min(viable)[1]
    return max(range(len(log)), key=lambda i: log[i].get("ep_done_mean", 0))


def load_learned_runs() -> list[RunMetrics]:
    rows: list[RunMetrics] = []
    for summary_path in glob(
        os.path.join(MAIN, "*", "*", "seed_*", "summary.json")
    ):
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        log_path = summary_path.replace("summary.json", "train_log.jsonl")
        if not os.path.exists(log_path):
            continue
        with open(log_path, encoding="utf-8") as fh:
            log = [json.loads(line) for line in fh if line.strip()]
        if not log:
            continue

        best_i = _pareto_best_iter(log)
        last_window = log[-LAST_N:] if len(log) >= LAST_N else log
        costs_last = np.array(
            [r["ep_cost_mean"] for r in last_window], dtype=float
        )
        dones_last = np.array(
            [r.get("ep_done_mean", 0) for r in last_window], dtype=float
        )

        rows.append(
            RunMetrics(
                scenario=summary["scenario"],
                algo=summary["algo"],
                seed=int(summary["seed"]),
                cost_final=float(log[-1]["ep_cost_mean"]),
                done_final=float(log[-1].get("ep_done_mean", 0)),
                cost_best=float(log[best_i]["ep_cost_mean"]),
                done_best=float(log[best_i].get("ep_done_mean", 0)),
                cost_per_task_best=(
                    float(log[best_i]["ep_cost_mean"])
                    / max(float(log[best_i].get("ep_done_mean", 0)), 1.0)
                ),
                best_iter=int(best_i),
                cost_last_mean=float(costs_last.mean()),
                cost_last_std=float(costs_last.std()),
                done_last_mean=float(dones_last.mean()),
                done_last_std=float(dones_last.std()),
                min_cost=float(min(r["ep_cost_mean"] for r in log)),
                max_done=float(
                    max(r.get("ep_done_mean", 0) for r in log)
                ),
                lambda_final=float(log[-1].get("lambda", float("nan"))),
            )
        )
    return rows


def load_baseline_runs() -> list[BaselineMetrics]:
    rows: list[BaselineMetrics] = []
    for path in glob(os.path.join(MAIN, "*", "*", "seed_*", "episode.json")):
        with open(path, encoding="utf-8") as fh:
            episodes = json.load(fh)
        if not episodes:
            continue
        parts = path.split(os.sep)
        scenario, algo, seed_dir = parts[-4], parts[-3], parts[-2]
        seed = int(seed_dir.replace("seed_", ""))
        for ep in episodes:
            cost = float(ep.get("cost", float("nan")))
            done = float(ep.get("tasks_done", 0))
            rows.append(
                BaselineMetrics(
                    scenario=scenario,
                    algo=algo,
                    seed=seed,
                    cost=cost,
                    done=done,
                    cost_per_task=cost / max(done, 1.0),
                )
            )
    return rows


# =====================================================================
# Aggregation helpers
# =====================================================================


def to_df(
    learned: Iterable[RunMetrics], baselines: Iterable[BaselineMetrics]
) -> pd.DataFrame:
    rows: list[dict] = [asdict(r) for r in learned]
    # Convert baselines to the same shape as learned rows for downstream code
    # by mirroring the *_best fields.
    for b in baselines:
        rows.append(
            {
                "scenario": b.scenario,
                "algo": b.algo,
                "seed": b.seed,
                "cost_final": b.cost,
                "done_final": b.done,
                "cost_best": b.cost,
                "done_best": b.done,
                "cost_per_task_best": b.cost_per_task,
                "best_iter": -1,
                "cost_last_mean": b.cost,
                "cost_last_std": 0.0,
                "done_last_mean": b.done,
                "done_last_std": 0.0,
                "min_cost": b.cost,
                "max_done": b.done,
                "lambda_final": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["scenario", "algo"])
        .agg(
            cost_best_mean=("cost_best", "mean"),
            cost_best_std=("cost_best", "std"),
            done_best_mean=("done_best", "mean"),
            done_best_std=("done_best", "std"),
            cpt_mean=("cost_per_task_best", "mean"),
            cpt_std=("cost_per_task_best", "std"),
        )
        .round(2)
    )


# =====================================================================
# Plots
# =====================================================================


def plot_pareto(df: pd.DataFrame, out_path: str) -> None:
    _set_pub_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for algo in ["sharp", "lagrangian_ppo", "caht", "milp"]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        ax.scatter(
            sub["cost_best"],
            sub["done_best"],
            c=COLORS[algo],
            marker=MARKERS[algo],
            s=110,
            alpha=0.85,
            label=DISPLAY[algo],
            edgecolors="#222222",
            linewidths=0.7,
        )
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Episode collisions at Pareto-best iter (symmetric log)")
    ax.set_ylabel("Tasks completed at Pareto-best iter")
    ax.set_title("Safety vs efficiency Pareto front")
    ax.grid(alpha=0.25, which="both", linestyle="--", linewidth=0.6)
    ax.legend(loc="lower right", title="Algorithm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_per_task_bar(df: pd.DataFrame, out_path: str) -> None:
    _set_pub_style()
    pivot = (
        df.groupby(["scenario", "algo"])["cost_per_task_best"]
        .agg(["mean", "std"])
        .reset_index()
    )
    scenarios = sorted(pivot["scenario"].unique())
    algos = ["sharp", "lagrangian_ppo", "caht", "milp"]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(scenarios))
    width = 0.2
    for i, algo in enumerate(algos):
        means: list[float] = []
        stds: list[float] = []
        for scn in scenarios:
            row = pivot[(pivot["scenario"] == scn) & (pivot["algo"] == algo)]
            means.append(float(row["mean"].iloc[0]) if len(row) else 0.0)
            stds.append(float(row["std"].iloc[0]) if len(row) else 0.0)
        ax.bar(
            x + i * width - 1.5 * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=DISPLAY[algo],
            color=COLORS[algo],
            edgecolor="#222222",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylabel("Collisions per completed task (log scale)")
    ax.set_title("Per-task collision rate — lower is safer")
    ax.grid(alpha=0.25, axis="y", which="both", linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", ncol=2, title="Algorithm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(learned: list[RunMetrics], out_path: str) -> None:
    _set_pub_style()
    fig, (ax_done, ax_cost) = plt.subplots(
        1, 2, figsize=(11.0, 4.2), sharex=True
    )
    seen: set[str] = set()
    for r in learned:
        log_path = os.path.join(
            MAIN, r.scenario, r.algo, f"seed_{r.seed}", "train_log.jsonl"
        )
        if not os.path.exists(log_path):
            continue
        with open(log_path, encoding="utf-8") as fh:
            log = [json.loads(line) for line in fh if line.strip()]
        if not log:
            continue
        iters = [d["iter"] for d in log]
        dones = [d.get("ep_done_mean", 0) for d in log]
        costs = [d["ep_cost_mean"] for d in log]
        color = COLORS.get(r.algo, "gray")
        label = DISPLAY.get(r.algo, r.algo) if r.algo not in seen else None
        seen.add(r.algo)
        ax_done.plot(iters, dones, color=color, alpha=0.45, lw=1.2,
                     label=label)
        ax_cost.plot(iters, costs, color=color, alpha=0.45, lw=1.2)

    ax_done.set_title("Tasks completed per episode")
    ax_done.set_xlabel("Training iteration")
    ax_done.set_ylabel("Tasks done")
    ax_done.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax_done.legend(loc="upper right", title="Algorithm")
    ax_cost.set_title("Episode collisions (cost)")
    ax_cost.set_xlabel("Training iteration")
    ax_cost.set_ylabel("Collisions per episode")
    ax_cost.set_yscale("symlog", linthresh=1)
    ax_cost.grid(alpha=0.25, which="both", linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Significance
# =====================================================================


def write_significance(df: pd.DataFrame, out_path: str) -> None:
    lines: list[str] = [
        "Statistical significance — Welch t-tests on per-task collision rate\n",
        f"(Pareto-best iter, min_done = {MIN_DONE})\n",
    ]
    for scen in sorted(df["scenario"].unique()):
        lines.append(f"\nScenario: {scen}")
        sub = df[df["scenario"] == scen]
        sharp_metric = sub.loc[
            sub["algo"] == "sharp", "cost_per_task_best"
        ].dropna()
        for algo in sub["algo"].unique():
            if algo == "sharp":
                continue
            other = sub.loc[
                sub["algo"] == algo, "cost_per_task_best"
            ].dropna()
            if len(sharp_metric) < 2 or len(other) < 2:
                continue
            t, p = stats.ttest_ind(sharp_metric, other, equal_var=False)
            diff = sharp_metric.mean() - other.mean()
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            lines.append(
                f"  cpt(sharp) vs cpt({algo:14s}): "
                f"mean_diff={diff:+8.3f}  t={t:+7.3f}  p={p:.4f}  {sig}"
            )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# =====================================================================
# Entry point
# =====================================================================


def main() -> None:
    learned = load_learned_runs()
    baselines = load_baseline_runs()
    if not learned and not baselines:
        sys.exit(
            "No runs found under results/main/. Run "
            "scripts\\run_all_main.bat and scripts\\run_all_baselines.bat first."
        )
    df = to_df(learned, baselines)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(f"[analyze] saved {OUT}/summary.csv  ({len(df)} rows)")

    table = summary_table(df)
    print("\n=== mean ± std per (scenario, algo), Pareto-best iter ===")
    print(table.to_string())
    table.to_csv(os.path.join(OUT, "summary_grouped.csv"))

    plot_pareto(df, os.path.join(OUT, "pareto.png"))
    print(f"[analyze] saved {OUT}/pareto.png")
    plot_per_task_bar(df, os.path.join(OUT, "per_task_collision.png"))
    print(f"[analyze] saved {OUT}/per_task_collision.png")
    plot_training_curves(learned, os.path.join(OUT, "training_curves.png"))
    print(f"[analyze] saved {OUT}/training_curves.png")
    write_significance(df, os.path.join(OUT, "significance.txt"))
    print(f"[analyze] saved {OUT}/significance.txt")

    print("\n=== aggregate done ===")


if __name__ == "__main__":
    main()
