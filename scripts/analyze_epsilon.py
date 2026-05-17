"""Aggregate the epsilon-sweep runs and produce the scaling figure.

Uses the same Pareto-best-iter convention as ``analyze.py`` so the ε
sweep can be reported in the same units as the main experiments.

Outputs (under ``results/eps_sweep/_aggregate/``):

* ``eps_summary.csv`` — one row per (scenario, ε).
* ``eps_scaling.png`` — best-iter cost & per-task collision vs ε (log-log).
* ``eps_training_curves.png`` — per-iter cost trajectory by ε.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd


SWEEP: str = os.path.join(ROOT, "results", "eps_sweep")
OUT: str = os.path.join(SWEEP, "_aggregate")
os.makedirs(OUT, exist_ok=True)

MIN_DONE: int = 20
EPISODE_STEPS: int = 100  # used for the "target line" in the scaling plot

DPI: int = 320

# Scenario colours (Okabe-Ito subset for two-scenario plot).
SCENARIO_COLORS: dict[str, str] = {
    "S40_low": "#0072B2",
    "S60_high": "#D55E00",
}


def _set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
        "savefig.dpi": DPI,
        "figure.dpi": DPI,
    })


# =====================================================================
# Data class
# =====================================================================


@dataclass(frozen=True)
class EpsRun:
    scenario: str
    epsilon: float
    cost_best: float
    done_best: float
    cost_per_task_best: float
    best_iter: int
    cost_final: float
    done_final: float
    lambda_final: float
    log_path: str


def _parse_epsilon(path: str) -> float:
    m = re.search(r"eps([\d.]+)", path)
    return float(m.group(1)) if m else float("nan")


def _pareto_best_iter(log: list[dict]) -> int:
    viable: list[tuple[float, int]] = [
        (float(r["ep_cost_mean"]) / max(float(r.get("ep_done_mean", 0)), 1.0), i)
        for i, r in enumerate(log)
        if r.get("ep_done_mean", 0) >= MIN_DONE
    ]
    if viable:
        return min(viable)[1]
    return max(range(len(log)), key=lambda i: log[i].get("ep_done_mean", 0))


def load_runs() -> list[EpsRun]:
    rows: list[EpsRun] = []
    for summary_path in glob(
        os.path.join(SWEEP, "eps*", "*", "sharp", "seed_*", "summary.json")
    ):
        with open(summary_path, encoding="utf-8") as fh:
            _ = json.load(fh)
        log_path = summary_path.replace("summary.json", "train_log.jsonl")
        if not os.path.exists(log_path):
            continue
        with open(log_path, encoding="utf-8") as fh:
            log = [json.loads(line) for line in fh if line.strip()]
        if not log:
            continue

        eps = _parse_epsilon(summary_path)
        scenario = summary_path.split(os.sep)[-4]
        best_i = _pareto_best_iter(log)
        rows.append(
            EpsRun(
                scenario=scenario,
                epsilon=eps,
                cost_best=float(log[best_i]["ep_cost_mean"]),
                done_best=float(log[best_i].get("ep_done_mean", 0)),
                cost_per_task_best=(
                    float(log[best_i]["ep_cost_mean"])
                    / max(float(log[best_i].get("ep_done_mean", 0)), 1.0)
                ),
                best_iter=int(best_i),
                cost_final=float(log[-1]["ep_cost_mean"]),
                done_final=float(log[-1].get("ep_done_mean", 0)),
                lambda_final=float(log[-1].get("lambda", float("nan"))),
                log_path=log_path,
            )
        )
    return rows


# =====================================================================
# Plots
# =====================================================================


def plot_scaling(df: pd.DataFrame, out_path: str) -> None:
    _set_pub_style()
    fig, (ax_cost, ax_cpt) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    for scen in df["scenario"].unique():
        sub = df[df["scenario"] == scen].sort_values("epsilon")
        color = SCENARIO_COLORS.get(scen, "#000000")
        ax_cost.plot(
            sub["epsilon"], sub["cost_best"], "o-", label=scen,
            color=color, lw=2.0, ms=8, markeredgecolor="#222222",
            markeredgewidth=0.6,
        )
        ax_cpt.plot(
            sub["epsilon"], sub["cost_per_task_best"], "o-", label=scen,
            color=color, lw=2.0, ms=8, markeredgecolor="#222222",
            markeredgewidth=0.6,
        )

    eps_grid = np.array(sorted(df["epsilon"].unique()))
    ax_cost.plot(
        eps_grid,
        eps_grid * EPISODE_STEPS,
        ls="--",
        color="#555555",
        alpha=0.6,
        label="ε × episode_steps",
    )

    for ax, ylabel, title in (
        (ax_cost, "Episode collisions (Pareto-best iter)",
         "Achieved cost vs ε budget"),
        (ax_cpt, "Collisions per task (Pareto-best iter)",
         "Per-task safety scaling"),
    ):
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("ε (per-step collision budget)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both", linestyle="--", linewidth=0.6)
        ax.legend(loc="best", title="Scenario")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(rows: list[EpsRun], out_path: str) -> None:
    _set_pub_style()
    scenarios = sorted({r.scenario for r in rows})
    fig, axes = plt.subplots(1, len(scenarios),
                              figsize=(5.5 * len(scenarios), 4.2),
                              sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    eps_cmap = {0.05: "#0072B2", 0.01: "#009E73",
                0.005: "#D55E00", 0.001: "#CC79A7"}
    for ax, scen in zip(axes, scenarios):
        for r in [x for x in rows if x.scenario == scen]:
            with open(r.log_path, encoding="utf-8") as fh:
                log = [json.loads(line) for line in fh if line.strip()]
            iters = [d["iter"] for d in log]
            costs = [d["ep_cost_mean"] for d in log]
            color = eps_cmap.get(round(r.epsilon, 4), "#444444")
            ax.plot(iters, costs, lw=1.8, label=f"ε = {r.epsilon:.3f}",
                    color=color)
        ax.set_title(scen)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Episode collisions")
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(alpha=0.25, which="both", linestyle="--", linewidth=0.6)
        ax.legend(loc="best", title="ε")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Entry point
# =====================================================================


def main() -> None:
    rows = load_runs()
    if not rows:
        sys.exit(
            f"No runs found under {SWEEP}/. "
            "Run scripts\\run_epsilon_sweep.bat first."
        )

    df = pd.DataFrame([asdict(r) for r in rows]).drop(columns=["log_path"])
    df = df.sort_values(["scenario", "epsilon"])
    df.to_csv(os.path.join(OUT, "eps_summary.csv"), index=False)
    print("=== epsilon sweep summary ===")
    print(df.to_string(index=False))

    plot_scaling(df, os.path.join(OUT, "eps_scaling.png"))
    print(f"[analyze_eps] saved {OUT}/eps_scaling.png")

    plot_training_curves(rows, os.path.join(OUT, "eps_training_curves.png"))
    print(f"[analyze_eps] saved {OUT}/eps_training_curves.png")

    print("\n=== epsilon-sweep aggregate done ===")


if __name__ == "__main__":
    main()
