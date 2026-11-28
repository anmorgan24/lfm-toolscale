"""Draw the article's figures from the saved results.

  python scripts/make_figures.py

Figure 1 is LFM3's layer layout (from its GGUF metadata). Figure 2 reads the laptop
latency results in results/laptop/*/*.jsonl. Figures 3 to 5 read results/gpu/summary.csv
(run scripts/score.py --machine gpu first) and the laptop results. Writes PNG and SVG
files to figures/.

Colors: LFM3 is always blue and MiniCPM5 always orange, and those two colors mean
nothing else in any figure. Red marks lookups in figure 4, the finding it exists to show.
"""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
MODELS = {"lfm3": ("LFM3-1.7B", "#2f6db5"), "minicpm5": ("MiniCPM5-2B", "#e08e3c")}
SIZES = [18, 36, 72, 108]
INK, QUIET, GRID = "#1f2328", "#57606a", "#e6e9ed"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": QUIET, "axes.labelcolor": QUIET, "xtick.color": QUIET,
    "ytick.color": QUIET, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK,
    "svg.fonttype": "none",
})


def gpu_rows():
    rows = list(csv.DictReader(open(ROOT / "results" / "gpu" / "summary.csv")))
    for r in rows:
        r["stem"] = Path(r["file"]).stem
        r["cell"] = r["stem"].rsplit("_", 2)[0] if r["stem"].endswith("_greedy") else r["stem"]
    return rows


def find(rows, model, stem):
    for r in rows:
        if r["model"] == model and r["stem"] == stem:
            return r
    raise KeyError(f"{model} {stem} not in results/gpu/summary.csv")


def titles(ax, title, subtitle=None):
    """A bold title with an optional grey line under it, both left-aligned."""
    ax.set_title(title, loc="left", pad=26 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=QUIET, fontsize=9.5)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote figures/{name}.png and .svg")


def figure1():
    """LFM3-1.7B's 24 layers by type, read from the GGUF metadata
    (lfm3.attention.head_count_kv and lfm3.attention.sliding_window_pattern)."""
    def kind(i):
        return "global" if i % 4 == 2 else ("swa" if i % 4 == 0 and i > 0 else "conv")
    styles = {"conv": dict(facecolor="white", edgecolor="#8c959f", linewidth=1.2),
              "swa": dict(facecolor="#d9cdf2", edgecolor="#9b84d6", linewidth=1.2),
              "global": dict(facecolor="#b9a3e3", edgecolor="#5b3fa8", linewidth=2)}
    fig, ax = plt.subplots(figsize=(8.5, 3.3))
    for i in range(24):
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (i * 1.12, 0), 0.95, 1.4, boxstyle="round,pad=0,rounding_size=0.12", **styles[kind(i)]))
    ax.text(0, -0.35, "layer 0", color=QUIET, fontsize=9.5, va="top")
    ax.text(23 * 1.12 + 0.95, -0.35, "layer 23", color=QUIET, fontsize=9.5, va="top", ha="right")
    legend = [("conv", "Gated short convolution (13 layers): mixes each token with the 6 before it"),
              ("swa", "Sliding-window attention (5 layers): sees the most recent 256 tokens"),
              ("global", "Global attention (6 layers): sees the whole prompt and keeps a full KV cache")]
    for j, (k, text) in enumerate(legend):
        y = -1.9 - j * 0.78
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (0, y), 0.55, 0.5, boxstyle="round,pad=0,rounding_size=0.08", **styles[k]))
        ax.text(0.85, y + 0.25, text, va="center", color=INK, fontsize=10)
    ax.set_xlim(-0.2, 24 * 1.12)
    ax.set_ylim(-3.9, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    titles(ax, "Only 6 of the 24 layers in LFM3-1.7B see the whole prompt",
           "Layer types read from the model's GGUF metadata, layer 0 at the left")
    save(fig, "fig1_lfm3_layer_layout")


def laptop_speeds(model, size):
    """Prompt throughput (mean tokens / mean seconds) and mean generation speed."""
    path = ROOT / "results" / "laptop" / model / f"{size}_latency_greedy.jsonl"
    recs = [json.loads(line) for line in open(path)]
    prompt = sum(r["prompt_tokens"] for r in recs) / (sum(r["prompt_ms"] for r in recs) / 1000)
    decode = sum(r["decode_tok_per_s"] for r in recs) / len(recs)
    return prompt, decode


def figure2():
    """Speed as the catalog grows, on the laptop: relative to 18 tools, and absolute."""
    speeds = {m: {s: laptop_speeds(m, s) for s in SIZES} for m in MODELS}
    for relative in (True, False):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=relative)
        for ax, idx, name in zip(axes, (0, 1), ("Prompt processing", "Generation")):
            for model, (label, color) in MODELS.items():
                base = speeds[model][18][idx]
                ys = [speeds[model][s][idx] / (base if relative else 1) for s in SIZES]
                ax.plot(range(len(SIZES)), ys, marker="o", color=color, linewidth=2, label=label)
                end = f"{ys[-1]:.0%}" if relative else f"{ys[-1]:.0f}"
                ax.text(len(SIZES) - 1 + 0.08, ys[-1], end, color=color, va="center", fontsize=9.5)
            ax.set_xticks(range(len(SIZES)), [str(s) for s in SIZES])
            ax.set_xlabel("tools in the catalog")
            ax.grid(axis="y", color=GRID)
            ax.set_axisbelow(True)
            if relative:
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            else:
                ax.set_ylim(0, None)
                ax.set_ylabel("tokens per second")
            note = ("Laptop CPU, 30 requests per point, each model relative to its speed at 18 tools"
                    if relative else "Laptop CPU, 30 requests per point")
            titles(ax, f"{name} speed as the catalog grows", note if idx == 0 else " ")
        axes[0].legend(frameon=False, loc="lower left")
        if relative:
            axes[0].set_ylabel("% of speed at 18 tools")
        save(fig, "fig2_speed_relative" if relative else "fig2_speed_absolute")


def figure3(rows):
    """Accuracy against catalog size, with each arrival order as its own point."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = {s: i for i, s in enumerate(SIZES)}
    for k, (model, (label, color)) in enumerate(MODELS.items()):
        offset = -0.08 if k == 0 else 0.08
        pts = [r for r in rows if r["model"] == model and re.fullmatch(r"\d+(s\d)?_all_greedy", r["stem"])]
        means = []
        for s in SIZES:
            accs = [float(r["accuracy"]) for r in pts if int(r["size"]) == s]
            means.append(sum(accs) / len(accs))
            ax.scatter([x[s] + offset] * len(accs), accs, s=28, color=color, alpha=0.55, zorder=3,
                       edgecolor="white", linewidth=0.6)
        ax.plot([x[s] + offset for s in SIZES], means, color=color, linewidth=2, label=label, zorder=2)
    ax.set_xticks(range(len(SIZES)), [str(s) for s in SIZES])
    ax.set_xlabel("tools in the catalog")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID)
    titles(ax, "Accuracy as the catalog grows",
           "Lines: average. Dots: each arrival order of the added tools (three at 36 and 72).")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig3_accuracy_by_catalog_size")


SEGMENTS = [  # (label, columns summed, color)
    ("looked something up instead", ["sub_lookup"], "#c0392b"),
    ("other wrong tool", ["substituted_call", "-sub_lookup"], "#e8a39b"),
    ("right call plus an extra one", ["spurious_call"], "#8e6bbf"),
    ("no call", ["no_call"], "#6e7781"),
    ("called a tool when none was needed", ["unneeded_call"], "#a9b4bf"),
    ("wrong arguments", ["argument_error"], "#4f9d8a"),
    ("other", ["format_error", "invented_function", "dropped_call", "extra_call"], "#d0d7de"),
]


def segment_value(row, cols):
    total = 0
    for c in cols:
        total += -int(row[c[1:]] or 0) if c.startswith("-") else int(row[c] or 0)
    return total


def figure4(rows):
    """What went wrong at 18 and 108 tools, as a share of all requests."""
    bars = [(f"{MODELS[m][0]}, {s} tools", find(rows, m, f"{s}_all_greedy"))
            for m in MODELS for s in (18, 108)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, (label, row) in enumerate(reversed(bars)):
        n = int(row["n"])
        left = 0
        for name, cols, color in SEGMENTS:
            share = segment_value(row, cols) / n
            ax.barh(i, share, left=left, color=color, edgecolor="white", linewidth=0.8,
                    label=name if i == 0 else None)
            left += share
        ax.text(left + 0.005, i, f"{left:.0%} wrong ({float(row['accuracy']):.0%} correct)",
                va="center", color=QUIET, fontsize=9)
    ax.set_yticks(range(len(bars)), [b[0] for b in reversed(bars)])
    ax.set_xlim(0, 0.45)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    titles(ax, "What went wrong, at 18 and 108 tools", "Share of all 274 requests, by type of mistake")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.12))
    save(fig, "fig4_failures_by_type")


def laptop_seconds(model, stem):
    path = ROOT / "results" / "laptop" / model / f"{stem}.jsonl"
    recs = [json.loads(line) for line in open(path)]
    return sum(r["wall_ms"] for r in recs) / len(recs) / 1000


def figure5(rows):
    """Deferred tools: accuracy by strategy (GPU) and time per request (laptop)."""
    strategies = [
        ("All 108 tools\nin context", {"lfm3": "108_all_greedy", "minicpm5": "108_all_greedy"}),
        ("Model decides\nwhether to search", {"lfm3": "deferred_all_greedy", "minicpm5": "deferred_all_greedy"}),
        ("Harness searches,\nLFM3's format", {"lfm3": "deferred_forced_all_greedy"}),
        ("Harness searches,\ntool-list format", {"lfm3": "deferred_forced_plain_all_greedy",
                                                "minicpm5": "deferred_forced_all_greedy"}),
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.3, 1]})
    width = 0.36
    for k, (model, (label, color)) in enumerate(MODELS.items()):
        for i, (_, stems) in enumerate(strategies):
            if model not in stems:
                continue
            acc = float(find(rows, model, stems[model])["accuracy"])
            xpos = i + ((k - 0.5) * width if len(stems) == 2 else 0)
            left.bar(xpos, acc, width, color=color, label=label if i == 0 else None)
            left.text(xpos, acc + 0.015, f"{acc:.0%}", ha="center", color=INK, fontsize=9)
    left.set_xticks(range(len(strategies)), [s[0] for s in strategies], fontsize=9.5)
    left.set_ylim(0, 1)
    left.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    left.grid(axis="y", color=GRID)
    left.set_axisbelow(True)
    titles(left, "Accuracy at 108 tools", "GPU, 274 requests per bar")
    left.legend(frameon=False, loc="upper right")

    timing = [("All 108 tools\nin context", {"lfm3": "108_latency_greedy", "minicpm5": "108_latency_greedy"}),
              ("Harness\nsearches", {"lfm3": "deferred_forced_plain_latency_greedy",
                                     "minicpm5": "deferred_forced_latency_greedy"})]
    for k, (model, (label, color)) in enumerate(MODELS.items()):
        for i, (_, stems) in enumerate(timing):
            secs = laptop_seconds(model, stems[model])
            xpos = i + (k - 0.5) * width
            right.bar(xpos, secs, width, color=color)
            right.text(xpos, secs + 2, f"{secs:.0f} s", ha="center", color=INK, fontsize=9)
    right.set_xticks(range(len(timing)), [t[0] for t in timing], fontsize=9.5)
    right.set_ylabel("seconds per request")
    right.grid(axis="y", color=GRID)
    right.set_axisbelow(True)
    titles(right, "Time per request", "Laptop CPU, 30 requests per bar")
    save(fig, "fig5_deferred_tools")


if __name__ == "__main__":
    figure1()
    figure2()
    rows = gpu_rows()
    figure3(rows)
    figure4(rows)
    figure5(rows)
