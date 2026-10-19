#!/usr/bin/env python
"""Render the architecture diagram to docs/architecture.png.

Three horizontal lanes (modelling, CI/CD, serving/monitoring); every arrow is
short and arrows never cross, so the figure stays readable in a printed report.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "data": "#dbeafe",
    "ml": "#dcfce7",
    "ci": "#fef3c7",
    "serve": "#fce7f3",
    "obs": "#ede9fe",
    "user": "#e5e7eb",
    "lane": "#f9fafb",
}


def box(ax, x, y, w, h, text, kind, fs=8.5, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            fc=COLORS[kind],
            ec="#374151",
            lw=1.1,
        )
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            linespacing=1.35,
            weight=weight,
        )
    return (x, y, w, h)


def point(b, side):
    x, y, w, h = b
    return {
        "r": (x + w, y + h / 2),
        "l": (x, y + h / 2),
        "t": (x + w / 2, y + h),
        "b": (x + w / 2, y),
    }[side]


def arrow(ax, a, b, side_a, side_b, label=None, label_xy=None, ha="center"):
    p1, p2 = point(a, side_a), point(b, side_b)
    ax.add_patch(
        FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#111827")
    )
    if label:
        lx, ly = label_xy or ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.12)
        ax.text(lx, ly, label, ha=ha, fontsize=7.5, color="#374151", style="italic")


def lane(ax, y, h, title):
    ax.add_patch(
        FancyBboxPatch(
            (0.1, y),
            12.8,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.1",
            fc=COLORS["lane"],
            ec="#d1d5db",
            lw=1,
        )
    )
    ax.text(0.25, y + h - 0.22, title, fontsize=8.5, weight="bold", color="#6b7280")


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(-1.1, 8.4)
    ax.axis("off")
    ax.text(6.5, 8.15, "Heart Disease MLOps: architecture", ha="center", fontsize=14, weight="bold")

    # ------------------------------------------------ lane 1: data & modelling
    lane(ax, 5.1, 2.85, "1  DATA & MODELLING")
    uci = box(ax, 0.4, 6.0, 2.0, 1.1, "UCI Heart Disease\n(Cleveland, 303 rows)", "data")
    dl = box(
        ax,
        2.9,
        6.0,
        2.1,
        1.1,
        "Download + clean\nscripts/download_data.py\nsrc/heart/data.py",
        "ml",
        8,
    )
    eda = box(ax, 5.5, 6.0, 1.7, 1.1, "EDA\nscripts/eda.py\nreports/figures", "ml", 8)
    trn = box(ax, 7.7, 6.0, 2.3, 1.1, "Train, tune, CV\nsklearn Pipeline\nLR / RF / GB", "ml", 8)
    mlf = box(
        ax, 10.5, 6.0, 2.2, 1.1, "MLflow tracking\nparams, metrics,\nplots, registry", "obs", 8
    )
    art = box(ax, 7.7, 5.2, 2.3, 0.6, "model.joblib + metadata.json", "ml", 8)
    arrow(ax, uci, dl, "r", "l")
    arrow(ax, dl, eda, "r", "l")
    arrow(ax, eda, trn, "r", "l")
    arrow(ax, trn, mlf, "r", "l", "logs")
    arrow(ax, trn, art, "b", "t")

    # ---------------------------------------------------------- lane 2: CI/CD
    lane(ax, 2.55, 2.35, "2  CI/CD  (GitHub Actions: fails on any lint, test or build error)")
    lint = box(ax, 0.4, 2.75, 1.9, 1.0, "1  Lint\nflake8 + black", "ci")
    tst = box(ax, 2.8, 2.75, 2.0, 1.0, "2  Unit tests\npytest + coverage", "ci")
    ctr = box(ax, 5.3, 2.75, 2.0, 1.0, "3  Train job\ndata, EDA, MLflow", "ci")
    dkr = box(ax, 7.8, 2.75, 2.0, 1.0, "4  Docker build\n+ smoke test", "ci")
    reg = box(ax, 10.3, 2.75, 2.2, 1.0, "GHCR\ncontainer registry", "ci")
    arrow(ax, lint, tst, "r", "l")
    arrow(ax, tst, ctr, "r", "l")
    arrow(ax, ctr, dkr, "r", "l")
    arrow(ax, dkr, reg, "r", "l", "push")
    arrow(ax, art, dkr, "b", "t", "model baked\ninto image", label_xy=(8.95, 4.3), ha="left")

    # ----------------------------------------- lane 3: serving & monitoring
    lane(ax, -0.2, 2.5, "3  SERVING & MONITORING")
    k8s = box(ax, 8.4, 0.0, 4.4, 1.85, "", "serve")
    ax.text(
        10.6,
        1.62,
        "Kubernetes (Minikube / GKE / EKS / AKS)",
        ha="center",
        fontsize=8,
        weight="bold",
    )
    svc = box(ax, 8.6, 0.25, 1.6, 1.1, "Service\nLoadBalancer\n(+ Ingress)", "serve", 7.5)
    pods = box(
        ax,
        10.6,
        0.15,
        2.0,
        1.3,
        "Deployment, 2+ pods\n(HPA) FastAPI\n/predict /health\n/metrics + JSON logs",
        "serve",
        7.5,
    )
    arrow(ax, svc, pods, "r", "l")
    arrow(ax, reg, k8s, "b", "t", "image pull", label_xy=(11.5, 2.35), ha="left")

    prom = box(ax, 5.6, 0.35, 2.0, 0.9, "Prometheus\nscrapes /metrics", "obs")
    graf = box(ax, 2.9, 0.35, 2.0, 0.9, "Grafana\ndashboard", "obs")
    arrow(ax, k8s, prom, "l", "r", "scrape", label_xy=(8.0, 0.95), ha="center")
    arrow(ax, prom, graf, "l", "r", "query", label_xy=(5.25, 0.88), ha="center")

    cli = box(ax, 8.6, -0.95, 1.6, 0.55, "Client (curl / app)", "user", 7.5)
    arrow(ax, cli, svc, "t", "b", "HTTP + JSON", label_xy=(10.35, -0.25), ha="left")

    out = ROOT / "docs" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    sys.exit(main())
