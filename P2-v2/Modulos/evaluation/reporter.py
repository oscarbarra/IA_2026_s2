"""Exportación de resultados y generación de gráficos.

Todos los artefactos se escriben en ``Graficos/``:
- Tabla comparativa en CSV y Markdown.
- Resultados completos de la validación cruzada.
- Matriz de confusión combinada e individual por experimento.
- Comparación de métricas y recall de validación cruzada.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Backend sin ventana, necesario para guardar PNG.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SUMMARY_COLUMNS = [
    "exp_id",
    "classifier",
    "compensation",
    "recall",
    "precision",
    "specificity",
    "f1",
    "accuracy",
    "fnr",
    "tn",
    "fp",
    "fn",
    "tp",
    "cv_recall_mean",
    "cv_recall_std",
    "cv_precision_mean",
    "cv_precision_std",
    "n_candidates",
    "elapsed_s",
    "best_params",
]


def _results_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convierte la lista de resultados en una tabla exportable."""
    rows = []
    for result in results:
        row = {column: result.get(column) for column in SUMMARY_COLUMNS}
        row["best_params"] = json.dumps(
            result.get("best_params", {}), default=str, ensure_ascii=False
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _to_markdown(df: pd.DataFrame) -> str:
    """Genera una tabla Markdown sin depender de ``tabulate``."""
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def save_results(results: list[dict], out_dir: str | Path) -> dict:
    """Guarda la tabla comparativa final en CSV y Markdown."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _results_dataframe(results)

    csv_path = out_dir / "resultados_experimentos.csv"
    md_path = out_dir / "resultados_experimentos.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(df), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "md": str(md_path),
        "dataframe": df,
    }


def save_cv_results(cv_frames: list[pd.DataFrame], out_dir: str | Path) -> str:
    """Concatena y guarda los resultados completos de todas las búsquedas."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "resultados_validacion_cruzada.csv"
    if cv_frames:
        pd.concat(cv_frames, ignore_index=True).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)
    return str(path)


def _draw_confusion_matrix(ax, result: dict) -> None:
    """Dibuja una matriz de confusión 2x2 sobre un eje existente."""
    cm = np.asarray(result["confusion_matrix"], dtype=int)

    ax.imshow(cm, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            value = cm[row, col]
            color = "white" if value > cm.max() / 2 else "black"
            ax.text(
                col,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["Real 0", "Real 1"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(
        f"{result['exp_id']} · {result['classifier']}\n"
        f"Recall={result['recall']:.3f} | Precision={result['precision']:.3f}",
        fontsize=9,
    )


def plot_confusion_matrices(
    results: list[dict],
    out_dir: str | Path,
    filename: str = "matrices_confusion.png",
) -> list[str]:
    """Genera una figura combinada y una imagen individual por experimento."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = out_dir / "matrices"
    individual_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []

    # Figura combinada.
    n_results = len(results)
    n_cols = min(4, max(1, n_results))
    n_rows = int(np.ceil(n_results / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, result in zip(axes.flat, results):
        ax.axis("on")
        _draw_confusion_matrix(ax, result)

    fig.suptitle(
        "Matrices de confusión sobre el conjunto de prueba (30%)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    combined_path = out_dir / filename
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(combined_path))

    # Imágenes individuales.
    for result in results:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        _draw_confusion_matrix(ax, result)
        fig.tight_layout()
        path = individual_dir / f"matriz_confusion_{result['exp_id']}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    return paths


def plot_metric_comparison(
    results: list[dict],
    out_dir: str | Path,
    filename: str = "comparacion_metricas.png",
) -> str:
    """Gráfico de barras agrupadas con recall, precisión, especificidad y F1."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("recall", "Sensibilidad (Recall)"),
        ("precision", "Precisión"),
        ("specificity", "Especificidad"),
        ("f1", "F1-Score"),
    ]
    labels = [result["exp_id"] for result in results]
    x = np.arange(len(results))
    width = 0.2
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for offset, ((key, label), color) in enumerate(zip(metrics, colors)):
        values = [result[key] for result in results]
        positions = x + (offset - 1.5) * width
        bars = ax.bar(positions, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas en el conjunto de prueba")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, loc="upper center", framealpha=0.95)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_cv_recall(
    results: list[dict],
    out_dir: str | Path,
    filename: str = "recall_validacion_cruzada.png",
) -> str:
    """Recall medio en CV=5 con barras de error (desviación estándar)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [result["exp_id"] for result in results]
    means = np.array([result["cv_recall_mean"] for result in results])
    stds = np.array([result["cv_recall_std"] for result in results])

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=4,
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
    )
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f"{mean:.3f} ± {std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, 1.35)
    ax.set_ylabel("Recall medio (clase fraude)")
    ax.set_title("Sensibilidad media en validación cruzada estratificada (k=5)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)
