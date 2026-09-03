"""Reusable plotting helpers for the Telco Churn analysis.

All functions save figures to ``reports/figures/`` by default and support
both offline saving and inline rendering within Jupyter notebooks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

sns.set_theme(style="whitegrid", palette="muted")


def _prepare_output(fig: plt.Figure, name: str, output_dir: Path | None) -> None:
    """Save a figure to disk and optionally close it.

    Args:
        fig: Matplotlib figure to save.
        name: File name for the output figure.
        output_dir: Directory to save the figure. Defaults to ``reports/figures``.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")


def plot_target_distribution(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot the distribution of the target variable ``Churn``.

    Args:
        df: Dataframe containing the ``Churn`` column.
        output_dir: Optional directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["Churn"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Distribución de la variable objetivo (Churn)")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Número de clientes")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 20, f"{value} ({value / df.shape[0]:.1%})", ha="center")
    _prepare_output(fig, "target_distribution.png", output_dir)


def plot_numeric_distributions(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot KDE distributions of numeric features split by target.

    Args:
        df: Dataframe with numeric columns and the ``Churn`` column.
        output_dir: Optional directory to save the figure.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 4))
    for ax, col in zip(axes, numeric_cols, strict=False):
        sns.kdeplot(data=df, x=col, hue="Churn", fill=True, ax=ax)
        ax.set_title(f"Distribución de {col} por Churn")
    fig.tight_layout()
    _prepare_output(fig, "numeric_distributions.png", output_dir)


def plot_categorical_by_target(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot stacked bar charts of categorical features vs. target.

    Args:
        df: Dataframe with categorical columns and the ``Churn`` column.
        output_dir: Optional directory to save the figure.
    """
    cat_cols = df.select_dtypes(include="object").columns.difference(["customerID", "Churn"])
    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    axes = axes.flatten()
    for ax, col in zip(axes, cat_cols, strict=False):
        crosstab = pd.crosstab(df[col], df["Churn"], normalize="index")
        crosstab.plot(kind="bar", stacked=True, ax=ax, legend=False)
        ax.set_title(col, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_ylabel("Proporción")
    for ax in axes[len(cat_cols) :]:
        ax.axis("off")
    fig.tight_layout()
    _prepare_output(fig, "categorical_by_target.png", output_dir)


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot the correlation matrix of numeric features.

    Args:
        df: Dataframe with numeric columns.
        output_dir: Optional directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=False)
    ax.set_title("Matriz de correlación (variables numéricas)")
    _prepare_output(fig, "correlation_matrix.png", output_dir)