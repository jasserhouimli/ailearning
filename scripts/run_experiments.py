"""Reproduce every Carseats experiment and write the figures used in the README.

Run it from the repository root::

    python scripts/run_experiments.py

It prints the accuracy comparison as a Markdown table and saves the plots to
``reports/figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from carseats_ml import data, evaluate, models  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def save_bar(labels, values, title, ylabel, path, color="#1982C4", highlight=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [color] * len(labels)
    if highlight:
        colors[values.index(max(values))] = "#FF595E"
    bars = ax.bar([str(label) for label in labels], values, color=colors, width=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    prepared = data.prepare()
    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]

    scores = {
        "KNN (k=5)": models.fit_and_evaluate(models.knn(5), X_train, y_train, X_test, y_test)[0],
        "Random Forest (100)": models.fit_and_evaluate(
            models.random_forest(100), X_train, y_train, X_test, y_test
        )[0],
        "SVM (RBF, C=1)": models.fit_and_evaluate(
            models.svm("rbf", 1.0), X_train, y_train, X_test, y_test
        )[0],
    }

    print("## Model comparison\n")
    print(evaluate.to_markdown(scores))
    print(f"\nBest model: {max(scores, key=scores.get)}\n")
    save_bar(
        list(scores),
        list(scores.values()),
        "Model comparison",
        "Accuracy",
        FIGURES_DIR / "model_comparison.png",
        "#6A4C93",
        highlight=True,
    )

    tree_sweep = models.random_forest_sweep(X_train, y_train, X_test, y_test)
    print("## Random Forest: trees vs accuracy\n")
    print("\n".join(f"- {k} trees: {v:.4f}" for k, v in tree_sweep.items()))
    save_bar(
        list(tree_sweep),
        list(tree_sweep.values()),
        "Random Forest: n_estimators vs accuracy",
        "Accuracy",
        FIGURES_DIR / "rf_trees.png",
        "#1982C4",
    )

    kernel_sweep = models.svm_kernel_sweep(X_train, y_train, X_test, y_test)
    print("\n## SVM: kernel vs accuracy\n")
    print("\n".join(f"- {k}: {v:.4f}" for k, v in kernel_sweep.items()))
    save_bar(
        list(kernel_sweep),
        list(kernel_sweep.values()),
        "SVM: kernel vs accuracy",
        "Accuracy",
        FIGURES_DIR / "svm_kernels.png",
        "#2A9D8F",
        highlight=True,
    )

    c_sweep = models.svm_c_sweep(X_train, y_train, X_test, y_test)
    print("\n## SVM (RBF): C vs accuracy\n")
    print("\n".join(f"- C={k}: {v:.4f}" for k, v in c_sweep.items()))
    save_bar(
        [f"C={c}" for c in c_sweep],
        list(c_sweep.values()),
        "SVM (RBF): C vs accuracy",
        "Accuracy",
        FIGURES_DIR / "svm_c.png",
        "#F4A261",
    )

    rf_model = models.random_forest(100).fit(X_train, y_train)
    importances = evaluate.importances(rf_model, prepared["feature_names"])
    print("\n## Top features (Random Forest)\n")
    print(importances.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importances.head(10)[::-1]
    ax.barh(top["feature"], top["importance"], color="#A23B72")
    ax.set_title("Random Forest: top 10 features", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importances.png", dpi=120)
    plt.close(fig)

    _, rf_predictions = models.fit_and_evaluate(
        models.random_forest(100), X_train, y_train, X_test, y_test
    )
    print("\n## Classification report (Random Forest)\n")
    print(evaluate.report(y_test, rf_predictions))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=evaluate.confusion(y_test, rf_predictions),
        display_labels=evaluate.CLASS_LABELS,
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Random Forest: confusion matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_rf.png", dpi=120)
    plt.close(fig)

    cluster_frame = data.add_high_target(data.load_raw())
    cluster_features = data.split_features_target(cluster_frame)[0]
    clusters = models.kmeans(4).fit_predict(cluster_features)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        cluster_frame["Price"], cluster_frame["Income"], c=clusters, cmap="viridis"
    )
    ax.set_xlabel("Price")
    ax.set_ylabel("Income")
    ax.set_title("Store clusters (K-Means, k=4)", fontweight="bold")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kmeans_clusters.png", dpi=120)
    plt.close(fig)

    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
