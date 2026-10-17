"""Model factories and training helpers for the Carseats experiments.

Every estimator is created through a small factory so the notebook, the
experiment script and the tests all build models the same way, and every model
is seeded for reproducibility.
"""

from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RANDOM_STATE = 0


def knn(n_neighbors: int = 5) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def random_forest(n_estimators: int = 100) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)


def svm(kernel: str = "rbf", C: float = 1.0, gamma: str = "scale") -> SVC:
    return SVC(kernel=kernel, C=C, gamma=gamma, random_state=RANDOM_STATE)


def kmeans(n_clusters: int = 4) -> KMeans:
    return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)


def fit_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Fit ``model`` and return ``(accuracy, predictions)`` on the test split."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions), predictions


def random_forest_sweep(
    X_train,
    y_train,
    X_test,
    y_test,
    tree_options: tuple[int, ...] = (10, 50, 100, 200, 500),
) -> dict[int, float]:
    """Accuracy as a function of the number of trees."""
    return {
        trees: fit_and_evaluate(random_forest(trees), X_train, y_train, X_test, y_test)[0]
        for trees in tree_options
    }


def svm_kernel_sweep(
    X_train,
    y_train,
    X_test,
    y_test,
    kernels: tuple[str, ...] = ("linear", "poly", "rbf", "sigmoid"),
) -> dict[str, float]:
    """Accuracy as a function of the SVM kernel (C kept at 1.0)."""
    return {
        kernel: fit_and_evaluate(svm(kernel=kernel), X_train, y_train, X_test, y_test)[0]
        for kernel in kernels
    }


def svm_c_sweep(
    X_train,
    y_train,
    X_test,
    y_test,
    c_values: tuple[float, ...] = (0.01, 0.1, 1, 10, 100),
) -> dict[float, float]:
    """Accuracy as a function of ``C`` for the RBF kernel."""
    return {
        c: fit_and_evaluate(svm(kernel="rbf", C=c), X_train, y_train, X_test, y_test)[0]
        for c in c_values
    }
