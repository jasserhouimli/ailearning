from carseats_ml import evaluate, models


def _train(model, prepared):
    return models.fit_and_evaluate(
        model,
        prepared["X_train"],
        prepared["y_train"],
        prepared["X_test"],
        prepared["y_test"],
    )


def test_knn_reaches_a_reasonable_accuracy(prepared):
    accuracy, _ = _train(models.knn(5), prepared)
    assert accuracy > 0.7


def test_svm_kernel_sweep_covers_every_kernel(prepared):
    sweep = models.svm_kernel_sweep(
        prepared["X_train"], prepared["y_train"], prepared["X_test"], prepared["y_test"]
    )
    assert set(sweep) == {"linear", "poly", "rbf", "sigmoid"}
    assert max(sweep, key=sweep.get) == "linear"


def test_random_forest_importances_rank_price_first(prepared):
    model = models.random_forest(100).fit(prepared["X_train"], prepared["y_train"])
    ranking = evaluate.importances(model, prepared["feature_names"])
    assert ranking.iloc[0]["feature"] == "Price"
    assert abs(ranking["importance"].sum() - 1.0) < 1e-6


def test_comparison_table_is_sorted_by_accuracy():
    table = evaluate.comparison_table({"a": 0.7, "b": 0.9, "c": 0.8})
    assert table["model"].tolist() == ["b", "c", "a"]


def test_to_markdown_renders_a_table():
    markdown = evaluate.to_markdown({"svm": 0.883, "knn": 0.767})
    assert markdown.startswith("| Model | Accuracy |")
    assert "| svm | 0.883 |" in markdown
