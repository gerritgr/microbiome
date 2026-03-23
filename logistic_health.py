from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RESULT_DIRS = [
    Path("result_AGP_gai"),
    Path("results_GGMP_gai"),
]
OUTPUT_DIR = Path("logistic_health_output")
TARGET_COL = "health"
FEATURE_COL = "corrected GAI"
LABEL_MAP = {"n": 0, "y": 1}


def load_dataset(result_dir: Path) -> pd.DataFrame:
    result_path = result_dir / "result.tsv"
    df = pd.read_csv(result_path, sep="\t")

    required_cols = {"id", TARGET_COL, FEATURE_COL}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{result_path} is missing columns: {sorted(missing)}")

    df = df.loc[:, ["id", TARGET_COL, FEATURE_COL]].copy()
    df = df.dropna(subset=[TARGET_COL, FEATURE_COL])
    df = df[df[TARGET_COL].isin(LABEL_MAP)].copy()
    df["health_label"] = df[TARGET_COL].map(LABEL_MAP)

    if df["health_label"].nunique() < 2:
        raise ValueError(f"{result_path} does not contain both health classes")

    return df


def fit_and_score(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    x = df[[FEATURE_COL]]
    y = df["health_label"]

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x, y)

    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)[:, 1]

    metrics = {
        "samples_total": int(len(df)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
    }

    predictions = pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "corrected GAI": x[FEATURE_COL].to_numpy(),
            "health_true": y.map({0: "n", 1: "y"}).to_numpy(),
            "health_pred": pd.Series(y_pred).map({0: "n", 1: "y"}).to_numpy(),
            "probability_y": y_prob,
        }
    )

    return metrics, predictions


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_rows = []

    for result_dir in RESULT_DIRS:
        df = load_dataset(result_dir)
        metrics, predictions = fit_and_score(df)

        dataset_output_dir = OUTPUT_DIR / result_dir.name
        dataset_output_dir.mkdir(exist_ok=True)

        pd.DataFrame([{"dataset": result_dir.name, **metrics}]).to_csv(
            dataset_output_dir / "metrics.tsv",
            sep="\t",
            index=False,
        )
        predictions.to_csv(
            dataset_output_dir / "predictions.tsv",
            sep="\t",
            index=False,
        )

        summary_rows.append({"dataset": result_dir.name, **metrics})

        print(result_dir.name)
        print(f"  accuracy: {metrics['accuracy']:.4f}")
        print(f"  balanced_accuracy: {metrics['balanced_accuracy']:.4f}")

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "summary.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
