import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("/tmp") / "binary_classifier_mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


RESULT_DIRS = [
    Path("result_AGP_gai"),
    Path("results_GGMP_gai"),
]
OUTPUT_DIR = Path("output_binary")
PAPER_BASELINE = 0.67


def load_analysis_df(result_dir: Path) -> tuple[pd.DataFrame, int, int, int]:
    result_path = result_dir / "result.tsv"
    df = pd.read_csv(result_path, sep="\t")
    total_rows = len(df)

    analysis_df = df[["id", "health", "corrected GAI"]].copy()
    analysis_df = analysis_df.dropna(subset=["corrected GAI", "health"]).copy()
    analysis_df = analysis_df[analysis_df["health"].isin(["y", "n"])].copy()
    analysis_df["y"] = (analysis_df["health"] == "n").astype(int)

    used_rows = len(analysis_df)
    dropped_rows = total_rows - used_rows
    class_count = int(analysis_df["y"].nunique())
    if class_count < 2:
        raise ValueError(f"{result_path} does not contain both classes after filtering")

    return analysis_df, total_rows, used_rows, dropped_rows


def fold_metrics_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def mean_std(metrics_list: list[dict[str, float]], key: str) -> tuple[float, float]:
    values = np.array([m[key] for m in metrics_list], dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


def evaluate_dataset(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df, total_rows, used_rows, dropped_rows = load_analysis_df(result_dir)
    dataset_output_dir = OUTPUT_DIR / result_dir.name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    x = analysis_df["corrected GAI"].to_numpy()
    y = analysis_df["y"].to_numpy()

    class_counts = analysis_df["y"].value_counts().sort_index().rename(index={0: "healthy", 1: "non_healthy"})
    class_counts.to_csv(dataset_output_dir / "class_counts.tsv", sep="\t", header=["count"])

    healthy_vals = analysis_df.loc[analysis_df["y"] == 0, "corrected GAI"].to_numpy()
    nonhealthy_vals = analysis_df.loc[analysis_df["y"] == 1, "corrected GAI"].to_numpy()

    class_summary = analysis_df.groupby("y")["corrected GAI"].agg(["count", "mean", "median", "std", "min", "max"])
    class_summary.index = ["healthy", "non_healthy"]
    class_summary.to_csv(dataset_output_dir / "class_summary.tsv", sep="\t")

    u_stat, p_value = mannwhitneyu(nonhealthy_vals, healthy_vals, alternative="two-sided")
    rank_biserial = (2 * u_stat) / (len(nonhealthy_vals) * len(healthy_vals)) - 1
    stats_df = pd.DataFrame(
        [
            {
                "dataset": result_dir.name,
                "rows_total": total_rows,
                "rows_used": used_rows,
                "rows_dropped": dropped_rows,
                "healthy_count": int((analysis_df["y"] == 0).sum()),
                "nonhealthy_count": int((analysis_df["y"] == 1).sum()),
                "mannwhitney_u": float(u_stat),
                "mannwhitney_pvalue": float(p_value),
                "rank_biserial": float(rank_biserial),
            }
        ]
    )
    stats_df.to_csv(dataset_output_dir / "dataset_stats.tsv", sep="\t", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    direction = 1.0 if nonhealthy_vals.mean() >= healthy_vals.mean() else -1.0
    x_direct_score = direction * x
    thr_direct = np.linspace(x_direct_score.min(), x_direct_score.max(), 201)
    mean_ba_direct = []

    for threshold in thr_direct:
        fold_bas = []
        for _, test_idx in cv.split(x.reshape(-1, 1), y):
            y_val = y[test_idx]
            y_pred = (x_direct_score[test_idx] >= threshold).astype(int)
            fold_bas.append(balanced_accuracy_score(y_val, y_pred))
        mean_ba_direct.append(float(np.mean(fold_bas)))

    best_thr_direct = float(thr_direct[int(np.argmax(mean_ba_direct))])
    best_thr_direct_gai = best_thr_direct / direction

    direct_fold_metrics = []
    direct_fold_aucs = []
    direct_oof_scores = np.zeros_like(x, dtype=float)
    direct_oof_pred = np.zeros_like(y, dtype=int)

    for _, test_idx in cv.split(x.reshape(-1, 1), y):
        y_val = y[test_idx]
        score_val = x_direct_score[test_idx]
        pred_val = (score_val >= best_thr_direct).astype(int)
        direct_fold_metrics.append(fold_metrics_from_labels(y_val, pred_val))
        direct_fold_aucs.append(float(roc_auc_score(y_val, score_val)))
        direct_oof_scores[test_idx] = score_val
        direct_oof_pred[test_idx] = pred_val

    thr_prob = np.linspace(0, 1, 201)
    mean_ba_prob = []

    for threshold in thr_prob:
        fold_bas = []
        for train_idx, test_idx in cv.split(x.reshape(-1, 1), y):
            x_train = x[train_idx].reshape(-1, 1)
            y_train = y[train_idx]
            x_val = x[test_idx].reshape(-1, 1)
            y_val = y[test_idx]

            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(x_train, y_train)
            prob_val = lr.predict_proba(x_val)[:, 1]
            y_pred = (prob_val >= threshold).astype(int)
            fold_bas.append(balanced_accuracy_score(y_val, y_pred))
        mean_ba_prob.append(float(np.mean(fold_bas)))

    best_thr_prob = float(thr_prob[int(np.argmax(mean_ba_prob))])

    logit_fold_metrics = []
    logit_fold_aucs = []
    logit_oof_probs = np.zeros_like(x, dtype=float)
    logit_oof_pred = np.zeros_like(y, dtype=int)
    coef_vals = []
    intercept_vals = []

    for train_idx, test_idx in cv.split(x.reshape(-1, 1), y):
        x_train = x[train_idx].reshape(-1, 1)
        y_train = y[train_idx]
        x_val = x[test_idx].reshape(-1, 1)
        y_val = y[test_idx]

        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(x_train, y_train)
        prob_val = lr.predict_proba(x_val)[:, 1]
        pred_val = (prob_val >= best_thr_prob).astype(int)

        logit_fold_metrics.append(fold_metrics_from_labels(y_val, pred_val))
        logit_fold_aucs.append(float(roc_auc_score(y_val, prob_val)))
        logit_oof_probs[test_idx] = prob_val
        logit_oof_pred[test_idx] = pred_val
        coef_vals.append(float(lr.coef_[0, 0]))
        intercept_vals.append(float(lr.intercept_[0]))

    comparison_rows = []

    ba_mean, ba_std = mean_std(direct_fold_metrics, "balanced_accuracy")
    sens_mean, sens_std = mean_std(direct_fold_metrics, "sensitivity")
    spec_mean, spec_std = mean_std(direct_fold_metrics, "specificity")
    comparison_rows.append(
        {
            "dataset": result_dir.name,
            "method": "direct_gai_threshold",
            "best_threshold": best_thr_direct_gai,
            "balanced_accuracy_mean": ba_mean,
            "balanced_accuracy_std": ba_std,
            "auc_mean": float(np.mean(direct_fold_aucs)),
            "auc_std": float(np.std(direct_fold_aucs, ddof=1)),
            "sensitivity_mean": sens_mean,
            "sensitivity_std": sens_std,
            "specificity_mean": spec_mean,
            "specificity_std": spec_std,
        }
    )

    ba_mean, ba_std = mean_std(logit_fold_metrics, "balanced_accuracy")
    sens_mean, sens_std = mean_std(logit_fold_metrics, "sensitivity")
    spec_mean, spec_std = mean_std(logit_fold_metrics, "specificity")
    mean_coef = float(np.mean(coef_vals))
    mean_intercept = float(np.mean(intercept_vals))
    implied_gai_cutoff = np.nan
    if abs(mean_coef) > 1e-10 and 0 < best_thr_prob < 1:
        logit_p = np.log(best_thr_prob / (1 - best_thr_prob))
        implied_gai_cutoff = float((logit_p - mean_intercept) / mean_coef)

    comparison_rows.append(
        {
            "dataset": result_dir.name,
            "method": "logistic_prob_threshold",
            "best_threshold": best_thr_prob,
            "balanced_accuracy_mean": ba_mean,
            "balanced_accuracy_std": ba_std,
            "auc_mean": float(np.mean(logit_fold_aucs)),
            "auc_std": float(np.std(logit_fold_aucs, ddof=1)),
            "sensitivity_mean": sens_mean,
            "sensitivity_std": sens_std,
            "specificity_mean": spec_mean,
            "specificity_std": spec_std,
            "mean_logistic_coef": mean_coef,
            "mean_logistic_intercept": mean_intercept,
            "implied_corrected_gai_cutoff": implied_gai_cutoff,
        }
    )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "balanced_accuracy_mean", ascending=False
    )
    comparison_df.to_csv(dataset_output_dir / "comparison.tsv", sep="\t", index=False)

    winner = comparison_df.iloc[0]
    winner_df = pd.DataFrame(
        [
            {
                "dataset": result_dir.name,
                "winning_method": winner["method"],
                "winning_balanced_accuracy_mean": winner["balanced_accuracy_mean"],
                "winning_auc_mean": winner["auc_mean"],
                "paper_baseline_balanced_accuracy": PAPER_BASELINE,
                "baseline_delta": winner["balanced_accuracy_mean"] - PAPER_BASELINE,
            }
        ]
    )
    winner_df.to_csv(dataset_output_dir / "winner.tsv", sep="\t", index=False)

    oof_df = pd.DataFrame(
        {
            "id": analysis_df["id"].to_numpy(),
            "health": analysis_df["health"].to_numpy(),
            "y_true": y,
            "corrected_GAI": x,
            "direct_score": direct_oof_scores,
            "direct_pred": direct_oof_pred,
            "logistic_prob": logit_oof_probs,
            "logistic_pred": logit_oof_pred,
        }
    )
    oof_df.to_csv(dataset_output_dir / "oof_predictions.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(healthy_vals, bins=30, alpha=0.6, label="healthy (0)", density=True)
    axes[0, 0].hist(nonhealthy_vals, bins=30, alpha=0.6, label="non-healthy (1)", density=True)
    axes[0, 0].set_title("Corrected GAI distribution by class")
    axes[0, 0].set_xlabel("corrected GAI")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend()

    axes[0, 1].boxplot(
        [healthy_vals, nonhealthy_vals],
        labels=["healthy (0)", "non-healthy (1)"],
        patch_artist=True,
    )
    axes[0, 1].set_title("Corrected GAI by class")
    axes[0, 1].set_ylabel("corrected GAI")

    axes[1, 0].plot(thr_direct / direction, mean_ba_direct, label="Method A: direct GAI threshold")
    axes[1, 0].axvline(best_thr_direct_gai, linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("direct threshold on corrected GAI")
    axes[1, 0].set_ylabel("mean CV balanced accuracy")
    ax2 = axes[1, 0].twinx()
    ax2.plot(thr_prob, mean_ba_prob, color="tab:orange", label="Method B: prob threshold")
    ax2.axvline(best_thr_prob, color="tab:orange", linestyle="--", linewidth=1)
    ax2.set_ylabel("mean CV balanced accuracy (logistic)")
    axes[1, 0].set_title("Threshold optimization landscapes")
    lines1, labels1 = axes[1, 0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 0].legend(lines1 + lines2, labels1 + labels2, loc="best")

    fpr_direct, tpr_direct, _ = roc_curve(y, direct_oof_scores)
    fpr_logit, tpr_logit, _ = roc_curve(y, logit_oof_probs)
    auc_direct_oof = roc_auc_score(y, direct_oof_scores)
    auc_logit_oof = roc_auc_score(y, logit_oof_probs)

    axes[1, 1].plot(fpr_direct, tpr_direct, label=f"Direct GAI (OOF AUC={auc_direct_oof:.3f})")
    axes[1, 1].plot(fpr_logit, tpr_logit, label=f"Logistic (OOF AUC={auc_logit_oof:.3f})")
    axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1, 1].set_title("ROC curves (out-of-fold)")
    axes[1, 1].set_xlabel("False positive rate")
    axes[1, 1].set_ylabel("True positive rate")
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(dataset_output_dir / "plots.png", dpi=200)
    plt.close(fig)

    print(result_dir.name)
    print(f"  rows used: {used_rows} / {total_rows}")
    print(f"  best method: {winner['method']}")
    print(f"  best BA: {winner['balanced_accuracy_mean']:.4f}")
    print(f"  best AUC: {winner['auc_mean']:.4f}")

    return comparison_df, winner_df


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_frames = []
    winner_frames = []

    for result_dir in RESULT_DIRS:
        comparison_df, winner_df = evaluate_dataset(result_dir)
        summary_frames.append(comparison_df)
        winner_frames.append(winner_df)

    pd.concat(summary_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "summary.tsv", sep="\t", index=False
    )
    pd.concat(winner_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "winners.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
