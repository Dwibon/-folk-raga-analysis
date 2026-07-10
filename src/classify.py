# src/classify.py

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── CONFIG ────────────────────────────────────────────────
FEATURES_PATH = "outputs/features/full_features.csv"
OUTPUT_DIR    = "outputs/"
RANDOM_STATE  = 42
# ─────────────────────────────────────────────────────────


def load_features(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns
                    if c not in ["recording_id", "genre", "singer"]]
    X       = df[feature_cols].values
    y       = df["genre"].values
    singers = df["singer"].values
    print(f"Feature matrix : {X.shape}")
    print(f"Classes        : {np.unique(y)}")
    print(f"Distribution   : {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, singers, df, feature_cols


def _loso_loop(X, y, singers, scaler, clf, show_per_singer=True):
    """Shared LOSO logic — scales, fits, predicts per singer fold."""
    unique_singers = np.unique(singers)
    all_preds      = np.empty(len(y), dtype=object)
    fold_accs      = []

    for singer in unique_singers:
        test_mask  = singers == singer
        train_mask = ~test_mask

        X_train = scaler.fit_transform(X[train_mask])
        X_test  = scaler.transform(X[test_mask])

        clf.fit(X_train, y[train_mask])
        preds = clf.predict(X_test)
        all_preds[test_mask] = preds

        acc = accuracy_score(y[test_mask], preds)
        fold_accs.append(acc)

        if show_per_singer:
            print(f"  {singer[:30]:<30}  "
                  f"n={test_mask.sum():>2}  acc={acc:.3f}")

    print(f"\n  Mean : {np.mean(fold_accs):.3f}  "
          f"Std  : {np.std(fold_accs):.3f}")
    return all_preds, fold_accs


def run_svm_loso(X, y, singers):
    print(f"\n{'='*50}")
    print("SVM — Leave-One-Singer-Out (all features)")
    print(f"{'='*50}")
    scaler = StandardScaler()
    clf    = SVC(kernel="rbf", C=1.0, gamma="scale",
                 random_state=RANDOM_STATE)
    return _loso_loop(X, y, singers, scaler, clf)


def run_svm_loso_pca(X, y, singers, n_components=15):
    print(f"\n{'='*50}")
    print(f"SVM + PCA({n_components}) — Leave-One-Singer-Out")
    print(f"{'='*50}")

    unique_singers = np.unique(singers)
    scaler = StandardScaler()
    pca    = PCA(n_components=n_components, random_state=RANDOM_STATE)
    clf    = SVC(kernel="rbf", C=1.0, gamma="scale",
                 random_state=RANDOM_STATE)

    all_preds = np.empty(len(y), dtype=object)
    fold_accs = []

    for singer in unique_singers:
        test_mask  = singers == singer
        train_mask = ~test_mask

        X_train_s = scaler.fit_transform(X[train_mask])
        X_test_s  = scaler.transform(X[test_mask])

        X_train_p = pca.fit_transform(X_train_s)
        X_test_p  = pca.transform(X_test_s)

        clf.fit(X_train_p, y[train_mask])
        preds = clf.predict(X_test_p)
        all_preds[test_mask] = preds

        acc = accuracy_score(y[test_mask], preds)
        fold_accs.append(acc)
        print(f"  {singer[:30]:<30}  "
              f"n={test_mask.sum():>2}  acc={acc:.3f}")

    print(f"\n  Mean : {np.mean(fold_accs):.3f}  "
          f"Std  : {np.std(fold_accs):.3f}")
    return all_preds, fold_accs


def run_random_forest_loso(X, y, singers, feature_names):
    print(f"\n{'='*50}")
    print("Random Forest — Leave-One-Singer-Out")
    print(f"{'='*50}")

    unique_singers  = np.unique(singers)
    clf             = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    all_preds       = np.empty(len(y), dtype=object)
    fold_accs       = []
    importances_all = []

    for singer in unique_singers:
        test_mask  = singers == singer
        train_mask = ~test_mask

        clf.fit(X[train_mask], y[train_mask])
        preds = clf.predict(X[test_mask])
        all_preds[test_mask] = preds

        acc = accuracy_score(y[test_mask], preds)
        fold_accs.append(acc)
        importances_all.append(clf.feature_importances_)
        print(f"  {singer[:30]:<30}  "
              f"n={test_mask.sum():>2}  acc={acc:.3f}")

    print(f"\n  Mean : {np.mean(fold_accs):.3f}  "
          f"Std  : {np.std(fold_accs):.3f}")

    mean_importances = np.mean(importances_all, axis=0)
    importance_df = pd.DataFrame({
        "feature"    : feature_names,
        "importance" : mean_importances
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 15 features:")
    print(importance_df.head(15).to_string(index=False))

    plot_feature_importance(
        importance_df,
        os.path.join(OUTPUT_DIR, "feature_importance.png")
    )
    return all_preds, fold_accs, importance_df


def run_topk_loso(X, y, singers, feature_names, importance_df,
                  k=20, show_per_singer=True):
    print(f"\n{'='*50}")
    print(f"SVM Top-{k} Features — Leave-One-Singer-Out")
    print(f"{'='*50}")

    top_features = importance_df.head(k)["feature"].tolist()
    feat_indices = [feature_names.index(f) for f in top_features]
    X_topk       = X[:, feat_indices]

    scaler = StandardScaler()
    clf    = SVC(kernel="rbf", C=1.0, gamma="scale",
                 random_state=RANDOM_STATE)
    return _loso_loop(X_topk, y, singers, scaler, clf,
                      show_per_singer=show_per_singer)


def run_topk_sweep(X, y, singers, feature_names, importance_df):
    print(f"\n{'='*50}")
    print("Top-K Feature Sweep — Finding Optimal K")
    print(f"{'='*50}")

    scaler         = StandardScaler()
    clf            = SVC(kernel="rbf", C=1.0, gamma="scale",
                         random_state=RANDOM_STATE)
    unique_singers = np.unique(singers)
    results        = []

    for k in [5, 10, 15, 20, 25, 30, 40, 50, 71]:
        top_features = importance_df.head(k)["feature"].tolist()
        feat_indices = [feature_names.index(f) for f in top_features]
        X_topk       = X[:, feat_indices]
        fold_accs    = []

        for singer in unique_singers:
            test_mask  = singers == singer
            train_mask = ~test_mask
            X_tr = scaler.fit_transform(X_topk[train_mask])
            X_te = scaler.transform(X_topk[test_mask])
            clf.fit(X_tr, y[train_mask])
            preds = clf.predict(X_te)
            fold_accs.append(accuracy_score(y[test_mask], preds))

        mean_acc = np.mean(fold_accs)
        results.append((k, mean_acc))
        print(f"  k={k:<3}  mean LOSO acc = {mean_acc:.3f}")

    best_k, best_acc = max(results, key=lambda x: x[1])
    print(f"\n  Best k = {best_k}  (acc = {best_acc:.3f})")

    # Plot k vs accuracy
    ks, accs = zip(*results)
    fig, ax  = plt.subplots(figsize=(8, 4))
    ax.plot(ks, accs, marker="o")
    ax.axvline(best_k, color="red", linestyle="--",
               label=f"Best k={best_k}")
    ax.set_xlabel("Number of features (k)")
    ax.set_ylabel("Mean LOSO Accuracy")
    ax.set_title("Feature Count vs LOSO Accuracy")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "topk_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return best_k


def run_ablation(df, y, singers):
    print(f"\n{'='*50}")
    print("Feature Ablation — LOSO per feature group")
    print(f"{'='*50}")

    spectral_cols = [
        "tempo",
        "centroid_mean", "centroid_std",
        "rolloff_mean",  "rolloff_std",
        "zcr_mean",      "zcr_std",
    ]
    groups = {
        "MFCC only"    : [c for c in df.columns if c.startswith("mfcc")],
        "Chroma only"  : [c for c in df.columns if c.startswith("chroma")],
        "Spectral only": [c for c in df.columns if c in spectral_cols],
        "All combined" : [c for c in df.columns
                          if c not in ["recording_id", "genre", "singer"]],
    }

    scaler         = StandardScaler()
    clf            = SVC(kernel="rbf", C=1.0, gamma="scale",
                         random_state=RANDOM_STATE)
    unique_singers = np.unique(singers)

    for group_name, cols in groups.items():
        X_group   = df[cols].values
        fold_accs = []
        for singer in unique_singers:
            test_mask  = singers == singer
            train_mask = ~test_mask
            X_tr = scaler.fit_transform(X_group[train_mask])
            X_te = scaler.transform(X_group[test_mask])
            clf.fit(X_tr, y[train_mask])
            preds = clf.predict(X_te)
            fold_accs.append(accuracy_score(y[test_mask], preds))
        print(f"  {group_name:<18} :  "
              f"mean={np.mean(fold_accs):.3f}  "
              f"std={np.std(fold_accs):.3f}")


def plot_confusion_matrix(y_true, y_pred, title, output_path):
    cm     = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_feature_importance(importance_df, output_path, top_n=20):
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1])
    ax.set_xlabel("Mean Importance (across LOSO folds)")
    ax.set_title(f"Top {top_n} Features — Random Forest LOSO")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def print_report(y_true, y_pred, title):
    mask = y_pred != None
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")
    print(classification_report(y_true[mask], y_pred[mask]))


def run(features_path=FEATURES_PATH):
    X, y, singers, df, feature_names = load_features(features_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. SVM baseline — all features
    svm_preds, _ = run_svm_loso(X, y, singers)
    print_report(y, svm_preds, "SVM LOSO Report (all features)")
    plot_confusion_matrix(
        y[svm_preds != None], svm_preds[svm_preds != None],
        "SVM LOSO — All Features",
        os.path.join(OUTPUT_DIR, "cm_svm_all.png")
    )

    # 2. SVM + PCA
    pca_preds, _ = run_svm_loso_pca(X, y, singers, n_components=15)
    print_report(y, pca_preds, "SVM + PCA(15) LOSO Report")
    plot_confusion_matrix(
        y[pca_preds != None], pca_preds[pca_preds != None],
        "SVM + PCA(15) LOSO",
        os.path.join(OUTPUT_DIR, "cm_svm_pca.png")
    )

    # 3. Random Forest + feature importance
    rf_preds, _, importance_df = run_random_forest_loso(
        X, y, singers, feature_names
    )
    print_report(y, rf_preds, "RF LOSO Report")
    plot_confusion_matrix(
        y[rf_preds != None], rf_preds[rf_preds != None],
        "Random Forest LOSO",
        os.path.join(OUTPUT_DIR, "cm_rf.png")
    )

    # 4. Sweep to find optimal k
    best_k = run_topk_sweep(X, y, singers, feature_names, importance_df)

    # 5. SVM with best k
    best_preds, _ = run_topk_loso(
        X, y, singers, feature_names, importance_df,
        k=best_k, show_per_singer=True
    )
    print_report(y, best_preds, f"SVM Top-{best_k} Features LOSO Report")
    plot_confusion_matrix(
        y[best_preds != None], best_preds[best_preds != None],
        f"SVM Top-{best_k} Features LOSO",
        os.path.join(OUTPUT_DIR, f"cm_svm_top{best_k}.png")
    )

    # 6. Ablation
    run_ablation(df, y, singers)

    # 7. Summary
    def acc(y_true, y_pred):
        mask = y_pred != None
        return accuracy_score(y_true[mask], y_pred[mask])

    print(f"\n{'='*50}")
    print("SUMMARY — Overall LOSO Accuracy")
    print(f"{'='*50}")
    print(f"  SVM all features       : {acc(y, svm_preds):.3f}")
    print(f"  SVM + PCA(15)          : {acc(y, pca_preds):.3f}")
    print(f"  Random Forest          : {acc(y, rf_preds):.3f}")
    print(f"  SVM Top-{best_k:<2} (best k)   : {acc(y, best_preds):.3f}")

    print("\nDone. Check outputs/ for all plots.")


if __name__ == "__main__":
    run()