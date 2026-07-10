# src/subsample_test.py

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

FEATURES_PATH = "outputs/features/full_features.csv"
RANDOM_STATE   = 42
CAP            = 15   # max songs kept per singer after subsampling


def load_features(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns
                    if c not in ["recording_id", "genre", "singer"]]
    return df, feature_cols


def subsample_dominant_singers(df, cap=CAP, random_state=RANDOM_STATE):
    """
    Caps every singer's song count at `cap`. Singers with fewer songs
    than the cap are kept as-is (untouched). Random subsample, not
    first-N, to avoid accidentally biasing toward whichever songs
    happen to be listed first in the CSV.
    """
    rng = np.random.default_rng(random_state)
    kept_rows = []
    for singer, group in df.groupby("singer"):
        if len(group) > cap:
            idx = rng.choice(group.index, size=cap, replace=False)
            kept_rows.append(df.loc[idx])
        else:
            kept_rows.append(group)
    return pd.concat(kept_rows).reset_index(drop=True)


def nested_loso_topk(X, y, singers, feature_names, k=25, n_estimators=200):
    """Same nested logic as classify.py - feature selection uses ONLY
    each fold's own training data, no leakage."""
    unique_singers = np.unique(singers)
    scaler = StandardScaler()
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    rf  = RandomForestClassifier(n_estimators=n_estimators,
                                  random_state=RANDOM_STATE,
                                  class_weight="balanced")

    all_preds = np.empty(len(y), dtype=object)
    fold_accs = []

    for singer in unique_singers:
        test_mask  = singers == singer
        train_mask = ~test_mask

        X_train_full, y_train = X[train_mask], y[train_mask]
        X_test_full = X[test_mask]

        rf.fit(X_train_full, y_train)
        top_idx = np.argsort(rf.feature_importances_)[::-1][:k]

        X_train = scaler.fit_transform(X_train_full[:, top_idx])
        X_test  = scaler.transform(X_test_full[:, top_idx])

        svm.fit(X_train, y_train)
        preds = svm.predict(X_test)
        all_preds[test_mask] = preds

        acc = accuracy_score(y[test_mask], preds)
        fold_accs.append(acc)
        print(f"  {singer[:30]:<30}  n={test_mask.sum():>2}  acc={acc:.3f}")

    print(f"\n  Mean : {np.mean(fold_accs):.3f}  Std  : {np.std(fold_accs):.3f}")
    return all_preds, fold_accs


def run():
    df, feature_names = load_features(FEATURES_PATH)

    print("="*50)
    print("BEFORE subsampling - dominant singers")
    print("="*50)
    print(df.groupby(["genre", "singer"]).size().sort_values(ascending=False).head(6))
    print(f"\nTotal songs: {len(df)}")

    df_sub = subsample_dominant_singers(df, cap=CAP)

    print(f"\n{'='*50}")
    print(f"AFTER capping every singer at {CAP} songs")
    print(f"{'='*50}")
    print(df_sub.groupby(["genre", "singer"]).size().sort_values(ascending=False).head(6))
    print(f"\nTotal songs: {len(df_sub)}  (was {len(df)})")
    print(df_sub["genre"].value_counts())

    X       = df_sub[feature_names].values
    y       = df_sub["genre"].values
    singers = df_sub["singer"].values

    print(f"\n{'='*50}")
    print(f"Nested LOSO Top-25 on SUBSAMPLED data (cap={CAP})")
    print(f"{'='*50}")
    preds, accs = nested_loso_topk(X, y, singers, feature_names, k=25)

    mask = preds != None
    print(f"\n{'='*50}")
    print("Classification Report - Subsampled")
    print(f"{'='*50}")
    print(classification_report(y[mask], preds[mask]))

    print(f"\n{'='*50}")
    print("COMPARISON")
    print(f"{'='*50}")
    print(f"  Original (110 songs, imbalanced), nested top-25 : 0.573  (your last run)")
    print(f"  Subsampled ({len(df_sub)} songs, cap={CAP}),        nested top-25 : {np.mean(accs):.3f}")


if __name__ == "__main__":
    run()