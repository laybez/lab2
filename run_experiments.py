"""Воспроизводит все эксперименты и пишет experiment_results.json.

Запуск: python run_experiments.py
Модели сравниваются по ROC-AUC 5-fold CV на train; test используется только для итоговых метрик.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ensemble_models import CATEGORICAL, add_features, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

warnings.filterwarnings("ignore")
SEED = 42


def prep():
    return make_pipeline(
        FunctionTransformer(add_features),
        ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
            remainder="passthrough", verbose_feature_names_out=False,
        ),
    )


def rf(): return RandomForestClassifier(500, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=2)
def et(): return ExtraTreesClassifier(500, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=2)
def gb(): return GradientBoostingClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.7, random_state=SEED)
def hgb(): return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=200, max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)
def ada(): return AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=SEED)
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
def mlp(): return make_pipeline(StandardScaler(), MLPClassifier((32,), alpha=1.0, max_iter=2000, random_state=SEED))
def vote(names):
    f = {"rf": rf, "et": et, "gb": gb, "lr": lr, "mlp": mlp}
    return VotingClassifier([(n, f[n]()) for n in names], voting="soft")


EXPERIMENTS = {
    "Logistic Regression (baseline)": lr,
    "Random Forest": rf,
    "Extra Trees": et,
    "AdaBoost": ada,
    "Gradient Boosting": gb,
    "HistGradientBoosting": hgb,
    "MLP (32,)": mlp,
    "Voting RF+LR": lambda: vote(["rf", "lr"]),
    "Voting RF+GB+LR": lambda: vote(["rf", "gb", "lr"]),
    "Voting RF+GB+LR+MLP": lambda: vote(["rf", "gb", "lr", "mlp"]),
    "Voting RF+ET+GB+LR+MLP": lambda: vote(["rf", "et", "gb", "lr", "mlp"]),
}


def metrics(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba), "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred), "recall": recall_score(y, pred), "f1": f1_score(y, pred),
    }


def main():
    df = pd.read_csv("german.csv", sep=";")
    y = df.pop("Creditability")
    X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.25, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows = []
    for name, make in EXPERIMENTS.items():
        cv_proba = cross_val_predict(make_pipeline(prep(), make()), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        test_proba = make_pipeline(prep(), make()).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        rows.append({"model": name, "cv_roc_auc_train": roc_auc_score(y_tr, cv_proba), **metrics(y_te, test_proba)})
        print(f"{name:32s} CV={rows[-1]['cv_roc_auc_train']:.4f} test={rows[-1]['roc_auc']:.4f}", flush=True)

    res = run_ensemble_experiment(EnsembleInput(X_tr, y_tr, X_te, y_te, SEED))
    final = {k: float(getattr(res, k)) for k in ("roc_auc", "accuracy", "precision", "recall", "f1")}
    print("FINAL", res.model_name, final)

    out = {
        "random_seed": SEED, "split": "train_test_split(test_size=0.25, stratify=y)",
        "n_train": len(X_tr), "n_test": len(X_te),
        "selection_criterion": "ROC-AUC 5-fold stratified CV on train",
        "final_model": res.model_name, "final_test_metrics": final,
        "experiments": rows,
    }
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
