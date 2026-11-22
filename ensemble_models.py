"""Эксперимент с ансамблем: мягкое голосование RandomForest + GradientBoosting + LogReg."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

# Номинальные признаки: коды категорий, порядок у них не несёт смысла.
CATEGORICAL = [
    "Account_Balance", "Payment_Status_of_Previous_Credit", "Purpose",
    "Value_Savings_Stocks", "Length_of_current_employment", "Sex_Marital_Status",
    "Guarantors", "Most_valuable_available_asset", "Concurrent_Credits",
    "Type_of_apartment", "Occupation",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признаки по сумме, сроку и нагрузке кредита (без целевой переменной)."""
    df = df.copy()
    df["log_amount"] = np.log1p(df["Credit_Amount"])
    df["amount_per_month"] = df["Credit_Amount"] / df["Duration_of_Credit_monthly"]
    df["burden"] = df["amount_per_month"] * df["Instalment_per_cent"]
    return df


def build_model(seed: int):
    """Предобработка + soft-voting ансамбль. Все параметры заданы заранее (подбор по CV на train)."""
    prep = lambda: make_pipeline(  # noqa: E731
        FunctionTransformer(add_features),
        ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ),
    )
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=2
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.7, random_state=seed
    )
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    vote = VotingClassifier([("rf", rf), ("gb", gb), ("lr", lr)], voting="soft")
    return make_pipeline(prep(), vote)


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает ансамбль только на train и считает метрики на test."""
    X_train = pd.DataFrame(data.train_features)
    X_test = pd.DataFrame(data.test_features)
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    model = build_model(data.random_seed).fit(X_train, y_train)

    positive = list(model.classes_).index(1)
    proba = model.predict_proba(X_test)[:, positive]
    pred = (proba >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Soft Voting: RandomForest + GradientBoosting + LogisticRegression",
        probabilities=proba,
        predictions=pred,
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
    )
