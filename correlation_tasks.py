"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def _mri_correlation(group: pd.DataFrame) -> dict[str, float]:
    """Корреляция Пирсона каждого признака с MRI_Count (пропуски NA игнорируются попарно)."""
    return {f: float(group[f].corr(group["MRI_Count"], method="pearson")) for f in FEATURES}


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_corr = _mri_correlation(men)
    women_corr = _mri_correlation(women)

    strongest = max(
        [(abs(v), name) for corr in (women_corr, men_corr) for name, v in corr.items()]
    )[1]

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest,
    )
