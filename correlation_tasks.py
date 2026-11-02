"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
import pandas as pd
 
from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput
 
FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
 
 
def _mri_correlation(group: pd.DataFrame) -> dict[str, float]:
    return {f: float(group[f].corr(group["MRI_Count"], method="pearson")) for f in FEATURES}
 
 
def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:

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

df = pd.read_csv('brainsize.txt', sep='\t', na_values='NA')
features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height', 'MRI_Count']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (gender, group) in zip(axes, df.groupby('Gender')):
    sns.heatmap(group[features].corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title(gender)
plt.show()

sns.pairplot(df, vars=features, hue='Gender')
plt.show()
