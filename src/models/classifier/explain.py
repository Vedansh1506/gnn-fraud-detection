"""SHAP explanations for baseline scores (TreeExplainer).

Returns structured per-feature contributions. The plain-language wording that
TRD 5.1 shows in the /score response belongs with the API, where that response
shape is defined - this module stays presentation-agnostic so the same
contributions can feed the API, the dashboard, or the audit log.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import shap
import xgboost as xgb


@dataclass
class FeatureContribution:
    feature: str
    value: float | str
    contribution: float


def build_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """TreeExplainer is exact and fast on tree models - cheap enough to run per
    request at serving time, which is why TRD picks it over the sampling-based
    explainers."""
    return shap.TreeExplainer(model)


def explain_transactions(
    explainer: shap.TreeExplainer, features: pd.DataFrame, top_k: int = 5
) -> list[list[FeatureContribution]]:
    """Top-k features per row, ranked by absolute contribution.

    Sign is preserved in `contribution` (positive pushes the score toward
    "laundering"), so a caller can distinguish evidence for from evidence
    against - ranking by magnitude alone would hide that.
    """
    shap_values = explainer.shap_values(features)
    results = []
    for row_index in range(len(features)):
        row = shap_values[row_index]
        ranked = sorted(
            range(len(features.columns)), key=lambda i: abs(row[i]), reverse=True
        )[:top_k]
        results.append(
            [
                FeatureContribution(
                    feature=features.columns[i],
                    value=features.iloc[row_index, i],
                    contribution=float(row[i]),
                )
                for i in ranked
            ]
        )
    return results
