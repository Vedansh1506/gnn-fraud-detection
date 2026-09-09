import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.classifier.explain import build_explainer, explain_transactions


def _trained_model() -> tuple[xgb.XGBClassifier, pd.DataFrame]:
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "strong_signal": rng.normal(size=200),
            "weak_signal": rng.normal(size=200),
            "noise": rng.normal(size=200),
        }
    )
    labels = (features["strong_signal"] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(features, labels)
    return model, features


def test_returns_top_k_per_row():
    model, features = _trained_model()
    explanations = explain_transactions(build_explainer(model), features.head(5), top_k=2)

    assert len(explanations) == 5
    assert all(len(row) == 2 for row in explanations)


def test_contributions_are_sorted_by_absolute_magnitude():
    model, features = _trained_model()
    explanations = explain_transactions(build_explainer(model), features.head(3), top_k=3)

    for row in explanations:
        magnitudes = [abs(c.contribution) for c in row]
        assert magnitudes == sorted(magnitudes, reverse=True)


def test_the_driving_feature_surfaces_first():
    model, features = _trained_model()
    explanations = explain_transactions(build_explainer(model), features.head(10), top_k=1)

    top_features = [row[0].feature for row in explanations]
    assert all(feature == "strong_signal" for feature in top_features)


def test_contribution_sign_is_preserved():
    """Direction matters - a caller must be able to tell evidence for from
    evidence against, which ranking by magnitude alone would hide."""
    model, features = _trained_model()
    explanations = explain_transactions(build_explainer(model), features.head(50), top_k=1)

    contributions = [row[0].contribution for row in explanations]
    assert any(c > 0 for c in contributions)
    assert any(c < 0 for c in contributions)
