from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

TARGET_COLUMN = 'Churn'
IDENTIFIER_COLUMN = 'customerID'
ENGINEERED_COLUMNS = [
    'has_internet', 'num_services', 'contract_short_term', 'new_customer',
    'monthly_to_tenure_ratio', 'total_over_tenure',
    'paperless_electronic_check'
]

YES_NO_COLUMNS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]

EXPECTED_CATEGORIES = {
    'gender': {'Female', 'Male'},
    'Partner': {'Yes', 'No'},
    'Dependents': {'Yes', 'No'},
    'PhoneService': {'Yes', 'No'},
    'MultipleLines': {'Yes', 'No', 'No phone service'},
    'InternetService': {'DSL', 'Fiber optic', 'No'},
    'OnlineSecurity': {'Yes', 'No', 'No internet service'},
    'OnlineBackup': {'Yes', 'No', 'No internet service'},
    'DeviceProtection': {'Yes', 'No', 'No internet service'},
    'TechSupport': {'Yes', 'No', 'No internet service'},
    'StreamingTV': {'Yes', 'No', 'No internet service'},
    'StreamingMovies': {'Yes', 'No', 'No internet service'},
    'Contract': {'Month-to-month', 'One year', 'Two year'},
    'PaperlessBilling': {'Yes', 'No'},
    'PaymentMethod': {
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)',
        'Credit card (automatic)'
    },
}

NUMERIC_COLUMNS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + ENGINEERED_COLUMNS


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build the model features from raw predictor columns."""
    frame = data.copy()
    frame['TotalCharges'] = pd.to_numeric(
        frame['TotalCharges'].replace(r'^\s*$', np.nan, regex=True),
        errors='coerce'
    )
    feature_map = {
        'Yes': 1,
        'No': 0,
        'No phone service': 0,
        'No internet service': 0,
    }
    frame['has_internet'] = (frame['InternetService'] != 'No').astype(int)
    frame['num_services'] = (
        frame[YES_NO_COLUMNS].replace(feature_map).astype(float).sum(axis=1)
    )
    frame['contract_short_term'] = (
        frame['Contract'] == 'Month-to-month'
    ).astype(int)
    frame['new_customer'] = (frame['tenure'] <= 12).astype(int)
    tenure = frame['tenure'].replace(0, np.nan)
    frame['monthly_to_tenure_ratio'] = frame['MonthlyCharges'] / tenure
    frame['total_over_tenure'] = frame['TotalCharges'] / tenure
    frame['paperless_electronic_check'] = (
        (frame['PaperlessBilling'] == 'Yes')
        & (frame['PaymentMethod'] == 'Electronic check')
    ).astype(int)
    return frame[FEATURE_COLUMNS]


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """scikit-learn transformer that keeps feature engineering in the pipeline."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'TelcoFeatureEngineer':
        validate_input(X, allow_target=False, allow_extra=False)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        validate_input(X, allow_target=False, allow_extra=False)
        return engineer_features(X)


def validate_input(
    data: pd.DataFrame,
    *,
    allow_target: bool = False,
    allow_extra: bool = False,
) -> dict[str, Any]:
    """Validate structure and values before feature engineering or inference."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError('La entrada debe ser un pandas.DataFrame.')

    required = RAW_FEATURE_COLUMNS.copy()
    if allow_target:
        required.append(TARGET_COLUMN)
    missing = [column for column in required if column not in data.columns]
    extra = [
        column for column in data.columns
        if column not in required and column != IDENTIFIER_COLUMN
    ]
    if missing:
        raise ValueError(f'Faltan columnas requeridas: {missing}')
    if extra and not allow_extra:
        raise ValueError(f'Hay columnas adicionales no permitidas: {extra}')

    issues: list[str] = []
    for column in ['SeniorCitizen', 'tenure', 'MonthlyCharges']:
        numeric_values = pd.to_numeric(data[column], errors='coerce')
        if numeric_values.isna().any() and data[column].notna().any():
            issues.append(f'{column} contiene valores no numericos.')
    if (pd.to_numeric(data['SeniorCitizen'], errors='coerce').dropna() > 1).any():
        issues.append('SeniorCitizen solo puede tomar valores 0 o 1.')
    if (pd.to_numeric(data['tenure'], errors='coerce').dropna() < 0).any():
        issues.append('tenure no puede ser negativo.')
    for column in ['MonthlyCharges', 'TotalCharges']:
        numeric_values = pd.to_numeric(data[column], errors='coerce').dropna()
        if (numeric_values < 0).any():
            issues.append(f'{column} no puede ser negativo.')

    unknown_categories = {}
    for column, categories in EXPECTED_CATEGORIES.items():
        observed = set(data[column].dropna().astype(str).unique())
        unknown = sorted(observed - categories)
        if unknown:
            unknown_categories[column] = unknown

    if issues:
        raise ValueError('; '.join(issues))
    return {
        'valid': True,
        'missing_columns': missing,
        'extra_columns': extra,
        'unknown_categories': unknown_categories,
        'row_count': len(data),
    }


def load_raw_csv(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    validate_input(frame, allow_target=True, allow_extra=False)
    target = frame[TARGET_COLUMN].map({'No': 0, 'Yes': 1})
    if target.isna().any():
        raise ValueError('Churn contiene categorias distintas de No/Yes.')
    return frame[RAW_FEATURE_COLUMNS].copy(), target.astype(int)
