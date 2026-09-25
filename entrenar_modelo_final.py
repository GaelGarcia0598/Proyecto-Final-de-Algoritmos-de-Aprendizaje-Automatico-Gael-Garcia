from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_modeling import (
    ENGINEERED_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TelcoFeatureEngineer,
    load_raw_csv,
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'Telco-Customer-Churn.csv'
ARTIFACT_PATH = BASE_DIR / 'telco_model_final.joblib'
METADATA_PATH = BASE_DIR / 'telco_model_metadata.json'
RANDOM_STATE = 42
SELECTED_THRESHOLD = 0.40

NUMERIC_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
    'has_internet', 'num_services', 'contract_short_term', 'new_customer',
    'monthly_to_tenure_ratio', 'total_over_tenure',
    'paperless_electronic_check'
]
CATEGORICAL_FEATURES = [
    column for column in RAW_FEATURE_COLUMNS + ENGINEERED_COLUMNS
    if column not in NUMERIC_FEATURES
]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), NUMERIC_FEATURES),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ('ingenieria', TelcoFeatureEngineer()),
        ('preprocesamiento', preprocessor),
        ('modelo', RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def metricas(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= SELECTED_THRESHOLD).astype(int)
    return {
        'Accuracy': float(accuracy_score(y_true, predictions)),
        'Precision': float(precision_score(y_true, predictions, zero_division=0)),
        'Recall': float(recall_score(y_true, predictions, zero_division=0)),
        'F1-score': float(f1_score(y_true, predictions, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(y_true, probabilities)),
    }


def main() -> None:
    X, y = load_raw_csv(DATA_PATH)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started

    validation_started = time.perf_counter()
    validation_prob = pipeline.predict_proba(X_validation)[:, 1]
    validation_seconds = time.perf_counter() - validation_started
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = metricas(y_test, test_prob)
    validation_metrics = metricas(y_validation, validation_prob)
    test_predictions = (test_prob >= SELECTED_THRESHOLD).astype(int)

    artifact = {
        'model_name': 'RandomForest',
        'model_version': '2.0',
        'pipeline': pipeline,
        'threshold': SELECTED_THRESHOLD,
        'feature_columns': RAW_FEATURE_COLUMNS,
        'target': 'Churn',
        'positive_class': 1,
        'created_at': '2026-09-23',
        'sklearn_version': __import__('sklearn').__version__,
    }
    joblib.dump(artifact, ARTIFACT_PATH)

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    metadata = {
        'model_name': 'RandomForest',
        'model_version': '2.0',
        'artifact_file': ARTIFACT_PATH.name,
        'dataset_file': DATA_PATH.name,
        'dataset_rows': int(len(X)),
        'dataset_columns': int(len(RAW_FEATURE_COLUMNS) + 2),
        'target': 'Churn',
        'positive_class': 1,
        'random_state': RANDOM_STATE,
        'split': {'train': 0.70, 'validation': 0.15, 'test': 0.15},
        'threshold': SELECTED_THRESHOLD,
        'training_seconds': training_seconds,
        'validation_inference_seconds': validation_seconds,
        'validation_metrics': validation_metrics,
        'test_metrics': test_metrics,
        'test_confusion_matrix': confusion_matrix(y_test, test_predictions).tolist(),
        'hyperparameters': {
            'n_estimators': 300,
            'min_samples_leaf': 3,
            'class_weight': 'balanced',
        },
        'raw_feature_columns': RAW_FEATURE_COLUMNS,
        'engineered_feature_columns': ENGINEERED_COLUMNS,
        'sklearn_version': __import__('sklearn').__version__,
        'serialization_warning': 'No cargues archivos Joblib/Pickle de fuentes desconocidas.',
        'dataset_sha256': sha256(DATA_PATH),
        'artifact_sha256': sha256(ARTIFACT_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Artifact: {ARTIFACT_PATH}')
    print(f'Metadata: {METADATA_PATH}')
    print(f'Test metrics: {test_metrics}')


if __name__ == '__main__':
    main()
