from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from telco_modeling import RAW_FEATURE_COLUMNS, engineer_features, validate_input

BASE_DIR = Path(__file__).parent
DEFAULT_ARTIFACT = BASE_DIR / 'telco_model_final.joblib'


def predict_record(
    record: dict[str, Any],
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    """Validate one raw record and return a human-review prediction."""
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f'No existe el artefacto: {artifact_path}')
    frame = pd.DataFrame([record]).copy()
    frame = frame.drop(columns=['Churn'], errors='ignore')
    frame = frame[[column for column in RAW_FEATURE_COLUMNS if column in frame.columns]]
    validation = validate_input(frame, allow_target=False, allow_extra=False)
    artifact = joblib.load(artifact_path)
    pipeline = artifact['pipeline']
    threshold = float(artifact['threshold'])
    prepared = engineer_features(frame[RAW_FEATURE_COLUMNS].copy())
    feature_columns = artifact.get('feature_columns') or prepared.columns.tolist()
    prepared = prepared.reindex(columns=feature_columns)
    probability = float(pipeline.predict_proba(prepared)[0, 1])
    classification = 'Riesgo de abandono' if probability >= threshold else 'Riesgo bajo'
    return {
        'probabilidad_abandono': round(probability, 6),
        'clasificacion': classification,
        'umbral': threshold,
        'modelo': artifact.get('model_name'),
        'version': artifact.get('model_version', '1.0'),
        'fecha_inferencia': date.today().isoformat(),
        'revision_humana': True,
        'validacion': validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Inferencia segura para Telco Churn.')
    parser.add_argument('input_json', type=Path, help='Archivo JSON con un registro raw.')
    parser.add_argument('--artifact', type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    record = json.loads(args.input_json.read_text(encoding='utf-8'))
    print(json.dumps(predict_record(record, args.artifact), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
